from glob import glob
import os
import re
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
import losses
from pose_dataset import Batch

mean_pixel = [123.68, 116.779, 103.939]
net_funcs = {'resnet_50': resnet_v1.resnet_v1_50,
             'resnet_101': resnet_v1.resnet_v1_101}

def get_batch_spec(cfg):
    num_joints = cfg.num_joints
    batch_size = cfg.batch_size
    batch_spec = {
        Batch.inputs: [batch_size, None, None, 3],
        Batch.part_score_targets: [batch_size, None, None, 1],
        Batch.part_score_weights: [batch_size, None, None, 1],
        Batch.pose_target: [batch_size, None, None, 1]
    }
    if cfg.location_refinement:
        batch_spec[Batch.locref_targets] = [batch_size, None, None, num_joints * 2]
        batch_spec[Batch.locref_mask] = [batch_size, None, None, num_joints * 2]

    return batch_spec


def prediction_layer(cfg, input, name, num_outputs):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='SAME',
                        activation_fn=None, normalizer_fn=None,
                        weights_regularizer=slim.l2_regularizer(cfg.weight_decay)):
        with tf.variable_scope(name):
            pred = slim.conv2d_transpose(input, num_outputs,
                                         kernel_size=[3, 3], stride=2,
                                         scope='block4')
            return pred


def _leaky_relu(x):
    return tf.where(tf.greater(x,0),x,0.01*x)


class pose_gan(object):

    class resblock(object):
        def __init__(self,channel,name,ac_fn = tf.nn.relu,weight_decay=0.0005):
            self.ac_fn = ac_fn
            self.name = name
            self.weight_decay = weight_decay
            self.ch = channel
        def __call__(self,x):
            with slim.arg_scope([slim.conv2d],padding = 'SAME',biases_initializer = tf.zeros_initializer(),
                weights_initializer = tf.truncated_normal_initializer(stddev=0.01), activation_fn = self.ac_fn,
                weights_regularizer = slim.l2_regularizer(self.weight_decay)):

                out = slim.conv2d(x,self.ch,[3,3],scope = self.name + '_conv1')
                out = slim.conv2d(out,self.ch,[3,3],scope = self.name + '_conv2')
                out += x
            return out

    # class batch_norm(object):
    #     def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    #         with tf.variable_scope(name):
    #             self.epsilon  = epsilon
    #             self.momentum = momentum
    #             self.name = name
    #
    #      def __call__(self, x, train=True):
    #         return tf.contrib.layers.batch_norm(x,
    #                           decay=self.momentum,
    #                           updates_collections=None,
    #                           epsilon=self.epsilon,
    #                           scale=True,
    #                           is_training=train,
    #                           scope=self.name)

    def __init__(self,cfg):
        self.cfg = cfg
        #
        # self.d_bn1 = batch_norm(name = 'd_bn1')
        # self.d_bn2 = batch_norm(name = 'd_bn2')
        # self.d_bn3 = batch_norm(name = 'd_bn3')
        # self.d_bn4 = batch_norm(name = 'd_bn4')
        # self.d_bn5 = batch_norm(name = 'd_bn5')
        # self.d_bn6 = batch_norm(name = 'd_bn6')

        self.g_e_res1 = self.resblock(64,'g_e_res1')
        self.g_e_res2 = self.resblock(128,'g_e_res2')
        self.g_e_res3 = self.resblock(256,'g_e_res3')
        self.g_e_res4 = self.resblock(384,'g_e_res4')
        self.g_e_res5 = self.resblock(512,'g_e_res5')
        self.g_e_res6 = self.resblock(640,'g_e_res6')

        self.g_d_res1 = self.resblock(640,'g_d_res1')
        self.g_d_res2 = self.resblock(512,'g_d_res2')
        self.g_d_res3 = self.resblock(384,'g_d_res3')
        self.g_d_res4 = self.resblock(256,'g_d_res4')
        self.g_d_res5 = self.resblock(128,'g_d_res5')
        self.g_d_res6 = self.resblock(64,'g_d_res6')

    def part_detection_loss(self, heads, batch, locref, intermediate):
        cfg = self.cfg
        part_score_weights = 1.0

        def add_part_loss(pred_layer):
            return tf.losses.sigmoid_cross_entropy(batch[Batch.part_score_targets],
                                                   heads[pred_layer],
                                                   part_score_weights)

        loss = {}
        loss['part_loss'] = add_part_loss('part_pred')
        total_loss = loss['part_loss']
        if intermediate:
            loss['part_loss_interm'] = add_part_loss('part_pred_interm')
            total_loss = total_loss + loss['part_loss_interm']

        if locref:
            locref_pred = heads['locref']
            locref_targets = batch[Batch.locref_targets]
            locref_weights = batch[Batch.locref_mask]

            loss_func = losses.huber_loss if cfg.locref_huber_loss else tf.losses.mean_squared_error
            loss['locref_loss'] = cfg.locref_loss_weight * loss_func(locref_targets, locref_pred, locref_weights)
            total_loss = total_loss + loss['locref_loss']

        # loss['total_loss'] = slim.losses.get_total_loss(add_regularization_losses=params.regularize)
        loss['total_loss'] = total_loss
        return loss


    def extract_features(self, inputs):
        net_fun = net_funcs[self.cfg.net_type]

        mean = tf.constant(mean_pixel,
                           dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        im_centered = inputs - mean

        with slim.arg_scope(resnet_v1.resnet_arg_scope(False)):
            net, end_points = net_fun(im_centered,
                                      global_pool=False, output_stride=16)

        return net, end_points


    def prediction_layers(self,features,end_points,reuse=None, no_interm=False, scope='pose'):
        cfg = self.cfg

        num_layers = re.findall("resnet_([0-9]*)", cfg.net_type)[0]
        layer_name = 'resnet_v1_{}'.format(num_layers) + '/block{}/unit_{}/bottleneck_v1'

        out = {}

        with tf.variable_scope(scope, reuse=reuse):
            out['part_pred'] = prediction_layer(cfg, features, 'part_pred',
                                                1)
            if cfg.location_refinement:
                out['locref'] = prediction_layer(cfg, features, 'locref_pred',
                                                 cfg.num_joints * 2)
            if cfg.intermediate_supervision and not no_interm:
                interm_name = layer_name.format(3, cfg.intermediate_supervision_layer)
                block_interm_out = end_points[interm_name]
                out['part_pred_interm'] = prediction_layer(cfg, block_interm_out,
                                                           'intermediate_supervision',
                                                           1)
        return out


    def get_net(self,inputs):
        net, end_points = self.extract_features(inputs)
        return self.prediction_layers(net, end_points)


    def test(self, inputs):
        heads = self.get_net(inputs)
        return self.add_test_layers(heads)


    def add_test_layers(self, heads):
        prob = tf.sigmoid(heads['part_pred'])
        outputs = {'part_prob': prob}
        if self.cfg.location_refinement:
            outputs['locref'] = heads['locref']
        return outputs

    def generator(self,inputs,heat, reuse = None):
        inputs = tf.image.resize_images(inputs,(320,128))
        heat = tf.image.resize_images(heat,(320,128))
        x = tf.concat([inputs,heat],3)
        with tf.variable_scope('generator',reuse = reuse) as net_scope:
            if reuse:
                net_scope.reuse_variables()
            with slim.arg_scope([slim.conv2d,slim.fully_connected,slim.conv2d_transpose],
                biases_initializer = tf.zeros_initializer(),
                weights_initializer = tf.truncated_normal_initializer(stddev=0.01), activation_fn = None,
                weights_regularizer = slim.l2_regularizer(0.0005)):
                with slim.arg_scope([slim.conv2d,slim.conv2d_transpose], padding = 'SAME'):
                    out_from_e_1 = self.g_e_res1(slim.conv2d(x,64,[3,3],scope= 'g_conv1'))
                    out_from_e_2 = self.g_e_res2(slim.conv2d(out_from_e_1,128,[3,3],stride = 2,scope = 'g_conv2'))
                    out_from_e_3 = self.g_e_res3(slim.conv2d(out_from_e_2,256,[3,3],stride = 2,scope = 'g_conv3'))
                    out_from_e_4 = self.g_e_res4(slim.conv2d(out_from_e_3,384,[3,3],stride = 2,scope = 'g_conv4'))
                    out_from_e_5 = self.g_e_res5(slim.conv2d(out_from_e_4,512,[3,3],stride = 2,scope = 'g_conv5'))
                    out_from_e_6 = self.g_e_res6(slim.conv2d(out_from_e_5,640,[3,3],stride = 2,scope = 'g_conv6'))
                    out_from_e =  slim.conv2d(out_from_e_6,640,[3,3],scope = 'g_conv7')

                    out_from_e_ = slim.flatten(out_from_e)
                    out_from_fc = slim.fully_connected(out_from_e_,64,scope = 'g_fc1')
                    out_from_fc = slim.fully_connected(out_from_fc,int(out_from_e_.shape[1]),scope ='g_fc2')
                    out_from_fc = tf.reshape(out_from_fc,tf.shape(out_from_e))

                    out_from_d_1 = slim.conv2d_transpose(self.g_d_res1(out_from_fc+out_from_e_6),512,[3,3],stride=2,scope='g_dconv1')
                    out_from_d_2 = slim.conv2d_transpose(self.g_d_res2(out_from_d_1+out_from_e_5),384,[3,3],stride=2,scope='g_dconv2')
                    out_from_d_3 = slim.conv2d_transpose(self.g_d_res3(out_from_d_2+out_from_e_4),256,[3,3],stride=2,scope='g_dconv3')
                    out_from_d_4 = slim.conv2d_transpose(self.g_d_res4(out_from_d_3+out_from_e_3),128,[3,3],stride=2,scope='g_dconv4')
                    out_from_d_5 = slim.conv2d_transpose(self.g_d_res5(out_from_d_4+out_from_e_2),64,[3,3],stride=2,scope='g_dconv5')
                    out_from_d_6 = slim.conv2d_transpose(self.g_d_res6(out_from_d_5+out_from_e_1),1,[3,3],scope='g_dconv6')

                    # output = tf.nn.sigmoid(out_from_d_6)
                    output = out_from_d_6
                    return output


    def discriminator(self,inputs,structure,reuse = None):
        inputs = tf.image.resize_images(inputs,(320,128))
        structure = tf.image.resize_images(structure,(320,128))
        x = tf.concat([inputs,structure],3)
        ndf = 64
        with tf.variable_scope('discriminator',reuse = reuse) as net_scope:
            if reuse:
                net_scope.reuse_variables()
            with slim.arg_scope([slim.conv2d,slim.fully_connected],
                weights_initializer = tf.truncated_normal_initializer(stddev=0.01), activation_fn = _leaky_relu,
                weights_regularizer = slim.l2_regularizer(0.0005),
                biases_initializer = tf.zeros_initializer()):
                with slim.arg_scope([slim.conv2d], padding = 'SAME',normalizer_fn = slim.batch_norm,
                normalizer_params={'is_training':True,'decay':0.5}):
                    out = slim.conv2d(x,ndf,[4,4],stride = 2,scope = 'd_conv1')
                    out = slim.conv2d(out,ndf*2,[4,4],stride = 2,scope = 'd_conv2')
                    out = slim.conv2d(out,ndf*4,[4,4],stride = 2,scope ='d_conv3')
                    out = slim.conv2d(out,ndf*8,[4,4],stride = 2,scope ='d_conv4')
                    out = slim.conv2d(out,ndf*16,[4,4],stride = 2,scope ='d_conv5')

                    out = slim.flatten(out)
                    logits = slim.fully_connected(out,1,activation_fn = None,scope = 'd_fc')

                    return logits,tf.nn.sigmoid(logits)


    def train(self,batch):
        cfg = self.cfg
        intermediate = cfg.intermediate_supervision
        locref = cfg.location_refinement
        heads = self.get_net(batch[Batch.inputs])
        loss_inter = self.part_detection_loss(heads, batch, locref, intermediate)

        # get adversarial losses and reconstruction loss
        # structure_hat = self.generator(batch[Batch.inputs],heads['part_pred'])
        structure_hat = self.generator(batch[Batch.inputs],batch[Batch.part_score_targets])

        d_real_logits,d_real = self.discriminator(batch[Batch.inputs],batch[Batch.pose_target])
        d_fake_logits,d_fake = self.discriminator(batch[Batch.inputs],structure_hat,reuse = True)
        loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits = d_fake_logits, labels = tf.ones_like(d_fake)))
        loss_D = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits = d_fake_logits, labels = tf.zeros_like(d_fake))) + \
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits = d_real_logits, labels = tf.ones_like(d_real))) ) * 0.5

        # loss_rec = losses.huber_los(tf.image.resize_images(batch[Batch.pose_target],(320,128)),
        #             tf.image.resize_images(structure_hat,(320,128)))
        # loss_rec = tf.losses.absolute_difference(tf.image.resize_images(batch[Batch.pose_target],(320,128)),
        #                 tf.image.resize_images(structure_hat,(320,128)))
        loss_rec = tf.losses.mean_squared_error(tf.image.resize_images(batch[Batch.pose_target],(320,128)),
                        tf.image.resize_images(structure_hat,(320,128)))
        return loss_inter, loss_G, loss_D , loss_rec, heads, d_real, d_fake,structure_hat
