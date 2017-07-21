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
        Batch.part_score_targets: [batch_size, None, None, num_joints],
        Batch.part_score_weights: [batch_size, None, None, num_joints],
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

    class stacked_hourglass():
        def __init__(self, nb_stack, name='stacked_hourglass',num_joints = 14):
            self.nb_stack = nb_stack
            self.name = name
            self.num_joints = num_joints

        def __call__(self, inputs,heat,reuse = False,stride = 8):
            heat = tf.image.resize_nearest_neighbor(heat,tf.shape(heat)[1:3]*stride)
            x = tf.concat([inputs,heat],3)

            with tf.variable_scope(self.name,reuse = reuse) as scope:
                if reuse:
                    scope.reuse_variables()
                padding = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], name='padding')
                with tf.variable_scope("preprocessing") as sc:
                    conv1 = self._conv(padding, 64, 7, 2, 'VALID', 'conv1')
                    norm1 = tf.contrib.layers.batch_norm(conv1, 0.9, epsilon=1e-5,
                                        activation_fn=tf.nn.relu, scope=sc)
                    r1 = self._residual_block(norm1, 128, 'r1')
                    pool = tf.contrib.layers.max_pool2d(r1, [2,2], [2,2], 'VALID', scope=scope.name)
                    r2 = self._residual_block(pool, 128, 'r2')
                    pool_1 = tf.contrib.layers.max_pool2d(r2,[2,2],[2,2], 'VALID', scope=scope.name)
                    r2_1 = self._residual_block(pool_1, 128, 'r2_1')
                    r3 = self._residual_block(r2_1, 256, 'r3')
                hg = [None] * self.nb_stack
                ll = [None] * self.nb_stack
                ll_ = [None] * self.nb_stack
                out = [None] * self.nb_stack
                out_ = [None] * self.nb_stack
                sum_ = [None] * self.nb_stack
                with tf.variable_scope('_hourglass_0_with_supervision') as sc:
                    hg[0] = self._hourglass(r3, 4, 256, '_hourglass')
                    ll[0] = self._conv_bn_relu(hg[0], 256, name='conv_1')
                    ll_[0] = self._conv(ll[0],256,1,1,'VALID','ll')
                    out[0] = self._conv(ll[0],self.num_joints,1,1,'VALID','out')
                    out_[0] = self._conv(out[0],256,1,1,'VALID','out_')
                    sum_[0] = tf.add_n([ll_[0], out_[0], r3])
                for i in range(1, self.nb_stack - 1):
                    with tf.variable_scope('_hourglass_' + str(i) + '_with_supervision') as sc:
                        hg[i] = self._hourglass(sum_[i-1], 4, 256, '_hourglass')
                        ll[i] = self._conv_bn_relu(hg[i], 256, name='conv_1')
                        ll_[i] = self._conv(ll[i],256,1,1,'VALID','ll')
                        out[i] = self._conv(ll[i],self.num_joints,1,1,'VALID','out')
                        out_[i] = self._conv(out[i],256,1,1,'VALID','out_')
                        sum_[i] = tf.add_n([ll_[i], out_[i], sum_[i-1]])
                with tf.variable_scope('_hourglass_' + str(self.nb_stack - 1) + '_with_supervision') as sc:
                    hg[self.nb_stack-1] = self._hourglass(sum_[self.nb_stack - 2], 4, 256, '_hourglass')
                    ll[self.nb_stack-1] = self._conv_bn_relu(hg[self.nb_stack - 1], 256, name='conv_1')
                    out[self.nb_stack-1] = self._conv(ll[self.nb_stack-1],self.num_joints,1,1,'VALID','out')
                return tf.stack(out)

        def _conv(self, inputs, nb_filter, kernel_size=1, strides=1, pad='VALID', name='conv'):
            with tf.variable_scope(name) as scope:
                kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,\
                                        kernel_size,inputs.get_shape().as_list()[3],nb_filter]), name='weights')
                conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding=pad, data_format='NHWC')
                return conv

        def _conv_bn_relu(self, inputs, nb_filter, kernel_size=1, strides=1, name=None):
             with tf.variable_scope(name) as scope:
                kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,\
                                        kernel_size,inputs.get_shape().as_list()[3],nb_filter]), name='weights')
                conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding='SAME', data_format='NHWC')
                norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu, scope=scope.name)
                return norm

        def _conv_block(self, inputs, nb_filter_out, name='_conv_block'):
            with tf.variable_scope(name) as scope:
                with tf.variable_scope('norm_conv1') as sc:
                    norm1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5,
                                        activation_fn=tf.nn.relu, scope=sc)
                    conv1 = self._conv(norm1, nb_filter_out / 2, 1, 1, 'SAME', name='conv1')
                with tf.variable_scope('norm_conv2') as sc:
                    norm2 = tf.contrib.layers.batch_norm(conv1, 0.9, epsilon=1e-5,
                                        activation_fn=tf.nn.relu, scope=sc)
                    conv2 = self._conv(norm2, nb_filter_out / 2, 3, 1, 'SAME', name='conv2')
                with tf.variable_scope('norm_conv3') as sc:
                    norm3 = tf.contrib.layers.batch_norm(conv2, 0.9, epsilon=1e-5,
                                        activation_fn=tf.nn.relu, scope=sc)
                    conv3 = self._conv(norm3, nb_filter_out, 1, 1, 'SAME', name='conv3')
                return conv3

        def _skip_layer(self, inputs, nb_filter_out, name='_skip_layer'):
            if inputs.get_shape()[3].__eq__(tf.Dimension(nb_filter_out)):
                return inputs
            else:
                with tf.variable_scope(name) as scope:
                    conv = self._conv(inputs, nb_filter_out, 1, 1, 'SAME', name='conv')
                    return conv

        def _residual_block(self, inputs, nb_filter_out, name='_residual_block'):
            with tf.variable_scope(name) as scope:
                _conv_block = self._conv_block(inputs, nb_filter_out)
                _skip_layer = self._skip_layer(inputs, nb_filter_out)
                return tf.add(_skip_layer, _conv_block)

        def _hourglass(self, inputs, n, nb_filter_res, name='_hourglass'):
            with tf.variable_scope(name) as scope:
                # Upper branch
                up1 = self._residual_block(inputs, nb_filter_res, 'up1')
                # Lower branch
                pool = tf.contrib.layers.max_pool2d(inputs, [2,2], [2,2], 'VALID', scope=scope.name)
                low1 = self._residual_block(pool, nb_filter_res, 'low1')
                if n > 1:
                    low2 = self._hourglass(low1, n-1, nb_filter_res, 'low2')
                else:
                    low2 = self._residual_block(low1, nb_filter_res, 'low2')
                low3 = self._residual_block(low2, nb_filter_res, 'low3')
                low4 = tf.image.resize_nearest_neighbor(low3, tf.shape(low3)[1:3] * 2,
                                                        name='upsampling')
                if n < 4:
                    return tf.add(up1, low4, name='merge')
                else:
                    return self._residual_block(tf.add(up1, low4), nb_filter_res, 'low4')


    def __init__(self,cfg):
        self.cfg = cfg
        self.discriminator = self.stacked_hourglass(4, 'stacked_hourglass',cfg.num_joints)

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
                                                cfg.num_joints)
            if cfg.location_refinement:
                out['locref'] = prediction_layer(cfg, features, 'locref_pred',
                                                 cfg.num_joints * 2)
            if cfg.intermediate_supervision and not no_interm:
                interm_name = layer_name.format(3, cfg.intermediate_supervision_layer)
                block_interm_out = end_points[interm_name]
                out['part_pred_interm'] = prediction_layer(cfg, block_interm_out,
                                                           'intermediate_supervision',
                                                           cfg.num_joints)
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


    def train(self,batch):
        cfg = self.cfg

        intermediate = cfg.intermediate_supervision
        locref = cfg.location_refinement
        heads = self.get_net(batch[Batch.inputs])
        loss_inter = self.part_detection_loss(heads, batch, locref, intermediate)

        # get adversarial losses and reconstruction loss
        recon = self.discriminator(batch[Batch.inputs],batch[Batch.part_score_targets],stride = int(cfg.stride))
        recon_hat = self.discriminator(batch[Batch.inputs],heads['part_pred'],reuse = True,stride = int(cfg.stride))
        target_heat = tf.stack([batch[Batch.part_score_targets],batch[Batch.part_score_targets],
                    batch[Batch.part_score_targets],batch[Batch.part_score_targets]])
        target_heat_hat = tf.stack([heads['part_pred'],heads['part_pred'],heads['part_pred'],heads['part_pred']])
        loss_D_real = tf.reduce_mean(tf.losses.mean_squared_error(predictions = recon, labels = target_heat))
        loss_D_fake = tf.reduce_mean(tf.losses.mean_squared_error(predictions = recon_hat, labels = target_heat_hat))
        loss_G = tf.reduce_mean(tf.losses.mean_squared_error(predictions = recon_hat, labels = target_heat_hat))

        return loss_inter, loss_G, loss_D_real, loss_D_fake, heads
