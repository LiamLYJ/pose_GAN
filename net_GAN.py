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

_networks_map = {
    'resnet101': {'C1':'resnet_v1_101/conv1/Relu:0',
               'C2':'resnet_v1_101/block1/unit_2/bottleneck_v1',
               'C3':'resnet_v1_101/block2/unit_3/bottleneck_v1',
               'C4':'resnet_v1_101/block3/unit_5/bottleneck_v1',
               'C5':'resnet_v1_101/block4/unit_3/bottleneck_v1',
               }
  }

def _extra_conv_arg_scope(weight_decay=0.00001, activation_fn=None, normalizer_fn=None):

  with slim.arg_scope(
      [slim.conv2d, slim.conv2d_transpose],
      padding='SAME',
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
      activation_fn=activation_fn,
      normalizer_fn=normalizer_fn,) as arg_sc:
    with slim.arg_scope(
      [slim.fully_connected],
          weights_regularizer=slim.l2_regularizer(weight_decay),
          weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
          activation_fn=activation_fn,
          normalizer_fn=normalizer_fn) as arg_sc:
          return arg_sc

def get_batch_spec(cfg):
    num_joints = cfg.num_joints
    batch_size = cfg.batch_size
    batch_spec = {
        Batch.inputs: [batch_size, None, None, 3],
        Batch.part_score_targets: [batch_size, None, None, num_joints],
        Batch.part_score_weights: [batch_size, None, None, num_joints]
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


class pose_gan(object):

    def __init__(self,cfg):
        self.cfg = cfg

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
        if cfg.redundent:
            for c in range(4,2,-1):
                if c == 4:
                    weight_redundent = 0.5
                if c == 3:
                    weight_redundent = 0.3
                loss['redundent_loss_R%d'%c] = weight_redundent * add_part_loss('R%d'%c)
                total_loss = total_loss + loss['redundent_loss_R%d'%c]
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
            if  not self.cfg.redundent:
                return net, end_points
            else:
                return self.build_redundent(net,end_points)

    def build_redundent(self,net,end_points):
        arg_scope = _extra_conv_arg_scope()
        redundent_end_points = {}
        redundent_map = _networks_map['resnet101']
        with tf.variable_scope('redundent'):
            with slim.arg_scope(arg_scope):
                for c in range(5,2,-1):
                    s,s_ = end_points[redundent_map['C%d'%c]], end_points[redundent_map['C%d'%(c-1)]]
                    up_shape = tf.shape(s_)
                    s = slim.conv2d(s,256,[1,1],stride = 1, scope = 'C%d_uplayer'%c)
                    s = tf.image.resize_bilinear(s,[up_shape[1],up_shape[2]], name ='C%d/upscale'%c)
                    s_ = slim.conv2d(s_,256,[1,1],stride = 1,scope = 'C%d_downlayer'%c)
                    s = tf.add(s,s_, name ='C%d/additon_together'%c)
                    s = slim.conv2d(s,256,[3,3],stride = 1,scope = 'C%d/redundent'%c)
                    for k in range(5-c):
                        s = slim.conv2d(s,256,[3,3],stride = 2,scope = 'C%d_%d/downscale'%(c,k))
                    redundent_end_points['R%d'%c] = s
        features = redundent_end_points['R5']
        return features,redundent_end_points

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
            if cfg.redundent:
                for c in range(4,2,-1):
                    block_interm_out = end_points['R%d'%c]
                    out['R%d'%c] = prediction_layer(cfg, block_interm_out,
                                                    'redundent_loss_R%d'%c,
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
        return self.part_detection_loss(heads, batch, locref, intermediate),heads
