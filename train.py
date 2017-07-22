from __future__ import division
import tensorflow as tf
import numpy as np
import threading
import os
from glob import glob
import time
slim = tf.contrib.slim
from net_GAN import pose_gan
from net_GAN import get_batch_spec
from factory import create as create_dataset
import config
from pose_dataset import Batch
from tensorflow.python import debug as tf_debug

cfg = tf.app.flags.FLAGS

def setup_preloading(batch_spec):
    placeholders = {name: tf.placeholder(tf.float32, shape=spec) for (name, spec) in batch_spec.items()}
    names = placeholders.keys()
    placeholders_list = list(placeholders.values())

    QUEUE_SIZE = 20

    q = tf.FIFOQueue(QUEUE_SIZE, [tf.float32]*len(batch_spec))
    enqueue_op = q.enqueue(placeholders_list)
    batch_list = q.dequeue()

    batch = {}
    for idx, name in enumerate(names):
        batch[name] = batch_list[idx]
        batch[name].set_shape(batch_spec[name])
    return batch, enqueue_op, placeholders


def load_and_enqueue(sess, enqueue_op, coord, dataset, placeholders):
    while not coord.should_stop():
        batch_np = dataset.next_batch()
        food = {pl: batch_np[name] for (name, pl) in placeholders.items()}
        sess.run(enqueue_op, feed_dict=food)


def start_preloading(sess, enqueue_op, dataset, placeholders):
    coord = tf.train.Coordinator()

    t = threading.Thread(target=load_and_enqueue,
                         args=(sess, enqueue_op, coord, dataset, placeholders))
    t.start()

    return coord, t


def get_optimizer(loss_op, cfg,whichone ='inter'):

    if (cfg.optimizer_name == 'adam' and (whichone in ('G','D','inter','recon'))):
        if whichone == 'inter':
            optimizer = tf.train.AdamOptimizer(cfg.learning_rate_inter)
            # optimizer = tf.train.MomentumOptimizer(learning_rate=cfg.learning_rate_inter, momentum=0.9)
        if whichone == 'G':
            optimizer = tf.train.AdamOptimizer(cfg.learning_rate_G)
            # optimizer = tf.train.MomentumOptimizer(learning_rate=cfg.learning_rate_G, momentum=0.9)
        if whichone == 'D':
            optimizer = tf.train.AdamOptimizer(cfg.learning_rate_D)
            # optimizer = tf.train.MomentumOptimizer(learning_rate=cfg.learning_rate_D, momentum=0.9)
    else:
        raise ValueError('unknown optimizer {}'.format(cfg.optimizer))
    train_op = slim.learning.create_train_op(loss_op, optimizer)

    return  train_op


def save_checkpoint(step,sess,saver,cfg):
    model_name = "pGAN.model"
    model_dir = cfg.dataset_name
    checkpoint_dir = cfg.checkpoint_dir
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess,os.path.join(checkpoint_dir, model_name),global_step=step)



def load_checkpoint(sess,saver,cfg):
    import re
    print(" [*] Reading checkpoint...")
    model_dir = cfg.dataset_name
    checkpoint_dir = cfg.checkpoint_dir
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0



def train():

    dataset = create_dataset(cfg)

    batch_spec = get_batch_spec(cfg)
    batch, enqueue_op, placeholders = setup_preloading(batch_spec)

    _lamda_fake = tf.placeholder(tf.float32,shape =[])
    pose_Gan = pose_gan(cfg)
    losses_inter,loss_G,loss_D_real,loss_D_fake,heads= pose_Gan.train(batch)
    total_loss_inter = losses_inter['total_loss']
    lamda_fake = _lamda_fake + cfg.weight_update_fake*(cfg.weight_real_importance * loss_D_real - loss_D_fake)
    lamda_fake = tf.where(tf.greater(lamda_fake,1.0),1.0, lamda_fake)
    lamda_fake = tf.where(tf.less(lamda_fake,0.0),0.0, lamda_fake)
    loss_D = loss_D_real - lamda_fake * loss_D_fake
    total_loss_inter = cfg.weight_inter * total_loss_inter
    loss_G = cfg.weight_G * loss_G
    loss_D = cfg.weight_D * loss_D

    G_sum = []
    D_sum = []
    inter_sum = []

    D_sum.append(tf.summary.histogram('lamda_fake',lamda_fake))
    for k, t in losses_inter.items():
        inter_sum.append(tf.summary.scalar(k, t))

    G_sum.append(tf.summary.scalar('loss_G',loss_G))
    D_sum.append(tf.summary.scalar('loss_D',loss_D))
    D_sum.append(tf.summary.scalar('loss_D_real', loss_D_real))
    D_sum.append(tf.summary.scalar('loss_D_fake', loss_D_fake))

    inter_sum.append(tf.summary.image('train_im',batch[Batch.inputs]))
    for i in range(cfg.num_joints):
        inter_sum.append(tf.summary.image('pred_joint_%d'%i,tf.expand_dims(tf.sigmoid(heads['part_pred'])[:,:,:,i],-1)))

    G_sums = tf.summary.merge(G_sum)
    D_sums = tf.summary.merge(D_sum)
    inter_sums = tf.summary.merge(inter_sum)

    saver = tf.train.Saver()
    variables_to_restore_backbone = slim.get_variables_to_restore(include=["resnet_v1"])
    restorer_backbone = tf.train.Saver(variables_to_restore_backbone)

    sess = tf.Session()
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    coord, thread = start_preloading(sess, enqueue_op, dataset, placeholders)

    train_writer = tf.summary.FileWriter(cfg.log_dir, sess.graph)

    train_op_inter = get_optimizer(total_loss_inter, cfg)
    train_op_G = get_optimizer(loss_G,cfg,'G')
    # extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(extra_update_ops):
    #     train_op_D = get_optimizer(loss_D,cfg,'D')
    train_op_D = get_optimizer(loss_D,cfg,'D')

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore pretrained pGAN, if cannot, retore pretrained resnet
    could_load,checkpoint_counter = load_checkpoint(sess,saver,cfg)
    if could_load:
        counter = checkpoint_counter
        print ('load checkpoint success')
    else:
        counter = 0
        print ('load checkpoint failed')
        print ('start to load resnet_v1')
        restorer_backbone.restore(sess, cfg.init_weights)

    max_iter = int(cfg.max_iter)

    display_iters = cfg.display_iters
    cum_loss_inter = 0.0
    cum_loss_G = 0.0
    cum_loss_D = 0.0

    for it in range(counter,max_iter+1):

        if it == counter:
            _,loss_val_D = sess.run([train_op_D,loss_D],
                            feed_dict = {_lamda_fake: cfg.weight_fake_init})
            _,loss_val_inter,summary = sess.run([train_op_inter, total_loss_inter,inter_sums])
            train_writer.add_summary(summary, it)
            _,summary,\
                loss_val_D,loss_val_D_real,loss_val_D_fake,lamda_fake_val = sess.run([train_op_D,D_sums,
                            loss_D,loss_D_real,loss_D_fake,lamda_fake],
                            feed_dict = {_lamda_fake: cfg.weight_fake_init})
            train_writer.add_summary(summary, it)
            _,loss_val_G,summary = sess.run([train_op_G,loss_G,G_sums])
            train_writer.add_summary(summary, it)
        else:
            _,loss_val_D = sess.run([train_op_D,loss_D],
                            feed_dict = {_lamda_fake: lamda})
            _,loss_val_inter,summary = sess.run([train_op_inter, total_loss_inter,inter_sums])
            train_writer.add_summary(summary, it)
            _,summary,\
            loss_val_D,loss_val_D_real,loss_val_D_fake,lamda_fake_val = sess.run([train_op_D,D_sums,
                            loss_D,loss_D_real,loss_D_fake,lamda_fake],
                            feed_dict = {_lamda_fake: lamda})
            train_writer.add_summary(summary, it)
            _,loss_val_G,summary = sess.run([train_op_G,loss_G,G_sums])
            train_writer.add_summary(summary, it)

        lamda = lamda_fake_val + cfg.weight_update_fake*(cfg.weight_real_importance * loss_val_D_real - loss_val_D_fake)
        lamda = 1.0 if lamda >1 else lamda
        lamda = 0.0 if lamda <0 else lamda

        cum_loss_inter += loss_val_inter
        cum_loss_G += loss_val_G
        cum_loss_D += loss_val_D

        if it % display_iters == 0:
            average_loss_inter = cum_loss_inter / display_iters
            average_loss_G = cum_loss_G / display_iters
            average_loss_D = cum_loss_D / display_iters

            cum_loss_inter = 0.0
            cum_loss_G = 0.0
            cum_loss_D = 0.0

            print ("iteration:[%d], loss_inter: %.4f, loss_G: %.4f, loss_D: %.4f"
                % (it, average_loss_inter,average_loss_G,average_loss_D))
        # Save snapshot
        if (it % cfg.save_iters == 0 and it != 0) or it == max_iter:
            save_checkpoint(it,sess,saver,cfg)
            print ('saved model with iteration[%d]'%it)

    sess.close()
    coord.request_stop()
    coord.join([thread])


if __name__ == '__main__':
    train()
