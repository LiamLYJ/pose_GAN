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

    if (cfg.optimizer_name == "adam" and (whichone in ('G','D','inter','recon'))):
        if whichone == 'inter':
            optimizer = tf.train.AdamOptimizer(cfg.learning_rate_inter)
        if whichone == 'G':
            optimizer = tf.train.AdamOptimizer(cfg.learning_rate_G)
        if whichone == 'D':
            optimizer = tf.train.AdamOptimizer(cfg.learning_rate_D)
        if whichone == 'recon':
            optimizer = tf.train.AdamOptimizer(cfg.learning_rate_recon)
    else:
        raise ValueError('unknown optimizer {}'.format(cfg.optimizer))
    train_op = slim.learning.create_train_op(loss_op, optimizer)

    return  train_op


def save_checkpoint(step,sess,saver,cfg):
    model_name = "pGAN.model"
    model_dir = cfg.dataset
    checkpoint_dir = cfg.checkpoint_dir
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess,os.path.join(checkpoint_dir, model_name),global_step=step)



def load_checkpoint(sess,saver,cfg):
    import re
    print(" [*] Reading checkpoint...")
    model_dir = cfg.dataset
    checkpoint_dir = cfg.checkpoint_dirs
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

    losses_inter,loss_G,loss_D,loss_recon,heads,d_real,d_fake = pose_gan(cfg).train(batch)
    total_loss_inter = losses_inter['total_loss']

    tf.summary.histogram('d_real',d_real)
    tf.summary.histogram('d_fake',d_fake)
    for k, t in losses_inter.items():
        tf.summary.scalar(k, t)

    tf.summary.scalar('loss_G',loss_G)
    tf.summary.scalar('loss_D',loss_D)
    tf.summary.scalar('loss_recon',loss_recon)

    tf.summary.image('train_im',batch[Batch.inputs])

    tf.summary.image('pred_heat',tf.sigmoid(heads['part_pred']))

    merged_summaries = tf.summary.merge_all()

    saver = tf.train.Saver()

    sess = tf.Session()

    coord, thread = start_preloading(sess, enqueue_op, dataset, placeholders)

    train_writer = tf.summary.FileWriter(cfg.log_dir, sess.graph)

    train_op_inter = get_optimizer(total_loss_inter, cfg)
    train_op_G = get_optimizer(loss_G,cfg,'G')
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_op_D = get_optimizer(loss_D,cfg,'D')
    train_op_recon = get_optimizer(loss_recon,'recon')

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
        variables_to_restore_backbone = slim.get_variables_to_restore(include=["resnet_v1"])
        restorer_backbone = tf.train.Saver(variables_to_restore_backbone)
        restorer_backbone.restore(sess, cfg.init_weights)

    max_iter = int(cfg.max_iter)

    display_iters = cfg.display_iters
    cum_loss_inter = 0.0
    cum_loss_G = 0.0
    cum_loss_D = 0.0
    cum_loss_recon = 0.0
    for it in range(counter,max_iter+1):

        [_, _, _,_,
        loss_val_inter, loss_val_G, loss_val_D,loss_val_recon,
         summary] = sess.run([train_op,train_op_G,train_op_D,train_op_recon
                            total_loss_inter,loss_G,loss_D,loss_recon
                             merged_summaries])
        cum_loss_inter += loss_val_inter
        cum_loss_G += loss_val_G
        cum_loss_D += loss_val_D
        cum_loss_recon += loss_val_recon
        train_writer.add_summary(summary, it)

        if it % display_iters == 0:
            average_loss_inter = cum_loss_inter / display_iters
            average_loss_G = cum_loss_G / display_iters
            average_loss_D = cum_loss_D / display_iters
            average_loss_recon = cum_loss_recon / display_iters
            cum_loss_inter = 0.0
            cum_loss_G = 0.0
            cum_loss_D = 0.0
            cum_loss_recon = 0.0
            print ("iteration:[%d], loss_inter: %.4f, loss_G: %.4f, loss_D: %.4f, loss_recon: %.4f"
                % (it, average_loss_inter,average_loss_G,average_loss_D,average_loss_recon))
        # Save snapshot
        if (it % cfg.save_iters == 0 and it != 0) or it == max_iter:
            save_checkpoint(it,sess,saver,cfg)
            print ('saved model with iteration[%d]'%it)

    sess.close()
    coord.request_stop()
    coord.join([thread])


if __name__ == '__main__':
    train()
