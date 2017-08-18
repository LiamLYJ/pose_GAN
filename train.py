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


def get_optimizer(loss_op, cfg):

    if cfg.optimizer_name == "adam":
        optimizer = tf.train.AdamOptimizer(cfg.learning_rate)
    else:
        raise ValueError('unknown optimizer {}'.format(cfg.optimizer))
    train_op = slim.learning.create_train_op(loss_op, optimizer)

    return  train_op


def save_checkpoint(step,sess,saver,cfg):
    model_name = cfg.snapshot_prefix
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

    losses,heads = pose_gan(cfg).train(batch)
    total_loss = losses['total_loss']

    for k, t in losses.items():
        tf.summary.scalar(k, t)

    tf.summary.image('train_im',batch[Batch.inputs])

    for i in range(cfg.num_joints):
        tf.summary.image('pred_joint_%d'%i,tf.expand_dims(tf.sigmoid(heads['part_pred'])[:,:,:,i],-1))

    merged_summaries = tf.summary.merge_all()

    variables_to_restore_backbone = slim.get_variables_to_restore(include=["resnet_v1"])
    restorer_backbone = tf.train.Saver(variables_to_restore_backbone)

    saver = tf.train.Saver()

    sess = tf.Session()

    coord, thread = start_preloading(sess, enqueue_op, dataset, placeholders)

    train_writer = tf.summary.FileWriter(cfg.log_dir, sess.graph)

    train_op = get_optimizer(total_loss, cfg)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore variables from disk.
    could_load,checkpoint_counter = load_checkpoint(sess,saver,cfg)
    if could_load:
        counter = checkpoint_counter
        print ('load checkpoint Success')
    else:
        counter = 0
        print ('load checkpoint failed')
        print ('start to load resnet_v1')
        restorer_backbone.restore(sess, cfg.init_weights)

    max_iter = int(cfg.max_iter)

    display_iters = cfg.display_iters
    cum_loss = 0.0

    for it in range(counter,max_iter+1):

        [_, loss_val, summary] = sess.run([train_op, total_loss, merged_summaries])
        cum_loss += loss_val
        train_writer.add_summary(summary, it)

        if it % display_iters == 0:
            average_loss = cum_loss / display_iters
            cum_loss = 0.0
            print("iteration: {} loss: {} "
                         .format(it, "{0:.4f}".format(average_loss)))

        # Save snapshot
        if (it % cfg.save_iters == 0 and it != 0) or it == max_iter:
            save_checkpoint(it,sess,saver,cfg)
            print ('saved model once with iteration[%d]'%it)

    sess.close()
    coord.request_stop()
    coord.join([thread])


if __name__ == '__main__':
    train()
