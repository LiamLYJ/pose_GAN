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

    global_step = slim.create_global_step()

    variables_to_restore = slim.get_variables_to_restore(include=["resnet_v1"])
    restorer = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver(max_to_keep=5)

    sess = tf.Session()

    coord, thread = start_preloading(sess, enqueue_op, dataset, placeholders)

    train_writer = tf.summary.FileWriter(cfg.log_dir, sess.graph)

    train_op = get_optimizer(total_loss, cfg)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore variables from disk.
    restorer.restore(sess, cfg.init_weights)

    max_iter = int(cfg.max_iter)

    display_iters = cfg.display_iters
    cum_loss = 0.0

    for it in range(sess.run(global_step),max_iter+1):

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
            model_name = cfg.save_path + cfg.snapshot_prefix
            saver.save(sess, model_name, global_step=it)
            print ('saved model once')

    sess.close()
    coord.request_stop()
    coord.join([thread])


if __name__ == '__main__':
    if not os.path.exists(cfg.save_path):
        os.mkdir(cfg.save_path)
    train()
