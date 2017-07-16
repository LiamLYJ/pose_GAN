import os
import sys
from scipy.misc import imread,imshow,imresize
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import config
import predict
import visualize
from pose_dataset import data_to_input

cfg = tf.app.flags.FLAGS
cfg.init_weights = '/home/hpc/ssd/lyj/pose_GAN/trained_inter/snapshot-350000'
cfg.global_scale = 1.0

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

# Read image from file
file_name = "./demo/2.jpg"
image = imread(file_name, mode='RGB')

image_batch = data_to_input(image)

# Compute prediction with the CNN
outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})

# for name,_ in outputs_np.items():
#     print (name)
# raise

scmap, _ = predict.extract_cnn_output(outputs_np, cfg)
shape = image.shape
scmap = imresize(scmap,shape)
f, axarr = plt.subplots(1,2)
plot_1 = axarr[0]
plot_2 = axarr[1]
plot_1.imshow(image)
plot_2.imshow(scmap)
plt.show()
