import os
import sys
from scipy.misc import imread
import tensorflow as tf

import config
import predict
import visualize
from pose_dataset import data_to_input

cfg = tf.app.flags.FLAGS
# set couple of flags for show results for singler person
cfg.all_joints = [[0, 5], [1, 4], [2, 3], [6, 11], [7, 10], [8, 9], [12], [13]]
cfg.all_joints_names = ['ankle', 'knee', 'hip', 'wrist', 'elbow', 'shoulder', 'chin', 'forehead']
cfg.init_weights = '/home/hpc/ssd/lyj/pose_GAN/trained_model/snapshot-1000'
cfg.global_scale = 1.0

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

# Read image from file
file_name = "./demo/3.jpg"
image = imread(file_name, mode='RGB')

image_batch = data_to_input(image)

# Compute prediction with the CNN
outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})

# for name,_ in outputs_np.items():
#     print (name)
# raise

scmap, locref = predict.extract_cnn_output(outputs_np, cfg)

# Extract maximum scoring location from the heatmap, assume 1 person
pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)

# Visualise the reuslts
visualize.show_heatmaps(cfg, image, scmap, pose)
visualize.waitforbuttonpress()
