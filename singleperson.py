import os
import sys
from scipy.misc import imread
import tensorflow as tf
import scipy.io as sio
import numpy as np
import config
import predict
import visualize
from pose_dataset import data_to_input

from time import time

cfg = tf.app.flags.FLAGS
# set couple of flags for show results for singler person
cfg.all_joints = [[0, 5], [1, 4], [2, 3], [6, 11], [7, 10], [8, 9], [12], [13]]
cfg.all_joints_names = ['ankle', 'knee', 'hip', 'wrist', 'elbow', 'shoulder', 'chin', 'forehead']
# cfg.init_weights = '/home/hpc/ssd/lyj/pose_GAN/checkpoint_PGAN/customer/pGAN.model-10000'
# cfg.init_weights = '/home/hpc/ssd/lyj/pose-tensorflow/models/mpii/train/snapshot-1030000'
cfg.init_weights = '/home/hpc/ssd/lyj/pose-tensorflow/models/mpii/mpii-single-resnet-101'
# cfg.init_weights = '/home/hpc/ssd/lyj/pose_GAN/checkpoint_redundent/customer/redundent-3690000'

cfg.global_scale = 1.0
cfg.redundent = False
# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

# Read image from file
# file_name = "/home/hpc/ssd/lyj/mpii_data/mpii_test_data/im00036_1.png"
# image = imread(file_name, mode='RGB')
#
# image_batch = data_to_input(image)
#
# # check for the computation time
# # start_time = time()
#
# # Compute prediction with the CNN
# outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
#
# # for name,_ in outputs_np.items():
# #     print (name)
# # raise
#
# scmap, locref = predict.extract_cnn_output(outputs_np, cfg)
# # finish_time = time()
# # print ('the time is: ', str(finish_time - start_time))
# # Extract maximum scoring location from the heatmap, assume 1 person
# pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
#
# # Visualise the reuslts
# visualize.show_heatmaps(cfg, image, scmap, pose)
# visualize.waitforbuttonpress()
# raise



# Read test images
# folder_name = '/home/hpc/ssd/lyj/liu_data/tmp/'
folder_name = '/home/hpc/ssd/lyj/liu_data/testing_nobg/'
# folder_name = '/home/hpc/ssd/lyj/mpii_data/mpii_test_data'
start_time = time()
prediction = []
for root,dirs,files in os.walk(folder_name):
    # root.sort()
    dirs.sort()
    files.sort()
    for file in files:
        file_name = os.path.join(root,file)
        image = imread(file_name, mode='RGB')
        image_batch = data_to_input(image)
        outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
        scmap, locref = predict.extract_cnn_output(outputs_np, cfg)
        pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
        pose_tmp = pose[:,0:2]
        prediction.append(pose_tmp)
prediction = np.array(prediction)
pre = np.transpose(prediction,(1,2,0))
# sio.savemat('predictions.mat', {'prediction':pre})
# sio.savemat('pretrain.mat',{'prediction':pre})
# sio.savemat('redundent.mat',{'prediction':pre})
sio.savemat('custom_nopretrain.mat',{'prediction':pre})
# sio.savemat('mpii_pre.mat',{'mpii_pre':pre})
end_time = time()
print (end_time - start_time)
