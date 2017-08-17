import tensorflow as tf

tf.app.flags.DEFINE_integer(
    'num_joints', 14,
    'the numbers of body joints'
)
tf.app.flags.DEFINE_float(
    'stride', 8.0,
    'the stride ->origial to heatmap'
)
tf.app.flags.DEFINE_bool(
    'shuffle', True,
    'shuffle the data or not'
)
tf.app.flags.DEFINE_string(
    'snapshot_prefix', 'snapshot',
    'the snapshot prefix'
)
tf.app.flags.DEFINE_string(
    'checkpoint_dir', './checkpoint_PGAN/',
    'where to save the trained_model'
)
tf.app.flags.DEFINE_string(
    'log_dir', './log_PGAN',
    'save log dir'
)
tf.app.flags.DEFINE_float(
    # 'global_scale', 0.8452830189,
    'global_scale', 1.0,
    'the global_scale wrt origial image'
)
tf.app.flags.DEFINE_bool(
    'location_refinement', True,
    'use location_refinement or not'
)
tf.app.flags.DEFINE_float(
    'locref_stdev', 7.2801,
    'locref_stdev'
)
tf.app.flags.DEFINE_float(
    'locref_loss_weight', 1.0,
    'locref_loss_weight'
)
tf.app.flags.DEFINE_bool(
    'locref_huber_loss', True,
    'if use huber_loss in locref'
)
tf.app.flags.DEFINE_string(
    'optimizer_name', 'adam',
    'which optimizer'
)
tf.app.flags.DEFINE_bool(
    'intermediate_supervision', True,
    'if use intermediate_supervision in Resnet'
)
tf.app.flags.DEFINE_integer(
    'intermediate_supervision_layer', 12,
    'which layer to intermediate supervise'
)
tf.app.flags.DEFINE_bool(
    'regularize', False,
    'if apply regularize'
)
tf.app.flags.DEFINE_float(
    'weight_decay', 0.0001,
    'weight_decay'
)
tf.app.flags.DEFINE_bool(
    'mirror', True,
    'if mirror the data'
)
tf.app.flags.DEFINE_bool(
    'crop', False,
    'if crop the data'
)
tf.app.flags.DEFINE_integer(
    'crop_pad', 0,
    'crop_pad'
)
tf.app.flags.DEFINE_string(
    # 'dataset', '/home/hpc/ssd/lyj/liu_data/dataset_bg.mat',
    'dataset', '/home/hpc/ssd/lyj/liu_data/train_.mat',
    'where to load dataset mat file'
)
tf.app.flags.DEFINE_string(
    'dataset_name', 'customer',
    'the name of dataset'
)
tf.app.flags.DEFINE_string(
    'dataset_type', 'mpii',
    'dataset_type'
)
tf.app.flags.DEFINE_integer(
    'batch_size', 8,
    'batch size'
)
tf.app.flags.DEFINE_integer(
    'input_size', 256,
    'input image size '
)
tf.app.flags.DEFINE_integer(
    'pos_dist_thresh', 17,
    'threshold in computing the gt'
)
# tf.app.flags.DEFINE_float(
#     'scale_jitter_lo', 0.85,
#     'scale_jitter_lo'
# )
# tf.app.flags.DEFINE_float(
#     'scale_jitter_up', 1.15,
#     'scale_jitter_up'
# )
tf.app.flags.DEFINE_string(
    'net_type', 'resnet_101',
    'resnet_101 or resnet_50'
)
tf.app.flags.DEFINE_string(
    'init_weights', '/home/hpc/ssd/lyj/pose-tensorflow/models/pretrained/resnet_v1_101.ckpt',
    'init_weights'
)
tf.app.flags.DEFINE_integer(
    'max_input_size', 850,
    'max input image size'
)
tf.app.flags.DEFINE_integer(
    'display_iters', 10,
    'how frequency need to display'
)
tf.app.flags.DEFINE_integer(
    'save_iters', 5000,
    'how frequency need to save model'
)
tf.app.flags.DEFINE_float(
    'learning_rate_inter', 0.0002,
    'learning rate for inter'
)
tf.app.flags.DEFINE_float(
    'learning_rate_G', 0.0002,
    'learning rate for Generator part'
)
tf.app.flags.DEFINE_float(
    'learning_rate_D', 0.0002,
    'learning rate for Discriminator part'
)
tf.app.flags.DEFINE_integer(
    'max_iter', 10000000,
    'max iteration'
)
tf.app.flags.DEFINE_bool(
    'weigh_only_present_joints', False,
    'if only weight the present joints'
)
tf.app.flags.DEFINE_float(
    'weight_inter', 1.0,
    'loss weight for inter in pose estimation net'
)
tf.app.flags.DEFINE_float(
    'weight_G', 0.0,
    'loss weight for generator in pose estimation net'
)
tf.app.flags.DEFINE_float(
    'weight_D', 0.0,
    'loss weight for discriminator'
)
tf.app.flags.DEFINE_float(
    'weight_fake_init', 0.15,
    'intial update weight for lamda_fake in discriminator'
)
tf.app.flags.DEFINE_float(
    'weight_update_fake', 10.0,
    'weight_update_fake for lamda_fake in discriminator'
)
tf.app.flags.DEFINE_float(
    'weight_real_importance', 4.0,
    'how much is the real loss importance in discriminator'
)
