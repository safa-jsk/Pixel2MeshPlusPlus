import os
import sys
import numpy as np
import pickle
import tensorflow as tf
import tflearn

os.chdir('/workspace')
sys.path.insert(0, '/workspace')

from modules.models_mvp2m import MeshNetMVP2M as MVP2MNet
from modules.models_p2mpp import MeshNet as P2MPPNet
from modules.config import execute
from utils.tools import construct_feed_dict, load_demo_image

# Config
sys.argv = ['', '-f', '/workspace/cfgs/mvp2m.yaml']
cfg = execute()

# Load mesh data
with open('data/iccv_p2mpp.dat', 'rb') as f:
    pkl = pickle.load(f, encoding='latin1')

# Load images 
imgs = np.load('/workspace/input_images_tf.npy')
cameras = np.loadtxt('data/demo/cameras.txt')

num_blocks = 3
num_supports = 2

# Placeholders from demo.py
placeholders = {
    'features': tf.placeholder(tf.float32, shape=(None, 3)),
    'img_inp': tf.placeholder(tf.float32, shape=(3, 224, 224, 3)),
    'labels': tf.placeholder(tf.float32, shape=(None, 6)),
    'support1': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'support2': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'support3': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'faces': [tf.placeholder(tf.int32, shape=(None, 4)) for _ in range(num_blocks)],
    'edges': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks)],
    'lape_idx': [tf.placeholder(tf.int32, shape=(None, 20)) for _ in range(num_blocks)],
    'pool_idx': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_supports)],
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),
    'sample_coord': tf.placeholder(tf.float32, shape=(43, 3), name='sample_coord'),
    'cameras': tf.placeholder(tf.float32, shape=(3, 5), name='Cameras'),
    'faces_triangle': [tf.placeholder(tf.int32, shape=(None, 3)) for _ in range(num_blocks)],
    'sample_adj': [tf.placeholder(tf.float32, shape=(43, 43)) for _ in range(num_supports)],
}

# Build models
model1 = MVP2MNet(placeholders, logging=True, args=cfg)
model2 = P2MPPNet(placeholders, logging=True, args=cfg)

# Initialize session
sesscfg = tf.ConfigProto()
sesscfg.gpu_options.allow_growth = True
sesscfg.allow_soft_placement = True
sess = tf.Session(config=sesscfg)
sess.run(tf.global_variables_initializer())

# Restore checkpoints
saver = tf.train.Saver(list(model1.vars.values()))
saver.restore(sess, 'results/coarse_mvp2m/models/meshnetmvp2m.ckpt-50')
print('Stage 1 restored')

# Feed dict
feed_dict = construct_feed_dict(pkl, placeholders)
feed_dict.update({placeholders['img_inp']: imgs})
feed_dict.update({placeholders['labels']: np.zeros([10, 6])})
feed_dict.update({placeholders['cameras']: cameras})

tflearn.is_training(False, sess)

# Run Stage 1
output1, output3 = sess.run([model1.output1, model1.output3], feed_dict=feed_dict)

print('Stage 1 output1 shape:', output1.shape)
print('Stage 1 output3 shape:', output3.shape)

np.save('/workspace/tf_output1.npy', output1)
np.save('/workspace/tf_output3.npy', output3)

print('\nTensorFlow Stage 1 output3 (coords3) stats:')
print('  X: min={:.4f}, max={:.4f}, mean={:.4f}'.format(output3[:, 0].min(), output3[:, 0].max(), output3[:, 0].mean()))
print('  Y: min={:.4f}, max={:.4f}, mean={:.4f}'.format(output3[:, 1].min(), output3[:, 1].max(), output3[:, 1].mean()))
print('  Z: min={:.4f}, max={:.4f}, mean={:.4f}'.format(output3[:, 2].min(), output3[:, 2].max(), output3[:, 2].mean()))

sess.close()
