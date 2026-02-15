import os
import sys
import numpy as np
import pickle
import time
import tensorflow as tf
import tflearn

os.chdir('/workspace')
sys.path.insert(0, '/workspace')

from modules.models_mvp2m import MeshNetMVP2M as MVP2MNet
from modules.models_p2mpp import MeshNet as P2MPPNet
from modules.config import execute
from utils.tools import construct_feed_dict

sys.argv = ['', '-f', '/workspace/cfgs/mvp2m.yaml']
cfg = execute()

with open('data/iccv_p2mpp.dat', 'rb') as f:
    pkl = pickle.load(f, encoding='latin1')

imgs = np.load('/workspace/input_images_tf.npy')
cameras = np.loadtxt('data/demo/cameras.txt')

num_blocks = 3
num_supports = 2

placeholders = {
    'features': tf.placeholder(tf.float32, shape=(None, 3), name='features'),
    'img_inp': tf.placeholder(tf.float32, shape=(3, 224, 224, 3), name='img_inp'),
    'labels': tf.placeholder(tf.float32, shape=(None, 6), name='labels'),
    'support1': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'support2': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'support3': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'faces': [tf.placeholder(tf.int32, shape=(None, 4)) for _ in range(num_blocks)],
    'edges': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks)],
    'lape_idx': [tf.placeholder(tf.int32, shape=(None, 10)) for _ in range(num_blocks)],
    'pool_idx': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks - 1)],
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),
    'sample_coord': tf.placeholder(tf.float32, shape=(43, 3), name='sample_coord'),
    'cameras': tf.placeholder(tf.float32, shape=(3, 5), name='Cameras'),
    'faces_triangle': [tf.placeholder(tf.int32, shape=(None, 3)) for _ in range(num_blocks)],
    'sample_adj': [tf.placeholder(tf.float32, shape=(43, 43)) for _ in range(num_supports)],
}

model1 = MVP2MNet(placeholders, logging=True, args=cfg)
model2 = P2MPPNet(placeholders, logging=True, args=cfg)

sesscfg = tf.ConfigProto()
sesscfg.gpu_options.allow_growth = True
sesscfg.allow_soft_placement = True
sess = tf.Session(config=sesscfg)
sess.run(tf.global_variables_initializer())

model1.load(sess=sess, ckpt_path='results/coarse_mvp2m/models/', step=50)
model2.load(sess=sess, ckpt_path='results/refine_p2mpp/models/', step=10)
print('Models restored')

feed_dict = construct_feed_dict(pkl, placeholders)
feed_dict.update({placeholders['img_inp']: imgs})
feed_dict.update({placeholders['labels']: np.zeros([10, 6])})
feed_dict.update({placeholders['cameras']: cameras})

tflearn.is_training(False, sess)

# Warm up
_ = sess.run([model1.output1, model1.output3], feed_dict=feed_dict)

# Time Stage 1
times_s1 = []
for _ in range(10):
    start = time.time()
    output1, output3 = sess.run([model1.output1, model1.output3], feed_dict=feed_dict)
    end = time.time()
    times_s1.append((end - start) * 1000)

# Now run Stage 2 - need coarse mesh from Stage 1
# Update feed_dict with Stage 1 output
feed_dict[placeholders['features']] = output3

# Time Stage 2 - use correct attribute names: output1l and output2l
times_s2 = []
for _ in range(10):
    start = time.time()
    blk1, blk2 = sess.run([model2.output1l, model2.output2l], feed_dict=feed_dict)
    end = time.time()
    times_s2.append((end - start) * 1000)

# Full pipeline timing
times_full = []
for _ in range(10):
    start = time.time()
    # Stage 1
    output1, output3 = sess.run([model1.output1, model1.output3], feed_dict=feed_dict)
    # Update for Stage 2
    feed_dict[placeholders['features']] = output3
    # Stage 2
    blk1, blk2 = sess.run([model2.output1l, model2.output2l], feed_dict=feed_dict)
    end = time.time()
    times_full.append((end - start) * 1000)

print(f"\n=== Design A (TensorFlow CPU) Timing ===")
print(f"Stage 1 (Coarse): {np.mean(times_s1):.1f}ms")
print(f"Stage 2 (Refine): {np.mean(times_s2):.1f}ms")
print(f"Full Pipeline:    {np.mean(times_full):.1f}ms")
print(f"Throughput:       {1000/np.mean(times_full):.2f} samples/sec")

sess.close()
