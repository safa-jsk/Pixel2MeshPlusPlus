"""
TensorFlow 1.x compatibility wrapper for TF 2.4
Automatically enables TF 1.x mode when importing tensorflow
"""
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Make tf.compat.v1 available as tf for legacy code
import sys
sys.modules['tensorflow'] = tf

print('[Design B] TensorFlow 1.x compatibility mode activated')
