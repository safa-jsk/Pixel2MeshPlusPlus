#!/usr/bin/env python3
"""TensorFlow 1.x compatibility setup for TF 2.4+"""
import sys
import os

# Disable TF 2.x behavior globally
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import and configure TF 1.x mode
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Make tf.compat.v1 the default tf
sys.modules['tensorflow'] = tf

print('[Design B] TensorFlow 1.x compatibility mode enabled')
print(f'[Design B] TensorFlow version: {tf.__version__}')
print(f'[Design B] CUDA available: {tf.test.is_built_with_cuda()}')
