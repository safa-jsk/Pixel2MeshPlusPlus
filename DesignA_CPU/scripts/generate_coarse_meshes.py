#!/usr/bin/env python3
"""
Generate Stage 1 (MVP2M) coarse meshes using TensorFlow CPU

This script runs the Stage 1 model on CPU to generate coarse meshes
that will be used as input for the PyTorch Stage 2 refinement.

Usage:
    docker run --rm -v "$(pwd)":/workspace -w /workspace p2mpp:cpu \
        python designA/generate_coarse_meshes.py \
            --test_file data/designB_eval_test.txt \
            --output_dir outputs/designB/coarse_meshes
"""

import os
import sys
import argparse
import time
import pickle
import numpy as np

# Add parent directory to path
# --- Path bootstrap (post-refactor) ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
_TF_SRC = os.path.join(_PROJECT_ROOT, 'src', 'p2mpp', 'tf')
if _TF_SRC not in sys.path:
    sys.path.insert(0, _TF_SRC)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# --- End path bootstrap ---

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tflearn

from modules.config import create_parser
from modules.models_mvp2m import MeshNetMVP2M as Model
from utils.dataloader import DataFetcher
from utils.tools import construct_feed_dict


def main():
    parser = argparse.ArgumentParser(description='Generate coarse meshes using MVP2M Stage 1')
    parser.add_argument('--test_file', type=str, default='data/designB_eval_test.txt',
                        help='Path to test file list')
    parser.add_argument('--data_root', type=str, default='data/designA_subset/p2mppdata/test',
                        help='Path to test data root')
    parser.add_argument('--image_root', type=str, default='data/designA_subset/ShapeNetRendering/rendering_only',
                        help='Path to rendered images')
    parser.add_argument('--output_dir', type=str, default='outputs/designB/coarse_meshes',
                        help='Output directory for coarse meshes')
    parser.add_argument('--model_dir', type=str, default='results/coarse_mvp2m/models',
                        help='Path to Stage 1 model checkpoint')
    parser.add_argument('--mesh_data', type=str, default='data/iccv_p2mpp.dat',
                        help='Path to initial mesh data')
    args = parser.parse_args()
    
    # Force CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    print('=== Stage 1 (MVP2M) Coarse Mesh Generation ===')
    print(f'Test file: {args.test_file}')
    print(f'Output dir: {args.output_dir}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create placeholders
    print('Building model...')
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
    
    # Build model
    config_parser = create_parser()
    config = config_parser.parse_args([])
    config.gpu_id = -1  # CPU only
    
    model = Model(placeholders, logging=True, args=config)
    
    # Load data
    print('Loading data...')
    data = DataFetcher(
        file_list=args.test_file,
        data_root=args.data_root,
        image_root=args.image_root,
        is_val=True
    )
    data.setDaemon(True)
    data.start()
    
    # Initialize session
    print('Initializing session...')
    sesscfg = tf.ConfigProto()
    sesscfg.device_count['GPU'] = 0
    sess = tf.Session(config=sesscfg)
    sess.run(tf.global_variables_initializer())
    
    # Load checkpoint
    print('Loading checkpoint...')
    model.load(sess=sess, ckpt_path=args.model_dir, step=50)
    
    # Load initial mesh
    pkl = pickle.load(open(args.mesh_data, 'rb'))
    feed_dict = construct_feed_dict(pkl, placeholders)
    
    test_number = data.number
    tflearn.is_training(False, sess)
    
    print(f'\nGenerating coarse meshes for {test_number} samples...')
    print('=' * 60)
    
    times = []
    
    for iters in range(test_number):
        img_all_view, labels, poses, data_id, _ = data.fetch()
        
        feed_dict.update({placeholders['img_inp']: img_all_view})
        feed_dict.update({placeholders['labels']: labels})  # Use actual labels from data
        feed_dict.update({placeholders['cameras']: poses})
        
        t_start = time.time()
        coarse_mesh = sess.run(model.output3, feed_dict=feed_dict)
        t_elapsed = time.time() - t_start
        times.append(t_elapsed)
        
        # Save coarse mesh
        output_path = os.path.join(args.output_dir, data_id.replace('.dat', '_coarse.xyz'))
        np.savetxt(output_path, coarse_mesh)
        
        print(f'[{iters+1}/{test_number}] {data_id}: {t_elapsed:.2f}s')
    
    print('=' * 60)
    print(f'Total samples: {len(times)}')
    print(f'Mean time: {np.mean(times):.2f}s/sample (CPU)')
    print(f'Total time: {sum(times):.2f}s')
    print(f'Output dir: {args.output_dir}')
    
    sess.close()


if __name__ == '__main__':
    main()
