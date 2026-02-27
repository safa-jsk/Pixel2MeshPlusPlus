# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
# 
# Design A Evaluation - Stage 1 Only (Coarse MVP2M)
import sys
import os
# --- Path bootstrap (post-refactor) ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
_TF_SRC = os.path.join(_PROJECT_ROOT, 'src', 'p2mpp', 'tf')
if _TF_SRC not in sys.path:
    sys.path.insert(0, _TF_SRC)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# --- End path bootstrap ---

import tensorflow as tf
import tflearn
import numpy as np
import pickle
import time

from modules.models_mvp2m import MeshNetMVP2M
from modules.config import create_parser
from utils.dataloader import DataFetcher
from utils.tools import construct_feed_dict


def main(eval_list_file, output_dir):
    # [DESIGN.A] Force CPU-only execution (baseline)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
    
    print('=' * 70)
    print('Design A Evaluation - Stage 1: Coarse MVP2M')
    print('=' * 70)
    print('Eval list: {}'.format(eval_list_file))
    print('Output dir: {}'.format(output_dir))
    print('=' * 70)
    
    # [DESIGN.A][CAMFM.A5_METHOD] Fixed seed for reproducibility
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)
    
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

    model_dir = '../results/coarse_mvp2m/models'
    data_root = '../data/p2mppdata/test'
    image_root = '../data/ShapeNetImages/ShapeNetRendering'
    
    os.makedirs(output_dir, exist_ok=True)
    
    parser = create_parser()
    args = parser.parse_args([])
    args.gpu_id = 0
    
    print('=> Building Stage 1 model...')
    model = MeshNetMVP2M(placeholders, logging=True, args=args)
    
    print('=> Loading data...')
    data = DataFetcher(
        file_list=eval_list_file,
        data_root=data_root,
        image_root=image_root,
        is_val=True
    )
    data.setDaemon(True)
    data.start()
    
    print('=> Initializing session...')
    sesscfg = tf.ConfigProto()
    sesscfg.gpu_options.allow_growth = True
    sesscfg.allow_soft_placement = True
    sess = tf.Session(config=sesscfg)
    sess.run(tf.global_variables_initializer())
    
    print('=> Loading Stage 1 checkpoint...')
    model.load(sess=sess, ckpt_path=model_dir, step=50)
    
    print('=> Loading template mesh...')
    pkl = pickle.load(open('../data/iccv_p2mpp.dat', 'rb'))
    feed_dict = construct_feed_dict(pkl, placeholders)
    
    test_number = data.number
    tflearn.is_training(False, sess)
    
    print('=' * 70)
    print('Starting Stage 1 inference on {} samples'.format(test_number))
    print('=' * 70)
    
    timings = []
    
    for iters in range(test_number):
        img_all_view, labels, poses, data_id, mesh = data.fetch()
        feed_dict.update({placeholders['img_inp']: img_all_view})
        feed_dict.update({placeholders['labels']: labels})
        feed_dict.update({placeholders['cameras']: poses})
        
        t_start = time.time()
        out3 = sess.run(model.output3, feed_dict=feed_dict)
        t_elapsed = time.time() - t_start
        timings.append(t_elapsed)
        
        # Save ground truth
        label_path = os.path.join(output_dir, data_id.replace('.dat', '_ground.xyz'))
        np.savetxt(label_path, labels)
        
        # Save coarse prediction (named _predict.xyz for Stage 2 compatibility)
        predict_path = os.path.join(output_dir, data_id.replace('.dat', '_predict.xyz'))
        np.savetxt(predict_path, out3)
        
        print('[{:3d}/{:3d}] {} | Time: {:.2f}s'.format(
            iters + 1, test_number, data_id.split('.')[0][:40], t_elapsed))
    
    data.shutdown()
    
    # Save timing statistics
    benchmark_dir = output_dir.replace('eval_meshes', 'benchmark')
    os.makedirs(benchmark_dir, exist_ok=True)
    
    with open(os.path.join(benchmark_dir, 'stage1_timings.txt'), 'w') as f:
        f.write('Stage 1 (Coarse MVP2M) Timing Statistics\n')
        f.write('=' * 50 + '\n')
        f.write('Total samples: {}\n'.format(len(timings)))
        f.write('Total time: {:.2f}s\n'.format(sum(timings)))
        f.write('Average time: {:.3f}s\n'.format(np.mean(timings)))
        f.write('Median time: {:.3f}s\n'.format(np.median(timings)))
        f.write('Min time: {:.3f}s\n'.format(np.min(timings)))
        f.write('Max time: {:.3f}s\n'.format(np.max(timings)))
        f.write('Std dev: {:.3f}s\n'.format(np.std(timings)))
    
    print('=' * 70)
    print('Stage 1 Complete!')
    print('Coarse meshes saved to: {}'.format(output_dir))
    print('  (saved as *_predict.xyz for Stage 2 compatibility)')
    print('Timing stats: {:.2f}s total, {:.3f}s avg'.format(sum(timings), np.mean(timings)))
    print('=' * 70)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-list', type=str, default='designA_eval_list.txt')
    parser.add_argument('--output-dir', type=str, default='../outputs/designA/eval_meshes')
    args = parser.parse_args()
    
    main(args.eval_list, args.output_dir)
