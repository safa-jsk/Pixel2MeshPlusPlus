# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
# 
# Design A Evaluation - Stage 2 Only (Refined P2MPP)
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
import tflearn
import numpy as np
import pickle
import time

from modules.models_p2mpp import MeshNet
from modules.config import create_parser
from utils.dataloader import DataFetcher
from utils.tools import construct_feed_dict


def xyz2obj(xyz_path, obj_path, face_path):
    """Convert .xyz point cloud to .obj mesh using face topology."""
    vertices = np.loadtxt(xyz_path)
    faces = np.loadtxt(face_path, dtype='|S32')
    v_prefix = np.full([vertices.shape[0], 1], 'v')
    out = np.vstack((np.hstack((v_prefix, vertices)), faces))
    np.savetxt(obj_path, out, fmt='%s', delimiter=' ')


def main(eval_list_file, coarse_mesh_dir, output_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
    
    print('=' * 70)
    print('Design A Evaluation - Stage 2: Refined P2MPP')
    print('=' * 70)
    print('Eval list: {}'.format(eval_list_file))
    print('Coarse meshes: {}'.format(coarse_mesh_dir))
    print('Output dir: {}'.format(output_dir))
    print('=' * 70)
    
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

    model_dir = '../results/refine_p2mpp/models'
    data_root = '../data/p2mppdata/test'
    image_root = '../data/ShapeNetImages/ShapeNetRendering'
    
    os.makedirs(output_dir, exist_ok=True)
    
    parser = create_parser()
    args = parser.parse_args([])
    args.gpu_id = 0
    
    print('=> Building Stage 2 model...')
    import time as time_module
    t_build_start = time_module.time()
    model = MeshNet(placeholders, logging=True, args=args)
    print('   Model built in {:.2f}s'.format(time_module.time() - t_build_start))
    
    print('=> Loading data...')
    print('   Data root: {}'.format(data_root))
    print('   Image root: {}'.format(image_root))
    print('   Coarse mesh root: {}'.format(coarse_mesh_dir))
    t_data_start = time_module.time()
    data = DataFetcher(
        file_list=eval_list_file,
        data_root=data_root,
        image_root=image_root,
        is_val=True,
        mesh_root=coarse_mesh_dir
    )
    print('   DataFetcher created in {:.2f}s'.format(time_module.time() - t_data_start))
    data.setDaemon(True)
    t_start_thread = time_module.time()
    data.start()
    print('   DataFetcher thread started in {:.2f}s'.format(time_module.time() - t_start_thread))
    
    print('=> Initializing session...')
    sesscfg = tf.ConfigProto()
    sesscfg.gpu_options.allow_growth = True
    sesscfg.allow_soft_placement = True
    sess = tf.Session(config=sesscfg)
    sess.run(tf.global_variables_initializer())
    
    print('=> Loading Stage 2 checkpoint...')
    model.load(sess=sess, ckpt_path=model_dir, step=10)
    
    print('=> Loading template mesh...')
    pkl = pickle.load(open('../data/iccv_p2mpp.dat', 'rb'))
    feed_dict = construct_feed_dict(pkl, placeholders)
    
    test_number = data.number
    tflearn.is_training(False, sess)
    
    print('=' * 70)
    print('Starting Stage 2 inference on {} samples'.format(test_number))
    print('=' * 70)
    
    timings = []
    
    for iters in range(test_number):
        print('[{:3d}/{:3d}] Fetching data...'.format(iters + 1, test_number), end=' ', flush=True)
        t_fetch_start = time.time()
        img_all_view, labels, poses, data_id, coarse_mesh = data.fetch()
        t_fetch = time.time() - t_fetch_start
        print('fetched in {:.2f}s, running inference...'.format(t_fetch), end=' ', flush=True)
        
        feed_dict.update({placeholders['img_inp']: img_all_view})
        feed_dict.update({placeholders['features']: coarse_mesh})
        feed_dict.update({placeholders['labels']: labels})
        feed_dict.update({placeholders['cameras']: poses})
        
        t_start = time.time()
        refined_mesh = sess.run(model.output2l, feed_dict=feed_dict)
        t_elapsed = time.time() - t_start
        timings.append(t_elapsed)
        print('done in {:.2f}s'.format(t_elapsed))
        
        # Save refined prediction (.xyz)
        predict_path = os.path.join(output_dir, data_id.replace('.dat', '_predict.xyz'))
        np.savetxt(predict_path, refined_mesh)
        
        # Convert to .obj format
        obj_path = os.path.join(output_dir, data_id.replace('.dat', '_predict.obj'))
        xyz2obj(predict_path, obj_path, '../data/face3.obj')
        
        print('[{:3d}/{:3d}] {} | Time: {:.2f}s'.format(
            iters + 1, test_number, data_id.split('.')[0][:40], t_elapsed))
    
    data.shutdown()
    
    # Save timing statistics
    benchmark_dir = output_dir.replace('eval_meshes', 'benchmark')
    os.makedirs(benchmark_dir, exist_ok=True)
    
    with open(os.path.join(benchmark_dir, 'stage2_timings.txt'), 'w') as f:
        f.write('Stage 2 (Refined P2MPP) Timing Statistics\n')
        f.write('=' * 50 + '\n')
        f.write('Total samples: {}\n'.format(len(timings)))
        f.write('Total time: {:.2f}s\n'.format(sum(timings)))
        f.write('Average time: {:.3f}s\n'.format(np.mean(timings)))
        f.write('Median time: {:.3f}s\n'.format(np.median(timings)))
        f.write('Min time: {:.3f}s\n'.format(np.min(timings)))
        f.write('Max time: {:.3f}s\n'.format(np.max(timings)))
        f.write('Std dev: {:.3f}s\n'.format(np.std(timings)))
    
    # Compute combined timing statistics
    stage1_timing_file = os.path.join(benchmark_dir, 'stage1_timings.txt')
    if os.path.exists(stage1_timing_file):
        with open(stage1_timing_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('Total time:'):
                    stage1_total = float(line.split(':')[1].strip().replace('s', ''))
                    break
        
        combined_total = stage1_total + sum(timings)
        combined_avg = combined_total / len(timings)
        
        with open(os.path.join(benchmark_dir, 'combined_timings.txt'), 'w') as f:
            f.write('Combined 2-Stage Pipeline Timing Statistics\n')
            f.write('=' * 50 + '\n')
            f.write('Total samples: {}\n'.format(len(timings)))
            f.write('Stage 1 total: {:.2f}s\n'.format(stage1_total))
            f.write('Stage 2 total: {:.2f}s\n'.format(sum(timings)))
            f.write('Combined total: {:.2f}s\n'.format(combined_total))
            f.write('Average per sample: {:.3f}s\n'.format(combined_avg))
    
    print('=' * 70)
    print('Stage 2 Complete!')
    print('Refined meshes saved to: {}'.format(output_dir))
    print('Timing stats: {:.2f}s total, {:.3f}s avg'.format(sum(timings), np.mean(timings)))
    print('=' * 70)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-list', type=str, default='designA_eval_list.txt')
    parser.add_argument('--coarse-dir', type=str, default='../outputs/designA/eval_meshes')
    parser.add_argument('--output-dir', type=str, default='../outputs/designA/eval_meshes')
    args = parser.parse_args()
    
    main(args.eval_list, args.coarse_dir, args.output_dir)