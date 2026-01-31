# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
# 
# Design A Complete Evaluation (Stage 1 + Stage 2)
# With metrics: Chamfer Distance, F1@tau, F1@2tau
import sys
import os

# Add parent directory to path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
import tflearn
import numpy as np
import pickle
import time
import argparse

from modules.models_mvp2m import MeshNetMVP2M as MVP2MNet
from modules.models_p2mpp import MeshNet as P2MPPNet
from modules.config import create_parser
from utils.dataloader import DataFetcher
from utils.tools import construct_feed_dict

# Import metrics module (Chamfer Distance, F1@tau, F1@2tau)
from modules.chamfer import nn_distance


def main(eval_list_file, output_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
    # ---------------------------------------------------------------
    print('=' * 70)
    print('Design A Complete Evaluation (2-Stage Pipeline)')
    print('With Metrics: Chamfer Distance, F1@tau, F1@2tau')
    print('=' * 70)
    print('Eval list: {}'.format(eval_list_file))
    print('Output dir: {}'.format(output_dir))
    print('=' * 70)
    
    # Metric threshold (tau = 10^-4, as per Pixel2Mesh paper)
    TAU = 0.0001
    TAU_2 = 2 * TAU
    
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)
    # ---------------------------------------------------------------
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

    # Paths (relative to project root - running from designA/ folder)
    model1_dir = '../results/coarse_mvp2m/models'
    model2_dir = '../results/refine_p2mpp/models'
    data_root = '../data/designA_subset/p2mppdata/test'
    image_root = '../data/designA_subset/ShapeNetRendering/rendering_only'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create config object
    parser = create_parser()
    args = parser.parse_args([])
    args.gpu_id = 0
    
    # -------------------------------------------------------------------
    print('=> Building models...')
    model1 = MVP2MNet(placeholders, logging=True, args=args)
    model2 = P2MPPNet(placeholders, logging=True, args=args)
    
    # ---------------------------------------------------------------
    print('=> Loading data...')
    data = DataFetcher(
        file_list=eval_list_file,
        data_root=data_root,
        image_root=image_root,
        is_val=True,
        mesh_root=None  # No pre-computed coarse mesh
    )
    data.setDaemon(True)
    data.start()
    
    # ---------------------------------------------------------------
    print('=> Initializing TensorFlow session...')
    sesscfg = tf.ConfigProto()
    sesscfg.gpu_options.allow_growth = True
    sesscfg.allow_soft_placement = True
    sess = tf.Session(config=sesscfg)
    sess.run(tf.global_variables_initializer())
    
    # ---------------------------------------------------------------
    # Setup Chamfer Distance computation (using modules/chamfer.py)
    print('=> Setting up metrics computation (Chamfer Distance, F1@tau, F1@2tau)...')
    xyz1_ph = tf.placeholder(tf.float32, shape=(None, 3), name='pred_pts')
    xyz2_ph = tf.placeholder(tf.float32, shape=(None, 3), name='gt_pts')
    dist1_op, idx1_op, dist2_op, idx2_op = nn_distance(xyz1_ph, xyz2_ph)
    
    # ---------------------------------------------------------------
    print('=> Loading model checkpoints...')
    model1.load(sess=sess, ckpt_path=model1_dir, step=50)
    model2.load(sess=sess, ckpt_path=model2_dir, step=10)
    
    # ---------------------------------------------------------------
    # Load init ellipsoid
    print('=> Loading mesh template...')
    pkl = pickle.load(open('../data/iccv_p2mpp.dat', 'rb'))
    feed_dict = construct_feed_dict(pkl, placeholders)
    initial_coords = pkl['coord']  # Save initial ellipsoid coordinates (156, 3)
    
    # ---------------------------------------------------------------
    test_number = data.number
    tflearn.is_training(False, sess)
    
    print('=' * 70, flush=True)
    print('Starting 2-stage inference on {} samples'.format(test_number), flush=True)
    print('=' * 70, flush=True)
    
    timing_results = []
    stage1_times = []
    stage2_times = []
    
    # Metrics storage
    metrics_results = []
    category_metrics = {}  # Per-category aggregation
    
    for iters in range(test_number):
        # Fetch data
        img_all_view, labels, poses, data_id, _ = data.fetch()
        
        # Reset features to initial ellipsoid for Stage 1
        # (This is crucial - otherwise Stage 2 output from previous iteration persists)
        feed_dict.update({placeholders['features']: initial_coords})
        feed_dict.update({placeholders['img_inp']: img_all_view})
        feed_dict.update({placeholders['labels']: labels})
        feed_dict.update({placeholders['cameras']: poses})
        
        # === STAGE 1: Coarse mesh generation ===
        t1_start = time.time()
        stage1_out3 = sess.run(model1.output3, feed_dict=feed_dict)
        t1_elapsed = time.time() - t1_start
        stage1_times.append(t1_elapsed)
        
        # === STAGE 2: Mesh refinement ===
        t2_start = time.time()
        feed_dict.update({placeholders['features']: stage1_out3})
        stage2_out = sess.run(model2.output2l, feed_dict=feed_dict)
        t2_elapsed = time.time() - t2_start
        stage2_times.append(t2_elapsed)
        
        total_elapsed = t1_elapsed + t2_elapsed
        timing_results.append((data_id, t1_elapsed, t2_elapsed, total_elapsed))
        
        # Save ground truth
        label_path = os.path.join(output_dir, data_id.replace('.dat', '_ground.xyz'))
        np.savetxt(label_path, labels)
        
        # Save coarse prediction (stage 1)
        coarse_path = os.path.join(output_dir, data_id.replace('.dat', '_coarse.xyz'))
        np.savetxt(coarse_path, stage1_out3)
        
        # Save refined prediction (stage 2)
        pred_path = os.path.join(output_dir, data_id.replace('.dat', '_predict.xyz'))
        np.savetxt(pred_path, stage2_out)
        
        # Save as OBJ
        obj_path = os.path.join(output_dir, data_id.replace('.dat', '_predict.obj'))
        vert = np.hstack((np.full([stage2_out.shape[0], 1], 'v'), stage2_out))
        face = np.loadtxt('../data/face3.obj', dtype='|S32')
        mesh_data = np.vstack((vert, face))
        np.savetxt(obj_path, mesh_data, fmt='%s', delimiter=' ')
        
        # ===============================================================
        # COMPUTE METRICS: Chamfer Distance, F1@tau, F1@2tau
        # ===============================================================
        gt_pts = labels[:, :3]  # Ground truth (first 3 columns)
        pred_pts = stage2_out   # Prediction (2466, 3)
        
        d1, _, d2, _ = sess.run(
            [dist1_op, idx1_op, dist2_op, idx2_op],
            feed_dict={xyz1_ph: pred_pts, xyz2_ph: gt_pts}
        )
        d1 = np.squeeze(d1)
        d2 = np.squeeze(d2)
        
        # Chamfer Distance
        chamfer_dist = np.mean(d1) + np.mean(d2)
        
        # F1@tau
        precision_tau = 100.0 * (np.sum(d1 <= TAU) / len(d1))
        recall_tau = 100.0 * (np.sum(d2 <= TAU) / len(d2))
        f1_tau = (2 * precision_tau * recall_tau) / (precision_tau + recall_tau + 1e-6)
        
        # F1@2tau
        precision_2tau = 100.0 * (np.sum(d1 <= TAU_2) / len(d1))
        recall_2tau = 100.0 * (np.sum(d2 <= TAU_2) / len(d2))
        f1_2tau = (2 * precision_2tau * recall_2tau) / (precision_2tau + recall_2tau + 1e-6)
        
        metrics_results.append({
            'sample_id': data_id,
            'chamfer_distance': chamfer_dist,
            'f1_tau': f1_tau,
            'f1_2tau': f1_2tau,
            'precision_tau': precision_tau,
            'recall_tau': recall_tau,
        })
        
        # Aggregate by category
        category_id = data_id.split('_')[0]
        if category_id not in category_metrics:
            category_metrics[category_id] = {'cd': [], 'f1_tau': [], 'f1_2tau': []}
        category_metrics[category_id]['cd'].append(chamfer_dist)
        category_metrics[category_id]['f1_tau'].append(f1_tau)
        category_metrics[category_id]['f1_2tau'].append(f1_2tau)
        
        print('[{:2d}/{:2d}] {} | Stage1: {:.2f}s | Stage2: {:.2f}s | Total: {:.2f}s | CD: {:.6f} | F1@tau: {:.1f}% | F1@2tau: {:.1f}%'.format(
            iters + 1, test_number, data_id[:35], t1_elapsed, t2_elapsed, total_elapsed,
            chamfer_dist, f1_tau, f1_2tau), flush=True)
    
    # ---------------------------------------------------------------
    data.shutdown()
    
    # Save detailed timing results
    timing_file = os.path.join(output_dir, '../benchmark/timing_results_detailed.csv')
    os.makedirs(os.path.dirname(timing_file), exist_ok=True)
    with open(timing_file, 'w') as f:
        f.write('sample_id,stage1_time_sec,stage2_time_sec,total_time_sec\n')
        for data_id, t1, t2, total in timing_results:
            f.write('{},{:.4f},{:.4f},{:.4f}\n'.format(data_id, t1, t2, total))
    
    # ===============================================================
    # Save metrics results (Chamfer Distance, F1@tau, F1@2tau)
    # ===============================================================
    metrics_file = os.path.join(output_dir, '../benchmark/metrics_results.csv')
    with open(metrics_file, 'w') as f:
        f.write('sample_id,chamfer_distance,f1_tau,f1_2tau,precision_tau,recall_tau\n')
        for m in metrics_results:
            f.write('{},{:.8f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(
                m['sample_id'], m['chamfer_distance'], m['f1_tau'], m['f1_2tau'],
                m['precision_tau'], m['recall_tau']
            ))
    
    # Calculate overall metrics
    avg_cd = np.mean([m['chamfer_distance'] for m in metrics_results])
    std_cd = np.std([m['chamfer_distance'] for m in metrics_results])
    avg_f1_tau = np.mean([m['f1_tau'] for m in metrics_results])
    avg_f1_2tau = np.mean([m['f1_2tau'] for m in metrics_results])
    
    # Calculate statistics
    total_times = [t for _, _, _, t in timing_results]
    avg_time = np.mean(total_times)
    std_time = np.std(total_times)
    min_time = np.min(total_times)
    max_time = np.max(total_times)
    
    avg_stage1 = np.mean(stage1_times)
    avg_stage2 = np.mean(stage2_times)
    
    # Save summary statistics
    summary_file = os.path.join(output_dir, '../benchmark/summary_stats.txt')
    with open(summary_file, 'w') as f:
        f.write('Design A Baseline Performance Summary\n')
        f.write('=' * 60 + '\n')
        f.write('Evaluation Date: {}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
        f.write('Number of samples: {}\n'.format(test_number))
        f.write('Metric threshold tau: {}\n'.format(TAU))
        f.write('\n')
        f.write('Timing Statistics (per sample):\n')
        f.write('-' * 60 + '\n')
        f.write('  Average Stage 1: {:.3f}s ± {:.3f}s\n'.format(avg_stage1, np.std(stage1_times)))
        f.write('  Average Stage 2: {:.3f}s ± {:.3f}s\n'.format(avg_stage2, np.std(stage2_times)))
        f.write('  Average Total:   {:.3f}s ± {:.3f}s\n'.format(avg_time, std_time))
        f.write('  Min time:        {:.3f}s\n'.format(min_time))
        f.write('  Max time:        {:.3f}s\n'.format(max_time))
        f.write('\n')
        f.write('Quality Metrics:\n')
        f.write('-' * 60 + '\n')
        f.write('  Chamfer Distance: {:.8f} ± {:.8f}\n'.format(avg_cd, std_cd))
        f.write('  F1@tau:           {:.2f}%\n'.format(avg_f1_tau))
        f.write('  F1@2tau:          {:.2f}%\n'.format(avg_f1_2tau))
        f.write('\n')
        f.write('Per-Category Metrics:\n')
        f.write('-' * 60 + '\n')
        category_names = {
            '02691156': 'plane', '02958343': 'car', '03001627': 'chair',
            '03636649': 'lamp', '03691459': 'speaker', '04379243': 'table'
        }
        for cat_id, cat_data in sorted(category_metrics.items()):
            cat_name = category_names.get(cat_id, cat_id)
            f.write('  {:8s} ({:8s}): CD={:.8f}  F1@tau={:.2f}%  F1@2tau={:.2f}%  (n={})\n'.format(
                cat_id, cat_name,
                np.mean(cat_data['cd']),
                np.mean(cat_data['f1_tau']),
                np.mean(cat_data['f1_2tau']),
                len(cat_data['cd'])
            ))
        f.write('\n')
        f.write('Total Processing:\n')
        f.write('-' * 60 + '\n')
        f.write('  Total time: {:.2f}s ({:.2f} minutes)\n'.format(sum(total_times), sum(total_times)/60))
        f.write('=' * 60 + '\n')
    
    print('=' * 70)
    print('EVALUATION COMPLETE!')
    print('=' * 70)
    print('Samples processed: {}'.format(test_number))
    print('')
    print('TIMING:')
    print('  Average Stage 1 time: {:.3f}s ± {:.3f}s'.format(avg_stage1, np.std(stage1_times)))
    print('  Average Stage 2 time: {:.3f}s ± {:.3f}s'.format(avg_stage2, np.std(stage2_times)))
    print('  Average Total time:   {:.3f}s ± {:.3f}s'.format(avg_time, std_time))
    print('  Min time: {:.3f}s | Max time: {:.3f}s'.format(min_time, max_time))
    print('  Total processing time: {:.2f}s ({:.2f} minutes)'.format(sum(total_times), sum(total_times)/60))
    print('')
    print('QUALITY METRICS (tau={}):'.format(TAU))
    print('  Chamfer Distance: {:.8f} ± {:.8f}'.format(avg_cd, std_cd))
    print('  F1@tau:           {:.2f}%'.format(avg_f1_tau))
    print('  F1@2tau:          {:.2f}%'.format(avg_f1_2tau))
    print('')
    print('PER-CATEGORY METRICS:')
    category_names = {
        '02691156': 'plane', '02958343': 'car', '03001627': 'chair',
        '03636649': 'lamp', '03691459': 'speaker', '04379243': 'table'
    }
    for cat_id, cat_data in sorted(category_metrics.items()):
        cat_name = category_names.get(cat_id, cat_id)
        print('  {:8s} ({:8s}): CD={:.6f}  F1@tau={:.1f}%  F1@2tau={:.1f}%'.format(
            cat_id, cat_name,
            np.mean(cat_data['cd']),
            np.mean(cat_data['f1_tau']),
            np.mean(cat_data['f1_2tau'])
        ))
    print('=' * 70)
    print('Outputs:')
    print('  Meshes (.obj):     {}'.format(output_dir))
    print('  Timing (detailed): {}'.format(timing_file))
    print('  Metrics (detailed):{}'.format(metrics_file))
    print('  Summary stats:     {}'.format(summary_file))
    print('=' * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Design A Complete Evaluation')
    parser.add_argument('--eval_list', type=str, 
                        default='designA_eval_list.txt',
                        help='Path to evaluation list file')
    parser.add_argument('--output_dir', type=str,
                        default='../outputs/designA/eval_meshes',
                        help='Output directory for meshes')
    
    args = parser.parse_args()
    main(args.eval_list, args.output_dir)
