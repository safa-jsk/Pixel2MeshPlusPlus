# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
# 
# Design A Complete Evaluation with GPU ONLY (Stage 1 + Stage 2)
# With metrics: Chamfer Distance, F1@tau, F1@2tau
# This is Design A with GPU enabled - no other optimizations from Design B
import sys
import os

# Add parent directory to path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Use TensorFlow 2.x with TF1 compatibility mode
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

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


def main(eval_list_file, output_dir, gpu_id=0):
    # ============================================================
    # GPU CONFIGURATION - Enable GPU (Design A was CPU-only)
    # ============================================================
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print('=' * 70)
    print('DESIGN A - GPU ENABLED (Complete 2-Stage Pipeline)')
    print('With Metrics: Chamfer Distance, F1@tau, F1@2tau')
    print('=' * 70)
    print('GPU ID: {}'.format(gpu_id))
    
    # Check if GPU is available
    try:
        from tensorflow.python.client import device_lib
        devices = device_lib.list_local_devices()
        gpu_devices = [d for d in devices if d.device_type == 'GPU']
        if gpu_devices:
            print('GPU Devices Found: {}'.format(len(gpu_devices)))
            for d in gpu_devices:
                print('  - {}: {}'.format(d.name, d.physical_device_desc[:60] if d.physical_device_desc else 'N/A'))
        else:
            print('WARNING: No GPU devices found! Running on CPU.')
    except Exception as e:
        print('Could not enumerate devices: {}'.format(e))
    
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

    # Paths (relative to project root - running from designA_GPU/ folder)
    model1_dir = '../results/coarse_mvp2m/models'
    model2_dir = '../results/refine_p2mpp/models'
    data_root = '../data/p2mppdata/test'
    image_root = '../data/ShapeNetImages/ShapeNetRendering'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create config object
    parser = create_parser()
    args = parser.parse_args([])
    args.gpu_id = gpu_id
    
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
    print('=> Initializing TensorFlow session with GPU...')
    sesscfg = tf.ConfigProto()
    sesscfg.gpu_options.allow_growth = True
    sesscfg.allow_soft_placement = True
    sesscfg.log_device_placement = False
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
    
    # Warmup run (important for GPU)
    print('=> Running warmup iteration...')
    img_all_view, labels, poses, data_id, _ = data.fetch()
    feed_dict.update({placeholders['features']: initial_coords})
    feed_dict.update({placeholders['img_inp']: img_all_view})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['cameras']: poses})
    _ = sess.run(model1.output3, feed_dict=feed_dict)
    _ = sess.run(model2.output2l, feed_dict=feed_dict)
    
    # Re-initialize data loader
    data.shutdown()
    data = DataFetcher(
        file_list=eval_list_file,
        data_root=data_root,
        image_root=image_root,
        is_val=True,
        mesh_root=None
    )
    data.setDaemon(True)
    data.start()
    test_number = data.number
    
    print('=' * 70, flush=True)
    print('Starting 2-stage inference on {} samples (GPU ENABLED)'.format(test_number), flush=True)
    print('=' * 70, flush=True)
    
    timing_results = []
    stage1_times = []
    stage2_times = []
    
    # Metrics storage
    metrics_results = []
    category_metrics = {}  # Per-category aggregation
    
    total_start = time.time()
    
    for iters in range(test_number):
        # Fetch data
        img_all_view, labels, poses, data_id, _ = data.fetch()
        
        # Reset features to initial ellipsoid for Stage 1
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
        
        print('[{:4d}/{:4d}] {} | S1: {:.2f}s | S2: {:.2f}s | Tot: {:.2f}s ({:.1f}ms) | CD: {:.6f} | F1@tau: {:.1f}%'.format(
            iters + 1, test_number, data_id[:35], t1_elapsed, t2_elapsed, total_elapsed,
            total_elapsed * 1000, chamfer_dist, f1_tau), flush=True)
    
    total_wall_time = time.time() - total_start
    
    # ---------------------------------------------------------------
    data.shutdown()
    
    # Create benchmark directory
    benchmark_dir = os.path.join(output_dir, '../benchmark')
    os.makedirs(benchmark_dir, exist_ok=True)
    
    # Save detailed timing results
    timing_file = os.path.join(benchmark_dir, 'timing_results_detailed.csv')
    with open(timing_file, 'w') as f:
        f.write('sample_id,stage1_time_sec,stage2_time_sec,total_time_sec,total_time_ms\n')
        for data_id, t1, t2, total in timing_results:
            f.write('{},{:.4f},{:.4f},{:.4f},{:.2f}\n'.format(data_id, t1, t2, total, total * 1000))
    
    # Save metrics results
    metrics_file = os.path.join(benchmark_dir, 'metrics_results.csv')
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
    
    # Calculate timing statistics
    total_times = [t for _, _, _, t in timing_results]
    avg_time = np.mean(total_times)
    std_time = np.std(total_times)
    min_time = np.min(total_times)
    max_time = np.max(total_times)
    throughput = 1.0 / avg_time if avg_time > 0 else 0
    
    avg_stage1 = np.mean(stage1_times)
    avg_stage2 = np.mean(stage2_times)
    
    # Save summary statistics
    summary_file = os.path.join(benchmark_dir, 'summary_stats.txt')
    with open(summary_file, 'w') as f:
        f.write('=' * 70 + '\n')
        f.write('DESIGN A - GPU ENABLED - EVALUATION SUMMARY\n')
        f.write('=' * 70 + '\n')
        f.write('Configuration:\n')
        f.write('  GPU Enabled: YES\n')
        f.write('  GPU ID: {}\n'.format(gpu_id))
        f.write('  Framework: TensorFlow {}\n'.format(tf.__version__))
        f.write('  Eval List: {}\n'.format(eval_list_file))
        f.write('  Number of samples: {}\n'.format(test_number))
        f.write('  Metric threshold tau: {}\n'.format(TAU))
        f.write('\n')
        f.write('Timing Statistics (per sample):\n')
        f.write('-' * 70 + '\n')
        f.write('  Average Stage 1: {:.2f}ms ± {:.2f}ms\n'.format(avg_stage1 * 1000, np.std(stage1_times) * 1000))
        f.write('  Average Stage 2: {:.2f}ms ± {:.2f}ms\n'.format(avg_stage2 * 1000, np.std(stage2_times) * 1000))
        f.write('  Average Total:   {:.2f}ms ± {:.2f}ms\n'.format(avg_time * 1000, std_time * 1000))
        f.write('  Min time:        {:.2f}ms\n'.format(min_time * 1000))
        f.write('  Max time:        {:.2f}ms\n'.format(max_time * 1000))
        f.write('  Throughput:      {:.2f} samples/sec\n'.format(throughput))
        f.write('\n')
        f.write('Quality Metrics:\n')
        f.write('-' * 70 + '\n')
        f.write('  Chamfer Distance: {:.8f} ± {:.8f}\n'.format(avg_cd, std_cd))
        f.write('  Chamfer (x10^-3): {:.4f}\n'.format(avg_cd * 1000))
        f.write('  F1@tau:           {:.2f}%\n'.format(avg_f1_tau))
        f.write('  F1@2tau:          {:.2f}%\n'.format(avg_f1_2tau))
        f.write('\n')
        f.write('Per-Category Metrics:\n')
        f.write('-' * 70 + '\n')
        category_names = {
            '02691156': 'plane', '02958343': 'car', '03001627': 'chair',
            '03636649': 'lamp', '03691459': 'speaker', '04379243': 'table'
        }
        for cat_id, cat_data in sorted(category_metrics.items()):
            cat_name = category_names.get(cat_id, cat_id)
            f.write('  {:8s} ({:8s}): CD={:.6f}  F1@tau={:.2f}%  F1@2tau={:.2f}%  (n={})\n'.format(
                cat_id, cat_name,
                np.mean(cat_data['cd']),
                np.mean(cat_data['f1_tau']),
                np.mean(cat_data['f1_2tau']),
                len(cat_data['cd'])
            ))
        f.write('\n')
        f.write('Total Processing:\n')
        f.write('-' * 70 + '\n')
        f.write('  Total wall time: {:.2f}s ({:.2f} minutes)\n'.format(total_wall_time, total_wall_time/60))
        f.write('  Sum of sample times: {:.2f}s ({:.2f} minutes)\n'.format(sum(total_times), sum(total_times)/60))
        f.write('=' * 70 + '\n')
    
    print('=' * 70)
    print('DESIGN A - GPU ENABLED - EVALUATION COMPLETE!')
    print('=' * 70)
    print('Configuration:')
    print('  GPU Enabled: YES')
    print('  GPU ID: {}'.format(gpu_id))
    print('  Framework: TensorFlow {}'.format(tf.__version__))
    print('')
    print('TIMING (per sample):')
    print('  Average Stage 1: {:.2f}ms ± {:.2f}ms'.format(avg_stage1 * 1000, np.std(stage1_times) * 1000))
    print('  Average Stage 2: {:.2f}ms ± {:.2f}ms'.format(avg_stage2 * 1000, np.std(stage2_times) * 1000))
    print('  Average Total:   {:.2f}ms ± {:.2f}ms'.format(avg_time * 1000, std_time * 1000))
    print('  Min: {:.2f}ms | Max: {:.2f}ms'.format(min_time * 1000, max_time * 1000))
    print('  Throughput: {:.2f} samples/sec'.format(throughput))
    print('')
    print('QUALITY METRICS (tau={}):'.format(TAU))
    print('  Chamfer Distance: {:.6f} ± {:.6f} (×10⁻³: {:.4f})'.format(avg_cd, std_cd, avg_cd * 1000))
    print('  F1@tau:           {:.2f}%'.format(avg_f1_tau))
    print('  F1@2tau:          {:.2f}%'.format(avg_f1_2tau))
    print('')
    print('PER-CATEGORY METRICS:')
    for cat_id, cat_data in sorted(category_metrics.items()):
        cat_name = category_names.get(cat_id, cat_id)
        print('  {:8s} ({:8s}): CD={:.6f}  F1@tau={:.1f}%  F1@2tau={:.1f}%'.format(
            cat_id, cat_name,
            np.mean(cat_data['cd']),
            np.mean(cat_data['f1_tau']),
            np.mean(cat_data['f1_2tau'])
        ))
    print('=' * 70)
    print('Total wall time: {:.2f}s ({:.2f} minutes)'.format(total_wall_time, total_wall_time/60))
    print('=' * 70)
    print('Outputs:')
    print('  Meshes (.obj):      {}'.format(output_dir))
    print('  Timing (detailed):  {}'.format(timing_file))
    print('  Metrics (detailed): {}'.format(metrics_file))
    print('  Summary stats:      {}'.format(summary_file))
    print('=' * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Design A GPU Complete Evaluation')
    parser.add_argument('--eval_list', type=str, 
                        default='../data/designA_eval_1000.txt',
                        help='Path to evaluation list file')
    parser.add_argument('--output_dir', type=str,
                        default='../outputs/designA_GPU/eval_1000',
                        help='Output directory for meshes')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    
    args = parser.parse_args()
    main(args.eval_list, args.output_dir, args.gpu_id)
