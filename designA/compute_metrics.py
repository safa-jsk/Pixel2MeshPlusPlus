#!/usr/bin/env python
# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
#
# Design A - Compute Metrics (Chamfer Distance, F1@tau, F1@2tau)
# Run after evaluation to compute quality metrics on generated meshes
"""
Computes reconstruction quality metrics on Design A evaluation outputs.

Metrics:
  - Chamfer Distance (CD): Bidirectional point cloud distance
  - F1@tau: F1-score at threshold tau (default: 0.0001)
  - F1@2tau: F1-score at threshold 2*tau

Usage:
    python compute_metrics.py --mesh-dir ../outputs/designA/eval_meshes
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import argparse
import glob
import time

# Use TensorFlow for Chamfer Distance computation
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from modules.chamfer import nn_distance


def compute_f1_score(d1, d2, threshold):
    """
    Compute F1-score at a given threshold.
    
    Args:
        d1: Distances from pred to gt (N,)
        d2: Distances from gt to pred (M,)
        threshold: Distance threshold for matching
    
    Returns:
        f1, precision, recall
    """
    precision = 100.0 * (np.sum(d1 <= threshold) / len(d1))
    recall = 100.0 * (np.sum(d2 <= threshold) / len(d2))
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)
    return f1, precision, recall


# [DESIGN.A][CAMFM.A3_METRICS] Quality metrics computation
def main(mesh_dir, tau=0.0001):
    print('=' * 70)
    print('Design A - Quality Metrics Computation')
    print('=' * 70)
    print('Mesh directory: {}'.format(mesh_dir))
    print('Threshold tau: {}'.format(tau))
    print('=' * 70)
    
    tau_2 = 2 * tau
    
    # Find all prediction files
    pred_files = sorted(glob.glob(os.path.join(mesh_dir, '*_predict.xyz')))
    if not pred_files:
        print('ERROR: No *_predict.xyz files found in {}'.format(mesh_dir))
        return
    
    print('Found {} prediction files'.format(len(pred_files)))
    print('')
    
    # Setup TensorFlow
    print('=> Setting up TensorFlow for Chamfer Distance...')
    xyz1_ph = tf.placeholder(tf.float32, shape=(None, 3), name='pred_pts')
    xyz2_ph = tf.placeholder(tf.float32, shape=(None, 3), name='gt_pts')
    dist1_op, idx1_op, dist2_op, idx2_op = nn_distance(xyz1_ph, xyz2_ph)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    
    # Category names for ShapeNet
    category_names = {
        '02691156': 'plane',
        '02958343': 'car',
        '03001627': 'chair',
        '03636649': 'lamp',
        '03691459': 'speaker',
        '04379243': 'table',
        '02828884': 'bench',
        '04090263': 'firearm',
        '04530566': 'watercraft',
        '02933112': 'cabinet',
        '03211117': 'monitor',
        '04256520': 'couch',
        '04401088': 'cellphone',
    }
    
    # Storage
    all_results = []
    category_metrics = {}
    
    print('')
    print('Computing metrics for each sample...')
    print('-' * 90)
    print('{:<45} {:>12} {:>10} {:>10}'.format('Sample', 'CD (×10⁻³)', 'F1@τ (%)', 'F1@2τ (%)'))
    print('-' * 90)
    
    for i, pred_path in enumerate(pred_files):
        # Load prediction
        pred_pts = np.loadtxt(pred_path).astype(np.float32)
        
        # Load ground truth
        gt_path = pred_path.replace('_predict.xyz', '_ground.xyz')
        if not os.path.exists(gt_path):
            print('[{}/{}] SKIP: Ground truth not found for {}'.format(i+1, len(pred_files), os.path.basename(pred_path)))
            continue
        
        gt_pts = np.loadtxt(gt_path).astype(np.float32)
        if gt_pts.shape[1] > 3:
            gt_pts = gt_pts[:, :3]  # Take first 3 columns if more exist
        
        # Compute NN distances
        d1, _, d2, _ = sess.run(
            [dist1_op, idx1_op, dist2_op, idx2_op],
            feed_dict={xyz1_ph: pred_pts, xyz2_ph: gt_pts}
        )
        d1 = np.squeeze(d1)
        d2 = np.squeeze(d2)
        
        # Chamfer Distance
        chamfer_dist = np.mean(d1) + np.mean(d2)
        
        # F1@tau
        f1_tau, prec_tau, rec_tau = compute_f1_score(d1, d2, tau)
        
        # F1@2tau
        f1_2tau, prec_2tau, rec_2tau = compute_f1_score(d1, d2, tau_2)
        
        # Extract sample info
        basename = os.path.basename(pred_path).replace('_predict.xyz', '')
        category_id = basename.split('_')[0]
        
        result = {
            'sample_id': basename + '.dat',
            'chamfer_distance': chamfer_dist,
            'f1_tau': f1_tau,
            'f1_2tau': f1_2tau,
            'precision_tau': prec_tau,
            'recall_tau': rec_tau,
            'precision_2tau': prec_2tau,
            'recall_2tau': rec_2tau,
        }
        all_results.append(result)
        
        # Aggregate by category
        if category_id not in category_metrics:
            category_metrics[category_id] = {'cd': [], 'f1_tau': [], 'f1_2tau': []}
        category_metrics[category_id]['cd'].append(chamfer_dist)
        category_metrics[category_id]['f1_tau'].append(f1_tau)
        category_metrics[category_id]['f1_2tau'].append(f1_2tau)
        
        # Print progress
        print('{:<45} {:>12.4f} {:>10.2f} {:>10.2f}'.format(
            basename[:45],
            chamfer_dist * 1000,  # Scale for readability
            f1_tau,
            f1_2tau
        ))
    
    print('-' * 90)
    sess.close()
    
    if not all_results:
        print('ERROR: No valid results computed')
        return
    
    # Compute overall averages
    avg_cd = np.mean([r['chamfer_distance'] for r in all_results])
    std_cd = np.std([r['chamfer_distance'] for r in all_results])
    avg_f1_tau = np.mean([r['f1_tau'] for r in all_results])
    avg_f1_2tau = np.mean([r['f1_2tau'] for r in all_results])
    
    print('')
    print('=' * 70)
    print('OVERALL METRICS SUMMARY')
    print('=' * 70)
    print('  Samples evaluated: {}'.format(len(all_results)))
    print('  Threshold tau: {}'.format(tau))
    print('')
    print('  Chamfer Distance: {:.8f} ± {:.8f}  (×10⁻³: {:.4f})'.format(avg_cd, std_cd, avg_cd * 1000))
    print('  F1@tau:           {:.2f}%'.format(avg_f1_tau))
    print('  F1@2tau:          {:.2f}%'.format(avg_f1_2tau))
    print('')
    
    # Per-category breakdown
    print('PER-CATEGORY METRICS:')
    print('-' * 70)
    print('{:<10} {:<10} {:>8} {:>12} {:>10} {:>10}'.format(
        'Category', 'Name', 'Count', 'CD (×10⁻³)', 'F1@τ (%)', 'F1@2τ (%)'))
    print('-' * 70)
    
    for cat_id in sorted(category_metrics.keys()):
        cat_data = category_metrics[cat_id]
        cat_name = category_names.get(cat_id, 'unknown')
        print('{:<10} {:<10} {:>8} {:>12.4f} {:>10.2f} {:>10.2f}'.format(
            cat_id,
            cat_name,
            len(cat_data['cd']),
            np.mean(cat_data['cd']) * 1000,
            np.mean(cat_data['f1_tau']),
            np.mean(cat_data['f1_2tau'])
        ))
    print('-' * 70)
    
    # Save results
    benchmark_dir = mesh_dir.replace('eval_meshes', 'benchmark')
    os.makedirs(benchmark_dir, exist_ok=True)
    
    # Save detailed CSV
    metrics_csv = os.path.join(benchmark_dir, 'metrics_results.csv')
    with open(metrics_csv, 'w') as f:
        f.write('sample_id,chamfer_distance,f1_tau,f1_2tau,precision_tau,recall_tau,precision_2tau,recall_2tau\n')
        for r in all_results:
            f.write('{},{:.8f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(
                r['sample_id'],
                r['chamfer_distance'],
                r['f1_tau'],
                r['f1_2tau'],
                r['precision_tau'],
                r['recall_tau'],
                r['precision_2tau'],
                r['recall_2tau']
            ))
    print('')
    print('Saved detailed metrics to: {}'.format(metrics_csv))
    
    # Save summary
    summary_file = os.path.join(benchmark_dir, 'metrics_summary.txt')
    with open(summary_file, 'w') as f:
        f.write('Design A Quality Metrics Summary\n')
        f.write('=' * 60 + '\n')
        f.write('Date: {}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
        f.write('Samples: {}\n'.format(len(all_results)))
        f.write('Threshold tau: {}\n'.format(tau))
        f.write('\n')
        f.write('Overall Metrics:\n')
        f.write('-' * 60 + '\n')
        f.write('  Chamfer Distance: {:.8f} ± {:.8f}\n'.format(avg_cd, std_cd))
        f.write('  F1@tau:           {:.2f}%\n'.format(avg_f1_tau))
        f.write('  F1@2tau:          {:.2f}%\n'.format(avg_f1_2tau))
        f.write('\n')
        f.write('Per-Category Metrics:\n')
        f.write('-' * 60 + '\n')
        for cat_id in sorted(category_metrics.keys()):
            cat_data = category_metrics[cat_id]
            cat_name = category_names.get(cat_id, 'unknown')
            f.write('  {} ({}): CD={:.6f}  F1@tau={:.2f}%  F1@2tau={:.2f}%  (n={})\n'.format(
                cat_id, cat_name,
                np.mean(cat_data['cd']),
                np.mean(cat_data['f1_tau']),
                np.mean(cat_data['f1_2tau']),
                len(cat_data['cd'])
            ))
        f.write('=' * 60 + '\n')
    print('Saved summary to: {}'.format(summary_file))
    
    print('')
    print('=' * 70)
    print('METRICS COMPUTATION COMPLETE!')
    print('=' * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute metrics for Design A evaluation')
    parser.add_argument('--mesh-dir', type=str, 
                        default='../outputs/designA/eval_meshes',
                        help='Directory containing *_predict.xyz and *_ground.xyz files')
    parser.add_argument('--tau', type=float, default=0.0001,
                        help='Threshold for F1-score (default: 0.0001)')
    args = parser.parse_args()
    
    main(args.mesh_dir, args.tau)
