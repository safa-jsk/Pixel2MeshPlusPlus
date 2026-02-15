# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
#
# Design A Metrics Module
# Provides: Chamfer Distance, F1@tau, F1@2tau
"""
Metrics for 3D mesh reconstruction evaluation.

This module integrates with external/tf_nndistance_cpu.py (or the compiled 
tf_nndistance_so.so if available) to compute:
  1. Chamfer Distance (CD) - bidirectional point cloud distance
  2. F1@tau - F1-score at threshold tau
  3. F1@2tau - F1-score at threshold 2*tau

Usage:
    from metrics import MetricsCalculator
    
    calc = MetricsCalculator(tau=0.0001)  # tau = 10^-4
    results = calc.compute_all(prediction, ground_truth)
    print(f"CD: {results['chamfer_distance']:.6f}")
    print(f"F1@tau: {results['f1_tau']:.2f}%")
    print(f"F1_2tau: {results['f1_2tau']:.2f}%")
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Import the Chamfer distance module (GPU or CPU fallback)
from modules.chamfer import nn_distance


class MetricsCalculator:
    """
    Compute 3D reconstruction metrics using TensorFlow.
    
    Attributes:
        tau: Base threshold for F1-score computation (default: 0.0001)
        sess: TensorFlow session (created if not provided)
    """
    
    def __init__(self, tau=0.0001, session=None):
        """
        Initialize MetricsCalculator.
        
        Args:
            tau: Base threshold for F1-score (tau and 2*tau used)
            session: Optional existing TF session (shares session for efficiency)
        """
        self.tau = tau
        self.tau_2 = 2 * tau
        
        # Placeholders for point clouds
        self.xyz1_ph = tf.placeholder(tf.float32, shape=(None, 3), name='pred_points')
        self.xyz2_ph = tf.placeholder(tf.float32, shape=(None, 3), name='gt_points')
        
        # NN-distance ops (from modules/chamfer.py)
        self.dist1, self.idx1, self.dist2, self.idx2 = nn_distance(self.xyz1_ph, self.xyz2_ph)
        
        # Session management
        if session is not None:
            self.sess = session
            self._owns_session = False
        else:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            self.sess = tf.Session(config=config)
            self._owns_session = True
    
    def compute_chamfer_distance(self, pred, gt):
        """
        Compute Chamfer Distance between predicted and ground truth point clouds.
        
        CD = mean(dist1) + mean(dist2)
        where:
          dist1[i] = min_j ||pred[i] - gt[j]||^2
          dist2[j] = min_i ||gt[j] - pred[i]||^2
        
        Args:
            pred: Predicted point cloud (N, 3)
            gt: Ground truth point cloud (M, 3)
        
        Returns:
            chamfer_distance: Scalar value (sum of mean distances)
        """
        d1, _, d2, _ = self.sess.run(
            [self.dist1, self.idx1, self.dist2, self.idx2],
            feed_dict={self.xyz1_ph: pred, self.xyz2_ph: gt}
        )
        # Remove batch dimension if present
        d1 = np.squeeze(d1)
        d2 = np.squeeze(d2)
        chamfer_distance = np.mean(d1) + np.mean(d2)
        return chamfer_distance
    
    def compute_f1_score(self, pred, gt, threshold):
        """
        Compute F1-score at a given distance threshold.
        
        Precision = % of pred points within threshold of any gt point
        Recall    = % of gt points within threshold of any pred point
        F1 = 2 * P * R / (P + R)
        
        Args:
            pred: Predicted point cloud (N, 3)
            gt: Ground truth point cloud (M, 3)
            threshold: Distance threshold for matching
        
        Returns:
            f1_score: F1-score as percentage (0-100)
            precision: Precision as percentage
            recall: Recall as percentage
        """
        d1, _, d2, _ = self.sess.run(
            [self.dist1, self.idx1, self.dist2, self.idx2],
            feed_dict={self.xyz1_ph: pred, self.xyz2_ph: gt}
        )
        # Remove batch dimension if present
        d1 = np.squeeze(d1)
        d2 = np.squeeze(d2)
        
        # Precision: pred→gt
        num_pred_match = np.sum(d1 <= threshold)
        precision = 100.0 * (num_pred_match / len(d1))
        
        # Recall: gt→pred
        num_gt_match = np.sum(d2 <= threshold)
        recall = 100.0 * (num_gt_match / len(d2))
        
        # F1-score
        f1 = (2 * precision * recall) / (precision + recall + 1e-6)
        
        return f1, precision, recall
    
    def compute_all(self, pred, gt):
        """
        Compute all metrics in one call (most efficient).
        
        Args:
            pred: Predicted point cloud (N, 3)
            gt: Ground truth point cloud (M, 3)
        
        Returns:
            dict with keys:
              - chamfer_distance
              - f1_tau
              - f1_2tau
              - precision_tau
              - recall_tau
              - precision_2tau
              - recall_2tau
              - tau (threshold used)
        """
        # Single NN-distance call for all metrics
        d1, _, d2, _ = self.sess.run(
            [self.dist1, self.idx1, self.dist2, self.idx2],
            feed_dict={self.xyz1_ph: pred, self.xyz2_ph: gt}
        )
        # Remove batch dimension
        d1 = np.squeeze(d1)
        d2 = np.squeeze(d2)
        
        # Chamfer Distance
        chamfer_distance = np.mean(d1) + np.mean(d2)
        
        # F1@tau
        num_pred_tau = np.sum(d1 <= self.tau)
        num_gt_tau = np.sum(d2 <= self.tau)
        precision_tau = 100.0 * (num_pred_tau / len(d1))
        recall_tau = 100.0 * (num_gt_tau / len(d2))
        f1_tau = (2 * precision_tau * recall_tau) / (precision_tau + recall_tau + 1e-6)
        
        # F1@2tau
        num_pred_2tau = np.sum(d1 <= self.tau_2)
        num_gt_2tau = np.sum(d2 <= self.tau_2)
        precision_2tau = 100.0 * (num_pred_2tau / len(d1))
        recall_2tau = 100.0 * (num_gt_2tau / len(d2))
        f1_2tau = (2 * precision_2tau * recall_2tau) / (precision_2tau + recall_2tau + 1e-6)
        
        return {
            'chamfer_distance': chamfer_distance,
            'f1_tau': f1_tau,
            'f1_2tau': f1_2tau,
            'precision_tau': precision_tau,
            'recall_tau': recall_tau,
            'precision_2tau': precision_2tau,
            'recall_2tau': recall_2tau,
            'tau': self.tau,
        }
    
    def close(self):
        """Close TensorFlow session if we own it."""
        if self._owns_session and self.sess is not None:
            self.sess.close()
            self.sess = None
    
    def __del__(self):
        """Destructor - close session."""
        self.close()


def format_metrics_table(results_list, sample_ids=None):
    """
    Format metrics results as a printable table.
    
    Args:
        results_list: List of result dicts from compute_all()
        sample_ids: Optional list of sample IDs (same length as results_list)
    
    Returns:
        str: Formatted table string
    """
    header = '{:<50} {:>12} {:>10} {:>10}'.format('Sample', 'CD (×10⁻³)', 'F1@τ (%)', 'F1@2τ (%)')
    sep = '-' * 90
    lines = [sep, header, sep]
    
    for i, r in enumerate(results_list):
        sample_id = sample_ids[i] if sample_ids else f'Sample {i+1}'
        # Display CD scaled by 10^3 for readability
        lines.append('{:<50} {:>12.4f} {:>10.2f} {:>10.2f}'.format(
            sample_id[:50],
            r['chamfer_distance'] * 1000,  # Scale for readability
            r['f1_tau'],
            r['f1_2tau']
        ))
    
    lines.append(sep)
    
    # Compute averages
    if results_list:
        avg_cd = np.mean([r['chamfer_distance'] for r in results_list])
        avg_f1_tau = np.mean([r['f1_tau'] for r in results_list])
        avg_f1_2tau = np.mean([r['f1_2tau'] for r in results_list])
        lines.append('{:<50} {:>12.4f} {:>10.2f} {:>10.2f}'.format(
            'AVERAGE',
            avg_cd * 1000,
            avg_f1_tau,
            avg_f1_2tau
        ))
        lines.append(sep)
    
    return '\n'.join(lines)


# Convenience function for standalone usage
def compute_metrics_for_sample(pred_path, gt_path, tau=0.0001):
    """
    Compute metrics for a single sample (standalone usage).
    
    Args:
        pred_path: Path to predicted .xyz file (N, 3)
        gt_path: Path to ground truth .xyz file (M, 3 or M, 6)
        tau: F1-score threshold
    
    Returns:
        dict with all metrics
    """
    pred = np.loadtxt(pred_path)
    gt = np.loadtxt(gt_path)[:, :3]  # Ground truth may have 6 columns
    
    calc = MetricsCalculator(tau=tau)
    results = calc.compute_all(pred, gt)
    calc.close()
    
    return results


if __name__ == '__main__':
    # Test with example data
    print('Testing MetricsCalculator...')
    
    # Create random test data
    np.random.seed(42)
    pred = np.random.randn(2466, 3).astype(np.float32) * 0.1
    gt = np.random.randn(2466, 3).astype(np.float32) * 0.1
    
    calc = MetricsCalculator(tau=0.0001)
    results = calc.compute_all(pred, gt)
    
    print('\nMetrics Results:')
    print('-' * 50)
    print(f"  Chamfer Distance: {results['chamfer_distance']:.6f}")
    print(f"  F1@tau ({results['tau']:.5f}): {results['f1_tau']:.2f}%")
    print(f"  F1@2tau ({results['tau']*2:.5f}): {results['f1_2tau']:.2f}%")
    print(f"  Precision@tau: {results['precision_tau']:.2f}%")
    print(f"  Recall@tau: {results['recall_tau']:.2f}%")
    print('-' * 50)
    
    calc.close()
    print('\nTest complete!')
