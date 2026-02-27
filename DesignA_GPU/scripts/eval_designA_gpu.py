# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
# 
# Design A with GPU ONLY (no other optimizations)
# This is a direct copy of Design A with GPU enabled instead of CPU-only
import sys
import os

# Add parent directory to path for module imports
# --- Path bootstrap (post-refactor) ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
_TF_SRC = os.path.join(_PROJECT_ROOT, 'src', 'p2mpp', 'tf')
if _TF_SRC not in sys.path:
    sys.path.insert(0, _TF_SRC)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# --- End path bootstrap ---

# Use TensorFlow 2.x with TF1 compatibility mode
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tflearn
import numpy as np
import pickle
import time
import argparse

from modules.models_p2mpp import MeshNet
from modules.config import create_parser, parse_args
from utils.dataloader import DataFetcher
from utils.tools import construct_feed_dict


def main(eval_list_file, output_dir, gpu_id=0):
    # [DESIGN.A_GPU] GPU CONFIGURATION - Enable GPU (Design A was CPU-only)
    # ============================================================
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print('=' * 70)
    print('DESIGN A - GPU ENABLED')
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
                print('  - {}: {}'.format(d.name, d.physical_device_desc[:60]))
        else:
            print('WARNING: No GPU devices found! Running on CPU.')
    except Exception as e:
        print('Could not enumerate devices: {}'.format(e))
    
    print('=' * 70)
    
    # ---------------------------------------------------------------
    # Set random seed
    print('=> Design A GPU Batch Evaluation')
    print('=> Eval list: {}'.format(eval_list_file))
    print('=> Output dir: {}'.format(output_dir))
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

    # Paths (relative to project root)
    model_dir = '../results/refine_p2mpp/models'
    data_root = '../data/p2mppdata/test'
    image_root = '../data/ShapeNetImages/ShapeNetRendering'
    mesh_root = '../results/coarse_mvp2m/predict/50'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('==> Created output_dir {}'.format(output_dir))
    
    # Create config object
    parser = create_parser()
    args = parser.parse_args([])
    args.gpu_id = gpu_id
    args.test_epoch = 10
    
    # -------------------------------------------------------------------
    print('=> Build model')
    model = MeshNet(placeholders, logging=True, args=args)
    # ---------------------------------------------------------------
    print('=> Load data')
    data = DataFetcher(
        file_list=eval_list_file,
        data_root=data_root,
        image_root=image_root,
        is_val=True,
        mesh_root=mesh_root
    )
    data.setDaemon(True)
    data.start()
    # ---------------------------------------------------------------
    print('=> Initialize session')
    # GPU Configuration
    sesscfg = tf.ConfigProto()
    sesscfg.gpu_options.allow_growth = True  # Don't allocate all GPU memory
    sesscfg.allow_soft_placement = True      # Allow CPU fallback if needed
    sesscfg.log_device_placement = False     # Set True to see device placement
    
    sess = tf.Session(config=sesscfg)
    sess.run(tf.global_variables_initializer())
    
    # Verify GPU usage
    print('=> Checking TensorFlow device placement...')
    print('   Default device: {}'.format(sess.list_devices()))
    
    # ---------------------------------------------------------------
    print('=> Load checkpoint')
    model.load(sess=sess, ckpt_path=model_dir, step=10)
    # ---------------------------------------------------------------
    # Load init ellipsoid and info about vertices and edges
    pkl = pickle.load(open('../data/iccv_p2mpp.dat', 'rb'))
    feed_dict = construct_feed_dict(pkl, placeholders)
    # ---------------------------------------------------------------
    test_number = data.number
    tflearn.is_training(False, sess)
    
    print('=> Start inference on {} samples'.format(test_number))
    print('=' * 70)
    
    timing_results = []
    
    # Warmup run (important for GPU)
    print('=> Warmup run...')
    img_all_view, labels, poses, data_id, mesh = data.fetch()
    feed_dict.update({placeholders['img_inp']: img_all_view})
    feed_dict.update({placeholders['features']: mesh})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['cameras']: poses})
    _ = sess.run([model.output1l, model.output2l], feed_dict=feed_dict)
    
    # Re-queue the warmup sample data
    data.shutdown()
    data = DataFetcher(
        file_list=eval_list_file,
        data_root=data_root,
        image_root=image_root,
        is_val=True,
        mesh_root=mesh_root
    )
    data.setDaemon(True)
    data.start()
    test_number = data.number
    
    print('=> Warmup complete. Starting timed evaluation...')
    print('=' * 70)
    
    total_start = time.time()
    
    for iters in range(test_number):
        # Fetch data
        img_all_view, labels, poses, data_id, mesh = data.fetch()
        feed_dict.update({placeholders['img_inp']: img_all_view})
        feed_dict.update({placeholders['features']: mesh})
        feed_dict.update({placeholders['labels']: labels})
        feed_dict.update({placeholders['cameras']: poses})
        
        # Timing start
        start_time = time.time()
        
        # Run inference
        out1l, out2l = sess.run([model.output1l, model.output2l], feed_dict=feed_dict)
        
        # Timing end
        elapsed = time.time() - start_time
        timing_results.append((data_id, elapsed))
        
        # Save ground truth
        label_path = os.path.join(output_dir, data_id.replace('.dat', '_ground.xyz'))
        np.savetxt(label_path, labels)
        
        # Save prediction
        pred_path = os.path.join(output_dir, data_id.replace('.dat', '_predict.xyz'))
        np.savetxt(pred_path, out2l)
        
        # Save as OBJ (vertices only, faces from template)
        obj_path = os.path.join(output_dir, data_id.replace('.dat', '_predict.obj'))
        vert = np.hstack((np.full([out2l.shape[0], 1], 'v'), out2l))
        face = np.loadtxt('../data/face3.obj', dtype='|S32')
        mesh_data = np.vstack((vert, face))
        np.savetxt(obj_path, mesh_data, fmt='%s', delimiter=' ')
        
        print('[{}/{}] {} - {:.3f}s ({:.1f}ms)'.format(
            iters + 1, test_number, data_id, elapsed, elapsed * 1000))
    
    total_elapsed = time.time() - total_start
    
    # ---------------------------------------------------------------
    data.shutdown()
    
    # Save timing results
    timing_file = os.path.join(output_dir, 'timing_results.csv')
    with open(timing_file, 'w') as f:
        f.write('sample_id,time_sec,time_ms\n')
        for data_id, elapsed in timing_results:
            f.write('{},{:.4f},{:.2f}\n'.format(data_id, elapsed, elapsed * 1000))
    
    # Calculate statistics
    times = [t for _, t in timing_results]
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    throughput = 1.0 / avg_time if avg_time > 0 else 0
    
    # Save summary
    summary_file = os.path.join(output_dir, 'evaluation_summary.txt')
    with open(summary_file, 'w') as f:
        f.write('=' * 70 + '\n')
        f.write('DESIGN A - GPU EVALUATION SUMMARY\n')
        f.write('=' * 70 + '\n')
        f.write('Configuration:\n')
        f.write('  GPU Enabled: YES\n')
        f.write('  GPU ID: {}\n'.format(gpu_id))
        f.write('  Framework: TensorFlow {}\n'.format(tf.__version__))
        f.write('  Eval List: {}\n'.format(eval_list_file))
        f.write('=' * 70 + '\n')
        f.write('Performance Metrics:\n')
        f.write('  Samples processed: {}\n'.format(test_number))
        f.write('  Mean latency: {:.2f}ms ± {:.2f}ms\n'.format(avg_time * 1000, std_time * 1000))
        f.write('  Min latency: {:.2f}ms\n'.format(min_time * 1000))
        f.write('  Max latency: {:.2f}ms\n'.format(max_time * 1000))
        f.write('  Throughput: {:.2f} samples/sec\n'.format(throughput))
        f.write('  Total time: {:.2f}s ({:.2f}min)\n'.format(total_elapsed, total_elapsed / 60))
        f.write('=' * 70 + '\n')
    
    print('=' * 70)
    print('DESIGN A - GPU EVALUATION COMPLETE')
    print('=' * 70)
    print('Configuration:')
    print('  GPU Enabled: YES')
    print('  GPU ID: {}'.format(gpu_id))
    print('  Framework: TensorFlow {}'.format(tf.__version__))
    print('=' * 70)
    print('Performance Metrics:')
    print('  Samples processed: {}'.format(test_number))
    print('  Mean latency: {:.2f}ms ± {:.2f}ms'.format(avg_time * 1000, std_time * 1000))
    print('  Min latency: {:.2f}ms'.format(min_time * 1000))
    print('  Max latency: {:.2f}ms'.format(max_time * 1000))
    print('  Throughput: {:.2f} samples/sec'.format(throughput))
    print('  Total time: {:.2f}s ({:.2f}min)'.format(total_elapsed, total_elapsed / 60))
    print('=' * 70)
    print('Outputs saved to: {}'.format(output_dir))
    print('Timing data: {}'.format(timing_file))
    print('Summary: {}'.format(summary_file))
    print('=' * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Design A GPU Batch Evaluation')
    parser.add_argument('--eval_list', type=str, 
                        default='../data/designA_eval_1000.txt',
                        help='Path to evaluation list file')
    parser.add_argument('--output_dir', type=str,
                        default='../outputs/designA_GPU/eval_meshes',
                        help='Output directory for meshes')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    
    args = parser.parse_args()
    main(args.eval_list, args.output_dir, args.gpu_id)
