# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
# 
# Modified for Design A batch evaluation
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

import tensorflow as tf
import tflearn
import numpy as np
import pickle
import time
import argparse

from modules.models_p2mpp import MeshNet
from modules.config import create_parser, parse_args
from utils.dataloader import DataFetcher
from utils.tools import construct_feed_dict


def main(eval_list_file, output_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
    # ---------------------------------------------------------------
    # Set random seed
    print('=> Design A Batch Evaluation')
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
    args.gpu_id = 0
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
    sesscfg = tf.ConfigProto()
    sesscfg.gpu_options.allow_growth = True
    sesscfg.allow_soft_placement = True
    sess = tf.Session(config=sesscfg)
    sess.run(tf.global_variables_initializer())
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
    print('=' * 60)
    
    timing_results = []
    
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
        
        print('[{}/{}] {} - {:.3f}s'.format(
            iters + 1, test_number, data_id, elapsed))
    
    # ---------------------------------------------------------------
    data.shutdown()
    
    # Save timing results
    timing_file = os.path.join(output_dir, '../benchmark/timing_results.csv')
    os.makedirs(os.path.dirname(timing_file), exist_ok=True)
    with open(timing_file, 'w') as f:
        f.write('sample_id,time_sec\n')
        for data_id, elapsed in timing_results:
            f.write('{},{:.4f}\n'.format(data_id, elapsed))
    
    # Calculate statistics
    times = [t for _, t in timing_results]
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print('=' * 60)
    print('Evaluation Complete!')
    print('=' * 60)
    print('Samples processed: {}'.format(test_number))
    print('Average time/sample: {:.3f}s ± {:.3f}s'.format(avg_time, std_time))
    print('Min time: {:.3f}s'.format(min_time))
    print('Max time: {:.3f}s'.format(max_time))
    print('Total time: {:.2f}s ({:.2f}min)'.format(sum(times), sum(times)/60))
    print('=' * 60)
    print('Outputs saved to: {}'.format(output_dir))
    print('Timing data saved to: {}'.format(timing_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Design A Batch Evaluation')
    parser.add_argument('--eval_list', type=str, 
                        default='designA_eval_list.txt',
                        help='Path to evaluation list file')
    parser.add_argument('--output_dir', type=str,
                        default='../outputs/designA/eval_meshes',
                        help='Output directory for meshes')
    
    args = parser.parse_args()
    main(args.eval_list, args.output_dir)
