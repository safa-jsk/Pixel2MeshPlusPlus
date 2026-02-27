#!/usr/bin/env python3
"""
Convert TensorFlow MVP2M (Stage 1) checkpoint to PyTorch format
"""

import os
import sys
import numpy as np

# --- Path bootstrap (post-refactor) ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TORCH_SRC = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_TORCH_SRC, '..', '..', '..'))
if _TORCH_SRC not in sys.path:
    sys.path.insert(0, _TORCH_SRC)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# --- End path bootstrap ---

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available - using pre-extracted weights")


def convert_mvp2m_checkpoint(ckpt_path, output_path):
    """
    Convert TensorFlow MVP2M checkpoint to PyTorch format
    
    Args:
        ckpt_path: Path to TensorFlow checkpoint (without extension)
        output_path: Path to save .npz file
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required for conversion")
    
    # Read checkpoint
    reader = tf.train.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    
    pytorch_state_dict = {}
    
    # Map TF variable names to PyTorch names
    for tf_name in sorted(var_to_shape_map.keys()):
        tensor = reader.get_tensor(tf_name)
        
        # Skip optimizer variables
        if 'Adam' in tf_name or 'beta' in tf_name or 'ExponentialMovingAverage' in tf_name:
            continue
        
        # CNN layers: meshnetmvp2m/cnn/conv2d_X/W:0 -> cnn.convX.weight
        if '/cnn/conv2d_' in tf_name:
            # Remove :0 suffix
            clean_name = tf_name.replace(':0', '')
            parts = clean_name.split('/')
            conv_num = int(parts[2].replace('conv2d_', ''))
            var_type = parts[3]
            
            if var_type == 'W':
                # TF: (H, W, C_in, C_out) -> PyTorch: (C_out, C_in, H, W)
                tensor = np.transpose(tensor, (3, 2, 0, 1))
                pt_name = f'cnn.conv{conv_num}.weight'
            elif var_type == 'b':
                pt_name = f'cnn.conv{conv_num}.bias'
            else:
                continue
                
            pytorch_state_dict[pt_name] = tensor
            print(f'  {tf_name} -> {pt_name}: {tensor.shape}')
        
        # GCN layers: meshnetmvp2m/pixel2mesh/graph_conv_blk{block}_{layer}_layer_{global}_vars/{var}
        elif '/pixel2mesh/graph_conv_blk' in tf_name:
            clean_name = tf_name.replace(':0', '')
            pt_name = convert_gcn_name(clean_name)
            if pt_name:
                pytorch_state_dict[pt_name] = tensor
                print(f'  {tf_name} -> {pt_name}: {tensor.shape}')
    
    print(f'\nTotal converted: {len(pytorch_state_dict)} tensors')
    np.savez(output_path, **pytorch_state_dict)
    print(f'Saved to: {output_path}')
    
    return pytorch_state_dict


def convert_gcn_name(tf_name):
    """Convert TensorFlow GCN layer name to PyTorch name"""
    # Example: meshnetmvp2m/pixel2mesh/graph_conv_blk1_1_layer_1_vars/weights_0
    # -> gcn1_layers.0.weights.0
    
    import re
    
    # Parse layer info
    match = re.search(r'graph_conv_blk(\d+)_(\d+)_layer_(\d+)_vars/(\w+)', tf_name)
    if not match:
        return None
    
    block_num = int(match.group(1))
    layer_in_block = int(match.group(2)) - 1  # Convert to 0-indexed
    var_name = match.group(4)
    
    block_name = f'gcn{block_num}_layers'
    
    if 'weights_' in var_name:
        weight_idx = int(var_name.replace('weights_', ''))
        pt_name = f'{block_name}.{layer_in_block}.weights.{weight_idx}'
    elif var_name == 'bias':
        pt_name = f'{block_name}.{layer_in_block}.bias'
    else:
        return None
    
    return pt_name


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, 
                        default='results/coarse_mvp2m/models/meshnetmvp2m.ckpt-50',
                        help='TensorFlow checkpoint path')
    parser.add_argument('--output', type=str,
                        default='pytorch_impl/checkpoints/mvp2m_converted.npz',
                        help='Output path for PyTorch weights')
    
    args = parser.parse_args()
    
    print('Converting MVP2M checkpoint...')
    convert_mvp2m_checkpoint(args.ckpt, args.output)
