#!/usr/bin/env python3
"""
TensorFlow to PyTorch Checkpoint Converter for Pixel2Mesh++

This script converts TensorFlow 1.x checkpoint to PyTorch state_dict format.
Run this inside the CPU Docker container (p2mpp:cpu) which has TensorFlow 1.15.

Usage:
    docker run --rm -v "$(pwd)":/workspace -w /workspace p2mpp:cpu \
        python pytorch_impl/convert_tf_to_pytorch.py \
            --tf_checkpoint results/refine_p2mpp/models/meshnet.ckpt-10 \
            --output pytorch_impl/checkpoints/meshnet_converted.pth
"""

import argparse
import numpy as np
import os
import sys


def convert_checkpoint(tf_ckpt_path, output_path):
    """
    Convert TensorFlow checkpoint to PyTorch format
    
    TensorFlow variable naming convention:
    - CNN: meshnet/cnn/conv2d_N/W:0, meshnet/cnn/conv2d_N/b:0
    - DRB1: meshnet/pixel2mesh/graph_drb_blk1_layer_2/localgconv_N_vars/...
    - DRB2: meshnet/pixel2mesh/graph_drb_blk2_layer_5/localgconv_N_vars/...
    
    PyTorch naming convention:
    - CNN: cnn.convN.weight, cnn.convN.bias
    - DRB1: drb1.local_convN.weights.0/1, drb1.local_convN.bias
    - DRB2: drb2.local_convN.weights.0/1, drb2.local_convN.bias
    """
    
    # Import TensorFlow (only available in p2mpp:cpu container)
    try:
        import tensorflow as tf
    except ImportError:
        print("ERROR: TensorFlow not available.")
        print("Run this script inside the TensorFlow Docker container:")
        print('  docker run --rm -v "$(pwd)":/workspace -w /workspace p2mpp:cpu python pytorch_impl/convert_tf_to_pytorch.py --tf_checkpoint results/refine_p2mpp/models/meshnet.ckpt-10 --output pytorch_impl/checkpoints/meshnet_converted.pth')
        sys.exit(1)
    
    print(f"Loading TensorFlow checkpoint: {tf_ckpt_path}")
    reader = tf.train.NewCheckpointReader(tf_ckpt_path)
    var_to_shape = reader.get_variable_to_shape_map()
    
    print(f"Found {len(var_to_shape)} variables in checkpoint")
    
    # Build weight mapping
    pytorch_state_dict = {}
    converted = 0
    skipped = 0
    
    # === CNN Layers (conv2d_1 to conv2d_18) ===
    print("\n=== Converting CNN Layers ===")
    for i in range(1, 19):
        tf_w_name = f'meshnet/cnn/conv2d_{i}/W:0'
        tf_b_name = f'meshnet/cnn/conv2d_{i}/b:0'
        
        if tf_w_name in var_to_shape:
            tf_weight = reader.get_tensor(tf_w_name)
            # TensorFlow: [H, W, C_in, C_out] -> PyTorch: [C_out, C_in, H, W]
            pt_weight = np.transpose(tf_weight, (3, 2, 0, 1))
            pytorch_state_dict[f'cnn.conv{i}.weight'] = pt_weight
            print(f"  conv{i}.weight: {tf_weight.shape} -> {pt_weight.shape}")
            converted += 1
        else:
            print(f"  [SKIP] {tf_w_name} not found")
            skipped += 1
        
        if tf_b_name in var_to_shape:
            tf_bias = reader.get_tensor(tf_b_name)
            pytorch_state_dict[f'cnn.conv{i}.bias'] = tf_bias
            converted += 1
        else:
            print(f"  [SKIP] {tf_b_name} not found")
            skipped += 1
    
    # === DRB1 LocalGConv Layers (localgconv_1 to localgconv_6) ===
    print("\n=== Converting DRB1 (Block 1) Layers ===")
    for i in range(1, 7):
        tf_prefix = f'meshnet/pixel2mesh/graph_drb_blk1_layer_2/localgconv_{i}_vars'
        pt_prefix = f'drb1.local_conv{i}'
        
        # weights_0
        tf_name = f'{tf_prefix}/weights_0:0'
        if tf_name in var_to_shape:
            tf_weight = reader.get_tensor(tf_name)
            pytorch_state_dict[f'{pt_prefix}.weights.0'] = tf_weight
            print(f"  {pt_prefix}.weights.0: {tf_weight.shape}")
            converted += 1
        
        # weights_1
        tf_name = f'{tf_prefix}/weights_1:0'
        if tf_name in var_to_shape:
            tf_weight = reader.get_tensor(tf_name)
            pytorch_state_dict[f'{pt_prefix}.weights.1'] = tf_weight
            print(f"  {pt_prefix}.weights.1: {tf_weight.shape}")
            converted += 1
        
        # bias
        tf_name = f'{tf_prefix}/bias:0'
        if tf_name in var_to_shape:
            tf_bias = reader.get_tensor(tf_name)
            pytorch_state_dict[f'{pt_prefix}.bias'] = tf_bias
            print(f"  {pt_prefix}.bias: {tf_bias.shape}")
            converted += 1
    
    # === DRB2 LocalGConv Layers (localgconv_7 to localgconv_12) ===
    print("\n=== Converting DRB2 (Block 2) Layers ===")
    for i, tf_idx in enumerate(range(7, 13), start=1):
        tf_prefix = f'meshnet/pixel2mesh/graph_drb_blk2_layer_5/localgconv_{tf_idx}_vars'
        pt_prefix = f'drb2.local_conv{i}'
        
        # weights_0
        tf_name = f'{tf_prefix}/weights_0:0'
        if tf_name in var_to_shape:
            tf_weight = reader.get_tensor(tf_name)
            pytorch_state_dict[f'{pt_prefix}.weights.0'] = tf_weight
            print(f"  {pt_prefix}.weights.0: {tf_weight.shape}")
            converted += 1
        
        # weights_1
        tf_name = f'{tf_prefix}/weights_1:0'
        if tf_name in var_to_shape:
            tf_weight = reader.get_tensor(tf_name)
            pytorch_state_dict[f'{pt_prefix}.weights.1'] = tf_weight
            print(f"  {pt_prefix}.weights.1: {tf_weight.shape}")
            converted += 1
        
        # bias
        tf_name = f'{tf_prefix}/bias:0'
        if tf_name in var_to_shape:
            tf_bias = reader.get_tensor(tf_name)
            pytorch_state_dict[f'{pt_prefix}.bias'] = tf_bias
            print(f"  {pt_prefix}.bias: {tf_bias.shape}")
            converted += 1
    
    # === Save PyTorch Checkpoint ===
    print(f"\n=== Summary ===")
    print(f"Converted: {converted} weights")
    print(f"Skipped: {skipped} weights")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save as numpy arrays (can be loaded without TensorFlow)
    np.savez(output_path.replace('.pth', '.npz'), **pytorch_state_dict)
    print(f"\nSaved to: {output_path.replace('.pth', '.npz')}")
    
    # Also save a mapping file for verification
    mapping_path = output_path.replace('.pth', '_mapping.txt')
    with open(mapping_path, 'w') as f:
        f.write("# TensorFlow to PyTorch Weight Mapping\n")
        f.write(f"# Source: {tf_ckpt_path}\n")
        f.write(f"# Total weights: {converted}\n\n")
        for pt_name, weight in pytorch_state_dict.items():
            f.write(f"{pt_name}: {weight.shape}\n")
    print(f"Saved mapping to: {mapping_path}")
    
    return pytorch_state_dict


def verify_checkpoint(npz_path):
    """Verify the converted checkpoint"""
    print(f"\n=== Verifying Checkpoint: {npz_path} ===")
    
    data = np.load(npz_path)
    print(f"Total keys: {len(data.files)}")
    
    # Group by layer type
    cnn_keys = [k for k in data.files if k.startswith('cnn.')]
    drb1_keys = [k for k in data.files if k.startswith('drb1.')]
    drb2_keys = [k for k in data.files if k.startswith('drb2.')]
    
    print(f"  CNN layers: {len(cnn_keys)}")
    print(f"  DRB1 layers: {len(drb1_keys)}")
    print(f"  DRB2 layers: {len(drb2_keys)}")
    
    # Calculate total parameters
    total_params = sum(data[k].size for k in data.files)
    print(f"  Total parameters: {total_params:,}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert TensorFlow checkpoint to PyTorch')
    parser.add_argument('--tf_checkpoint', type=str, required=True,
                        help='Path to TensorFlow checkpoint (without extension)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for PyTorch checkpoint (.pth)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify the converted checkpoint')
    
    args = parser.parse_args()
    
    # Convert
    state_dict = convert_checkpoint(args.tf_checkpoint, args.output)
    
    # Verify if requested
    if args.verify:
        npz_path = args.output.replace('.pth', '.npz')
        verify_checkpoint(npz_path)
    
    print("\n✓ Conversion complete!")
    print("\nTo use in PyTorch:")
    print("  import numpy as np")
    print("  import torch")
    print(f"  data = np.load('{args.output.replace('.pth', '.npz')}')")
    print("  state_dict = {k: torch.from_numpy(data[k]) for k in data.files}")
    print("  model.load_state_dict(state_dict, strict=False)")


if __name__ == '__main__':
    main()
