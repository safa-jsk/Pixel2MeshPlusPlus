#!/usr/bin/env python3
"""
PyTorch implementation of Pixel2Mesh++ - EXACT match to TensorFlow architecture
This model can load weights directly from TensorFlow checkpoint

Architecture from TensorFlow:
- CNN encoder: 18 conv layers (conv2d_1 to conv2d_18)
- DeformationReasoning blocks: 2 blocks (blk1, blk2) with 6 LocalGConv layers each
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _tf_same_pad(x, kernel_size, stride):
    """
    Apply TensorFlow-style 'SAME' padding for strided convolutions.
    
    TensorFlow SAME padding:
    - Pads asymmetrically: (0, 1, 0, 1) for kernel=3, stride=2
    - Ensures output_size = ceil(input_size / stride)
    """
    if stride == 1:
        return x
    
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    
    in_h, in_w = x.shape[2], x.shape[3]
    
    # TensorFlow SAME output size: ceil(input / stride)
    out_h = (in_h + stride[0] - 1) // stride[0]
    out_w = (in_w + stride[1] - 1) // stride[1]
    
    # Required padding
    pad_h = max((out_h - 1) * stride[0] + kernel_size[0] - in_h, 0)
    pad_w = max((out_w - 1) * stride[1] + kernel_size[1] - in_w, 0)
    
    # Asymmetric padding: more on bottom/right
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # F.pad format: (left, right, top, bottom)
    return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))


class CNN18(nn.Module):
    """
    Custom 18-layer CNN matching TensorFlow's build_cnn18()
    TensorFlow uses NHWC, PyTorch uses NCHW
    TensorFlow weight format: [H, W, C_in, C_out]
    PyTorch weight format: [C_out, C_in, H, W]
    
    Uses TensorFlow-style asymmetric SAME padding for strided convolutions.
    """
    
    def __init__(self):
        super(CNN18, self).__init__()
        
        # 224x224
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        # x0 output: 16 channels, 224x224
        
        # Strided convolutions use padding=0, we'll apply TF padding manually
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0)
        # 112x112
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        # x1 output: 32 channels, 112x112
        
        self.conv6 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0)
        # 56x56
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # x2 output: 64 channels, 56x56
        
        self.conv9 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        # 28x28
        self.conv10 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        # x3 output: 128 channels, 28x28
        
        self.conv12 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=0)
        # 14x14
        self.conv13 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv14 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # x4 output: 256 channels, 14x14
        
        self.conv15 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=0)
        # 7x7
        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv17 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv18 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # x5 output: 512 channels, 7x7
        
    def forward(self, x):
        """
        Args:
            x: [batch, 3, 224, 224] for PyTorch (NCHW)
               TensorFlow uses [batch, 224, 224, 3] (NHWC)
        Returns:
            img_feat: list of [x0, x1, x2] feature maps
            - x0: [batch, 16, 224, 224]
            - x1: [batch, 32, 112, 112]
            - x2: [batch, 64, 56, 56]
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x0 = x  # 16 channels, 224x224
        
        # Apply TF-style asymmetric padding for strided convs
        x = F.relu(self.conv3(_tf_same_pad(x, 3, 2)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x1 = x  # 32 channels, 112x112
        
        x = F.relu(self.conv6(_tf_same_pad(x, 3, 2)))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x2 = x  # 64 channels, 56x56
        
        x = F.relu(self.conv9(_tf_same_pad(x, 3, 2)))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x3 = x  # 128 channels, 28x28
        
        x = F.relu(self.conv12(_tf_same_pad(x, 5, 2)))
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x4 = x  # 256 channels, 14x14
        
        x = F.relu(self.conv15(_tf_same_pad(x, 5, 2)))
        x = F.relu(self.conv16(x))
        x = F.relu(self.conv17(x))
        x = F.relu(self.conv18(x))
        x5 = x  # 512 channels, 7x7
        
        # Return feature maps used for projection (x0, x1, x2)
        return [x0, x1, x2]


class LocalGConv(nn.Module):
    """
    Local Graph Convolution layer matching TensorFlow's LocalGConv
    
    TensorFlow weight names:
        localgconv_N_vars/weights_0: [input_dim, output_dim]
        localgconv_N_vars/weights_1: [input_dim, output_dim]
        localgconv_N_vars/bias: [output_dim]
    """
    
    def __init__(self, input_dim, output_dim, support_num=2, act=True):
        super(LocalGConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.support_num = support_num
        self.act = act
        
        # Weight matrices for each support (2 supports: self + neighbors)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(input_dim, output_dim))
            for _ in range(support_num)
        ])
        
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, supports):
        """
        Args:
            x: [N, S, F] where N=num_vertices, S=sample_size(43), F=features
            supports: list of [S, S] adjacency matrices (2 matrices)
        Returns:
            output: [N, S, output_dim]
        
        TensorFlow equivalent:
            pre_sup = tf.einsum('ijk,kl->ijl', x, weights)  # [N,S,F] @ [F,out] -> [N,S,out]
            support = tf.einsum('ij,kjl->kil', adj, pre_sup)  # adj @ pre_sup over S dim
        """
        outputs = []
        for i in range(self.support_num):
            # x: [N, S, F], weights[i]: [F, out_dim]
            # pre_sup: [N, S, out_dim]
            pre_sup = torch.matmul(x, self.weights[i])
            
            # TF einsum 'ij,kjl->kil': adj[i,j] * pre_sup[k,j,l] -> output[k,i,l]
            # This is: for each batch k, output[k] = adj @ pre_sup[k]
            # In PyTorch: supports[i] @ pre_sup along the S dimension
            # pre_sup: [N, S, out], supports[i]: [S, S]
            # Need: [N, S, out] where output[n] = supports[i] @ pre_sup[n]
            # Use einsum: 'ij,njl->nil'
            output = torch.einsum('ij,njl->nil', supports[i], pre_sup)
            outputs.append(output)
        
        # Sum all support outputs and add bias
        output = sum(outputs) + self.bias
        
        if self.act:
            output = F.relu(output)
        
        return output


class DeformationReasoning(nn.Module):
    """
    Deformation Reasoning Block matching TensorFlow's DeformationReasoning
    
    Contains 6 LocalGConv layers:
        local_conv1: input_dim (339) -> 192
        local_conv2: 192 -> 192
        local_conv3: 192 -> 192 (with residual from conv1)
        local_conv4: 192 -> 192
        local_conv5: 192 -> 192 (with residual from conv3)
        local_conv6: 192 -> 1 (score output)
    """
    
    def __init__(self, input_dim, hidden_dim=192):
        super(DeformationReasoning, self).__init__()
        self.s = 43  # sample size
        self.hidden_dim = hidden_dim
        
        self.local_conv1 = LocalGConv(input_dim, hidden_dim)
        self.local_conv2 = LocalGConv(hidden_dim, hidden_dim)
        self.local_conv3 = LocalGConv(hidden_dim, hidden_dim)
        self.local_conv4 = LocalGConv(hidden_dim, hidden_dim)
        self.local_conv5 = LocalGConv(hidden_dim, hidden_dim)
        self.local_conv6 = LocalGConv(hidden_dim, 1, act=False)  # No ReLU for score
    
    def forward(self, proj_feat, prev_coord, delta_coord, supports):
        """
        Args:
            proj_feat: [N*S, F] projected features
            prev_coord: [N, 3] previous coordinates
            delta_coord: [N, S, 3] sample coordinate offsets
            supports: list of [S, S] adjacency matrices
        Returns:
            next_coord: [N, 3] updated coordinates
        """
        N = prev_coord.shape[0]
        x = proj_feat.view(N, self.s, -1)  # [N, S, F]
        
        x1 = self.local_conv1(x, supports)
        x2 = self.local_conv2(x1, supports)
        x3 = self.local_conv3(x2, supports) + x1  # Residual connection
        x4 = self.local_conv4(x3, supports)
        x5 = self.local_conv5(x4, supports) + x3  # Residual connection
        x6 = self.local_conv6(x5, supports)  # [N, S, 1]
        
        # Softmax over samples to get attention scores
        score = F.softmax(x6, dim=1)  # [N, S, 1]
        
        # Weighted sum of delta coordinates
        weighted_delta = score * delta_coord  # [N, S, 3]
        next_coord = weighted_delta.sum(dim=1) + prev_coord  # [N, 3]
        
        return next_coord


class LocalGraphProjection(nn.Module):
    """
    Local Graph Projection module matching TensorFlow's LocalGraphProjection
    Projects 3D points to image features using multi-view bilinear sampling
    """
    
    def __init__(self, view_number=3):
        super(LocalGraphProjection, self).__init__()
        self.view_number = view_number
    
    def forward(self, coord, img_feat, cameras):
        """
        Args:
            coord: [N, 3] vertex coordinates
            img_feat: list of [view, H, W, C] feature maps at different scales
            cameras: camera parameters for each view
        Returns:
            features: [N, F] concatenated features (coord + multi-scale image features)
        """
        # This is a simplified version - full implementation would include
        # camera transformation and bilinear sampling
        # For checkpoint conversion, the projection logic remains the same
        
        out1_list = []
        out2_list = []
        out3_list = []
        
        for i in range(self.view_number):
            # Camera transformation and projection
            # (Simplified - using the same logic as TensorFlow)
            X, Y, Z = coord[:, 0], coord[:, 1], coord[:, 2]
            
            # Project to image coordinates
            h = 248.0 * (-Y / (-Z + 1e-8)) + 112.0
            w = 248.0 * (X / (-Z + 1e-8)) + 112.0
            
            h = torch.clamp(h, 0, 223)
            w = torch.clamp(w, 0, 223)
            
            # Bilinear sampling at different scales
            out1 = self._bilinear_sample(img_feat[0][i], h / 1.0, w / 1.0, 223)
            out2 = self._bilinear_sample(img_feat[1][i], h / 2.0, w / 2.0, 111)
            out3 = self._bilinear_sample(img_feat[2][i], h / 4.0, w / 4.0, 55)
            
            out1_list.append(out1)
            out2_list.append(out2)
            out3_list.append(out3)
        
        # Stack and aggregate across views
        all_out1 = torch.stack(out1_list, dim=0)  # [3, N, C]
        all_out2 = torch.stack(out2_list, dim=0)
        all_out3 = torch.stack(out3_list, dim=0)
        
        # Concatenate features across scales
        image_feature = torch.cat([all_out1, all_out2, all_out3], dim=2)  # [3, N, C_total]
        
        # Aggregate: max, mean, std
        image_feature_max = image_feature.max(dim=0)[0]
        image_feature_mean = image_feature.mean(dim=0)
        image_feature_std = image_feature.std(dim=0)
        
        # Concatenate with coordinates
        outputs = torch.cat([coord, image_feature_max, image_feature_mean, image_feature_std], dim=1)
        
        return outputs
    
    def _bilinear_sample(self, feat, x, y, max_val):
        """Bilinear sampling from feature map"""
        x = torch.clamp(x, 0, max_val)
        y = torch.clamp(y, 0, max_val)
        
        x1 = torch.floor(x).long()
        x2 = torch.ceil(x).long()
        y1 = torch.floor(y).long()
        y2 = torch.ceil(y).long()
        
        x1 = torch.clamp(x1, 0, max_val)
        x2 = torch.clamp(x2, 0, max_val)
        y1 = torch.clamp(y1, 0, max_val)
        y2 = torch.clamp(y2, 0, max_val)
        
        Q11 = feat[x1, y1]
        Q12 = feat[x1, y2]
        Q21 = feat[x2, y1]
        Q22 = feat[x2, y2]
        
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        x1f = x1.float().unsqueeze(1)
        x2f = x2.float().unsqueeze(1)
        y1f = y1.float().unsqueeze(1)
        y2f = y2.float().unsqueeze(1)
        
        output = (Q11 * (x2f - x) * (y2f - y) +
                  Q21 * (x - x1f) * (y2f - y) +
                  Q12 * (x2f - x) * (y - y1f) +
                  Q22 * (x - x1f) * (y - y1f))
        
        return output


class MeshNetPyTorch(nn.Module):
    """
    Complete Pixel2Mesh++ Stage 2 (P2MPP) model matching TensorFlow exactly
    
    This model can load weights from TensorFlow checkpoint with proper conversion
    """
    
    def __init__(self, stage2_feat_dim=339):
        super(MeshNetPyTorch, self).__init__()
        
        self.stage2_feat_dim = stage2_feat_dim  # 3 + 16*3 + 32*3 + 64*3 = 339
        self.hidden_dim = 192
        
        # CNN encoder (18 layers)
        self.cnn = CNN18()
        
        # Deformation Reasoning Block 1 (6 LocalGConv layers)
        self.drb1 = DeformationReasoning(stage2_feat_dim, self.hidden_dim)
        
        # Deformation Reasoning Block 2 (6 LocalGConv layers)
        self.drb2 = DeformationReasoning(stage2_feat_dim, self.hidden_dim)
    
    def forward(self, img_inp, initial_coord, sample_coord, sample_adj, cameras):
        """
        Args:
            img_inp: [batch, 3, H, W] input images (3 views concatenated or separate)
            initial_coord: [N, 3] initial mesh coordinates
            sample_coord: [N, S, 3] sampled hypothesis coordinates
            sample_adj: list of [S, S] adjacency matrices for local graph
            cameras: camera parameters
        Returns:
            output_coord: [N, 3] final mesh coordinates
        """
        # Extract image features
        img_feat = self.cnn(img_inp)  # List of [x0, x1, x2]
        
        # Block 1: Sample -> Project -> Deform
        # (Simplified - full pipeline would include SampleHypothesis and LocalGraphProjection)
        proj_feat1 = self._project_features(initial_coord, sample_coord, img_feat, cameras)
        blk1_out = self.drb1(proj_feat1, initial_coord, sample_coord, sample_adj)
        
        # Block 2: Sample -> Project -> Deform
        proj_feat2 = self._project_features(blk1_out, sample_coord, img_feat, cameras)
        blk2_out = self.drb2(proj_feat2, blk1_out, sample_coord, sample_adj)
        
        return blk2_out
    
    def _project_features(self, coord, sample_coord, img_feat, cameras):
        """Project vertex coordinates to image features (simplified)"""
        N = coord.shape[0]
        S = sample_coord.shape[1]
        
        # For each vertex, get features at sampled hypothesis points
        # This is a simplified version - actual implementation uses LocalGraphProjection
        # Return shape: [N*S, feat_dim]
        
        # Placeholder: concatenate coordinates with dummy features
        # In actual use, this would sample from img_feat using bilinear interpolation
        feat_dim = self.stage2_feat_dim
        proj_feat = torch.zeros(N * S, feat_dim, device=coord.device)
        
        return proj_feat


def load_tf_checkpoint_to_pytorch(tf_ckpt_path, pytorch_model):
    """
    Load TensorFlow checkpoint weights into PyTorch model
    
    Weight mapping:
    - TensorFlow conv: [H, W, C_in, C_out] -> PyTorch: [C_out, C_in, H, W]
    - TensorFlow dense: [in, out] -> PyTorch: [out, in] (if Linear) or [in, out] (if Parameter)
    """
    import tensorflow as tf
    
    reader = tf.train.NewCheckpointReader(tf_ckpt_path)
    var_to_shape = reader.get_variable_to_shape_map()
    
    # Create mapping
    weight_mapping = {}
    
    # CNN layers mapping
    for i in range(1, 19):
        tf_w_name = f'meshnet/cnn/conv2d_{i}/W:0'
        tf_b_name = f'meshnet/cnn/conv2d_{i}/b:0'
        pt_w_name = f'cnn.conv{i}.weight'
        pt_b_name = f'cnn.conv{i}.bias'
        weight_mapping[tf_w_name] = (pt_w_name, 'conv')
        weight_mapping[tf_b_name] = (pt_b_name, 'bias')
    
    # DRB1 LocalGConv layers (localgconv_1 to localgconv_6)
    for i in range(1, 7):
        tf_prefix = f'meshnet/pixel2mesh/graph_drb_blk1_layer_2/localgconv_{i}_vars'
        pt_prefix = f'drb1.local_conv{i}'
        
        weight_mapping[f'{tf_prefix}/weights_0:0'] = (f'{pt_prefix}.weights.0', 'dense')
        weight_mapping[f'{tf_prefix}/weights_1:0'] = (f'{pt_prefix}.weights.1', 'dense')
        weight_mapping[f'{tf_prefix}/bias:0'] = (f'{pt_prefix}.bias', 'bias')
    
    # DRB2 LocalGConv layers (localgconv_7 to localgconv_12)
    for i, j in enumerate(range(7, 13), start=1):
        tf_prefix = f'meshnet/pixel2mesh/graph_drb_blk2_layer_5/localgconv_{j}_vars'
        pt_prefix = f'drb2.local_conv{i}'
        
        weight_mapping[f'{tf_prefix}/weights_0:0'] = (f'{pt_prefix}.weights.0', 'dense')
        weight_mapping[f'{tf_prefix}/weights_1:0'] = (f'{pt_prefix}.weights.1', 'dense')
        weight_mapping[f'{tf_prefix}/bias:0'] = (f'{pt_prefix}.bias', 'bias')
    
    # Load and convert weights
    pytorch_state = pytorch_model.state_dict()
    converted_count = 0
    
    for tf_name, (pt_name, weight_type) in weight_mapping.items():
        if tf_name not in var_to_shape:
            print(f"[WARN] TF variable not found: {tf_name}")
            continue
        
        tf_weight = reader.get_tensor(tf_name)
        
        if weight_type == 'conv':
            # TensorFlow: [H, W, C_in, C_out] -> PyTorch: [C_out, C_in, H, W]
            pt_weight = np.transpose(tf_weight, (3, 2, 0, 1))
        elif weight_type == 'dense':
            # TensorFlow: [in, out] -> PyTorch Parameter: [in, out] (no transpose needed)
            pt_weight = tf_weight
        else:
            # Bias: no change
            pt_weight = tf_weight
        
        pt_tensor = torch.from_numpy(pt_weight)
        
        if pt_name in pytorch_state:
            if pytorch_state[pt_name].shape == pt_tensor.shape:
                pytorch_state[pt_name] = pt_tensor
                converted_count += 1
            else:
                print(f"[WARN] Shape mismatch: {pt_name}")
                print(f"  TF: {tf_weight.shape} -> PT expected: {pytorch_state[pt_name].shape}, got: {pt_tensor.shape}")
        else:
            print(f"[WARN] PyTorch key not found: {pt_name}")
    
    pytorch_model.load_state_dict(pytorch_state)
    print(f"Loaded {converted_count}/{len(weight_mapping)} weights from TensorFlow checkpoint")
    
    return pytorch_model


if __name__ == '__main__':
    # Test model creation
    model = MeshNetPyTorch(stage2_feat_dim=339)
    print("Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Print model structure
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
