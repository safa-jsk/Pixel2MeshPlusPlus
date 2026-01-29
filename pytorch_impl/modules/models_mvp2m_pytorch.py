"""
PyTorch Implementation of MVP2M (Multi-View Pixel2Mesh) - Stage 1
Coarse mesh generation from multi-view images

This is a GPU-accelerated PyTorch version of the TensorFlow MVP2M model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNN18(nn.Module):
    """
    18-layer CNN for multi-view image feature extraction
    Same architecture as Stage 2 but different feature outputs
    
    Uses TensorFlow-style 'SAME' padding for strided convolutions:
    TF pads asymmetrically (0 on left/top, 1 on right/bottom) for stride=2
    while PyTorch's padding=1 adds 1 on all sides.
    """
    def __init__(self):
        super(CNN18, self).__init__()
        
        # 224x224 -> 224x224
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        
        # 224x224 -> 112x112 (stride=2, TF-style asymmetric padding)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        
        # 112x112 -> 56x56 (stride=2, TF-style asymmetric padding)
        self.conv6 = nn.Conv2d(32, 64, 3, stride=2, padding=0)
        self.conv7 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        
        # 56x56 -> 28x28 (stride=2, TF-style asymmetric padding)
        self.conv9 = nn.Conv2d(64, 128, 3, stride=2, padding=0)
        self.conv10 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        
        # 28x28 -> 14x14 (stride=2, 5x5 kernel, TF-style asymmetric padding)
        self.conv12 = nn.Conv2d(128, 256, 5, stride=2, padding=0)
        self.conv13 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv14 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        
        # 14x14 -> 7x7 (stride=2, 5x5 kernel, TF-style asymmetric padding)
        self.conv15 = nn.Conv2d(256, 512, 5, stride=2, padding=0)
        self.conv16 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv17 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv18 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
    
    def _tf_same_pad(self, x, kernel_size):
        """
        Apply TensorFlow-style 'SAME' padding for stride=2 convolution.
        TF pads asymmetrically: 0 on left/top, (kernel_size-1)//2 + 1 on right/bottom
        for odd kernel sizes with stride=2.
        
        For kernel=3: pads (0, 1, 0, 1)
        For kernel=5: pads (1, 2, 1, 2)
        """
        if kernel_size == 3:
            return F.pad(x, (0, 1, 0, 1))  # (left, right, top, bottom)
        elif kernel_size == 5:
            return F.pad(x, (1, 2, 1, 2))
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}")
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) input images
        Returns:
            x2: (B, 64, 56, 56) - for projection
            x3: (B, 128, 28, 28) - for projection
            x4: (B, 256, 14, 14) - for projection
            x5: (B, 512, 7, 7) - for projection
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x0 at 224x224
        
        x = F.relu(self.conv3(self._tf_same_pad(x, 3)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        # x1 at 112x112
        
        x = F.relu(self.conv6(self._tf_same_pad(x, 3)))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x2 = x  # 56x56, 64 channels
        
        x = F.relu(self.conv9(self._tf_same_pad(x, 3)))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x3 = x  # 28x28, 128 channels
        
        x = F.relu(self.conv12(self._tf_same_pad(x, 5)))
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x4 = x  # 14x14, 256 channels
        
        x = F.relu(self.conv15(self._tf_same_pad(x, 5)))
        x = F.relu(self.conv16(x))
        x = F.relu(self.conv17(x))
        x = F.relu(self.conv18(x))
        x5 = x  # 7x7, 512 channels
        
        return [x2, x3, x4, x5]


class GraphConvolution(nn.Module):
    """
    Graph Convolution layer for mesh deformation
    Supports sparse adjacency matrices
    """
    def __init__(self, input_dim, output_dim, num_supports=2, act=True, bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_supports = num_supports
        self.use_act = act
        self.use_bias = bias
        
        # One weight matrix per support
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, output_dim))
            for _ in range(num_supports)
        ])
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.register_parameter('bias', None)
        
        self._init_weights()
    
    def _init_weights(self):
        for w in self.weights:
            nn.init.xavier_uniform_(w)
    
    def forward(self, x, supports):
        """
        Args:
            x: (N, input_dim) node features
            supports: list of (N, N) sparse adjacency matrices
        Returns:
            output: (N, output_dim)
        """
        outputs = []
        for i, support in enumerate(supports):
            # x @ W
            pre_sup = x @ self.weights[i]
            # A @ (x @ W) using sparse matrix multiply
            if support.is_sparse:
                output = torch.sparse.mm(support, pre_sup)
            else:
                output = support @ pre_sup
            outputs.append(output)
        
        output = sum(outputs)
        
        if self.use_bias and self.bias is not None:
            output = output + self.bias
        
        if self.use_act:
            output = F.relu(output)
        
        return output


class GraphPooling(nn.Module):
    """
    Graph pooling layer for mesh upsampling
    Adds new vertices at edge midpoints
    """
    def __init__(self, pool_idx):
        super(GraphPooling, self).__init__()
        # Handle both numpy and tensor inputs
        if isinstance(pool_idx, torch.Tensor):
            self.register_buffer('pool_idx', pool_idx.long())
        else:
            self.register_buffer('pool_idx', torch.from_numpy(pool_idx.astype(np.int64)))
    
    def forward(self, x):
        """
        Args:
            x: (N, F) node features
        Returns:
            output: (N+M, F) upsampled features
        """
        # Average features of neighboring vertices
        neighbors = x[self.pool_idx]  # (M, 2, F)
        add_feat = neighbors.mean(dim=1)  # (M, F)
        output = torch.cat([x, add_feat], dim=0)
        return output


class MVP2MNet(nn.Module):
    """
    Multi-View Pixel2Mesh Network (Stage 1)
    
    Architecture:
    - CNN18 for image feature extraction
    - 3 GCN blocks with graph projection
    - Graph pooling between blocks for mesh upsampling
    """
    def __init__(self, feat_dim=963, hidden_dim=192, coord_dim=3):
        super(MVP2MNet, self).__init__()
        
        self.feat_dim = feat_dim  # 64+128+256+512 = 960 + 3 = 963 per view, *3 for max/mean/std
        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim
        
        # CNN for image features
        self.cnn = CNN18()
        
        # Block 1: 14 GCN layers
        # First layer: feat_dim -> hidden_dim
        self.gcn1_layers = nn.ModuleList()
        self.gcn1_layers.append(GraphConvolution(feat_dim, hidden_dim, num_supports=2))
        for _ in range(12):
            self.gcn1_layers.append(GraphConvolution(hidden_dim, hidden_dim, num_supports=2))
        self.gcn1_layers.append(GraphConvolution(hidden_dim, coord_dim, num_supports=2, act=False))
        
        # Block 2: 14 GCN layers (after pooling)
        self.gcn2_layers = nn.ModuleList()
        self.gcn2_layers.append(GraphConvolution(feat_dim + hidden_dim, hidden_dim, num_supports=2))
        for _ in range(12):
            self.gcn2_layers.append(GraphConvolution(hidden_dim, hidden_dim, num_supports=2))
        self.gcn2_layers.append(GraphConvolution(hidden_dim, coord_dim, num_supports=2, act=False))
        
        # Block 3: 15 GCN layers (after pooling)
        self.gcn3_layers = nn.ModuleList()
        self.gcn3_layers.append(GraphConvolution(feat_dim + hidden_dim, hidden_dim, num_supports=2))
        for _ in range(13):
            self.gcn3_layers.append(GraphConvolution(hidden_dim, hidden_dim, num_supports=2))
        self.gcn3_layers.append(GraphConvolution(hidden_dim, coord_dim, num_supports=2, act=False))
    
    def forward(self, imgs, initial_coords, supports1, supports2, supports3, 
                pool_idx1, pool_idx2, cameras, device='cuda'):
        """
        Forward pass for mesh generation
        
        Args:
            imgs: (3, 3, 224, 224) multi-view images
            initial_coords: (N1, 3) initial ellipsoid coordinates
            supports1: list of 2 sparse (N1, N1) adjacency matrices
            supports2: list of 2 sparse (N2, N2) adjacency matrices
            supports3: list of 2 sparse (N3, N3) adjacency matrices
            pool_idx1: (M1, 2) pooling indices for block 1->2
            pool_idx2: (M2, 2) pooling indices for block 2->3
            cameras: (3, 5) camera parameters
        
        Returns:
            output3: (N3, 3) final mesh coordinates
        """
        # Extract CNN features for all views
        img_feat = self.cnn(imgs)  # [x2, x3, x4, x5]
        
        # Block 1: layers 0-14 (projection + 14 GCN layers)
        # Eltwise indices in original: [3, 5, 7, 9, 11, 13]
        # These are layer indices 3,5,7,9,11,13 in the model
        # Which maps to GCN layer indices 2,4,6,8,10,12 (after first GCN)
        coords = initial_coords
        proj_feat1 = self._project_features(coords, img_feat, cameras, device)
        
        activations = [proj_feat1]  # idx 0: projection output
        
        # Layer 1: GCN (feat_dim -> hidden_dim)
        h = self.gcn1_layers[0](activations[-1], supports1)
        activations.append(h)  # idx 1
        
        # Layers 2-13: GCN (hidden_dim -> hidden_dim) with residuals at even indices
        for i in range(1, 13):
            h = self.gcn1_layers[i](activations[-1], supports1)
            # Residual at GCN layer indices 2,4,6,8,10,12 (original layer 3,5,7,9,11,13)
            if i % 2 == 0:
                h = 0.5 * (h + activations[-2])
            activations.append(h)
        
        # Layer 14: GCN (hidden_dim -> coord_dim) - output layer
        output1 = self.gcn1_layers[13](activations[-1], supports1)
        hidden1 = activations[-1]  # Save last hidden state for concat
        
        # Pool to block 2
        pool1 = GraphPooling(pool_idx1).to(device)
        output1_pooled = pool1(output1)
        hidden1_pooled = pool1(hidden1)
        
        # Block 2: layers 15-30 (projection + pooling + 14 GCN layers)
        # Eltwise indices in original: [19, 21, 23, 25, 27, 29]
        # Concat at layer 15 (projection output + previous hidden)
        proj_feat2 = self._project_features(output1_pooled, img_feat, cameras, device)
        x = torch.cat([proj_feat2, hidden1_pooled], dim=1)  # concat at layer 15
        
        activations = [x]  # idx 0: concat output (layer 16 in original = first GCN)
        
        # Layer 17: GCN (feat_dim + hidden_dim -> hidden_dim)
        h = self.gcn2_layers[0](activations[-1], supports2)
        activations.append(h)  # idx 1
        
        # Layers 18-29: GCN (hidden_dim -> hidden_dim) with residuals at [19,21,23,25,27,29]
        for i in range(1, 13):
            h = self.gcn2_layers[i](activations[-1], supports2)
            # Residual at GCN layer indices 2,4,6,8,10,12
            if i % 2 == 0:
                h = 0.5 * (h + activations[-2])
            activations.append(h)
        
        # Layer 30: GCN (hidden_dim -> coord_dim) - output layer
        output2 = self.gcn2_layers[13](activations[-1], supports2)
        hidden2 = activations[-1]
        
        # Pool to block 3
        pool2 = GraphPooling(pool_idx2).to(device)
        output2_pooled = pool2(output2)
        hidden2_pooled = pool2(hidden2)
        
        # Block 3: layers 31-47 (projection + pooling + 15 GCN layers)
        # Eltwise indices in original: [35, 37, 39, 41, 43, 45]
        # Concat at layer 31
        proj_feat3 = self._project_features(output2_pooled, img_feat, cameras, device)
        x = torch.cat([proj_feat3, hidden2_pooled], dim=1)  # concat at layer 31
        
        activations = [x]  # idx 0: concat output (layer 32 in original = first GCN)
        
        # Layer 33: GCN (feat_dim + hidden_dim -> hidden_dim)
        h = self.gcn3_layers[0](activations[-1], supports3)
        activations.append(h)  # idx 1
        
        # Layers 34-46: GCN (hidden_dim -> hidden_dim) with residuals at [35,37,39,41,43,45]
        for i in range(1, 14):
            h = self.gcn3_layers[i](activations[-1], supports3)
            # Residual at GCN layer indices 2,4,6,8,10,12
            if i % 2 == 0:
                h = 0.5 * (h + activations[-2])
            activations.append(h)
        
        # Layer 47: GCN (hidden_dim -> coord_dim) - output layer
        output3 = self.gcn3_layers[14](activations[-1], supports3)
        
        return {
            'coords1': output1,
            'coords2': output2,
            'coords3': output3
        }
    
    def _project_features(self, coords, img_feat, cameras, device):
        """
        Project 3D coordinates to multi-view image features
        
        Args:
            coords: (N, 3) vertex coordinates
            img_feat: [x2, x3, x4, x5] feature maps
            cameras: (3, 5) camera parameters
        
        Returns:
            proj_feat: (N, feat_dim) projected features
        """
        x2, x3, x4, x5 = img_feat  # (3, C, H, W)
        
        all_features = []
        
        # Handle both tensor and numpy cameras
        if isinstance(cameras, torch.Tensor):
            cameras_np = cameras.cpu().numpy()
        else:
            cameras_np = cameras
        
        # First transform from camera[0] space to world space (inverse transform)
        cam0 = cameras_np[0]
        coords_world = self._camera_trans_inv(coords, cam0, device)
        
        for view_idx in range(3):
            cam = cameras_np[view_idx]
            
            # Transform from world space to this view's camera space
            coords_cam = self._camera_trans(coords_world, cam, device)
            
            # Project to image coordinates
            h, w = self._coords_to_image(coords_cam, device)
            
            # Sample from each feature map using nearest neighbor (same as TF gather_nd)
            # NOTE: TensorFlow has a bug where it divides the view index by the scale factor,
            # causing it to always sample from view 0. We replicate this behavior for compatibility.
            def sample_features(feat, h, w, scale, view_idx_for_tf_compat):
                # Scale coordinates to feature map size (same as TF: indices / (224.0 / feat_size))
                # TF uses: tf.cast(indeces / (224.0 / feat_size), tf.int32)
                # This includes dividing the view index, so view 0,1,2 all become 0 after int cast
                tf_view_idx = int(view_idx_for_tf_compat / scale)  # TF bug: always 0 for scales >= 2
                h_scaled = (h / scale).long().clamp(0, feat.shape[2] - 1)
                w_scaled = (w / scale).long().clamp(0, feat.shape[3] - 1)
                
                # Gather using indices: feat[tf_view_idx, :, h, w] -> (N, C)
                feat_view = feat[tf_view_idx]  # (C, H, W) - always view 0 due to TF bug
                # Use advanced indexing
                sampled = feat_view[:, h_scaled, w_scaled].T  # (N, C)
                return sampled
            
            f2 = sample_features(x2, h, w, 224.0 / 56.0, view_idx)  # 64 channels
            f3 = sample_features(x3, h, w, 224.0 / 28.0, view_idx)  # 128 channels
            f4 = sample_features(x4, h, w, 224.0 / 14.0, view_idx)  # 256 channels
            f5 = sample_features(x5, h, w, 224.0 / 7.0, view_idx)   # 512 channels
            
            view_feat = torch.cat([f2, f3, f4, f5], dim=1)  # (N, 960)
            all_features.append(view_feat)
        
        # Stack and aggregate
        all_features = torch.stack(all_features, dim=0)  # (3, N, 960)
        feat_max = all_features.max(dim=0)[0]
        feat_mean = all_features.mean(dim=0)
        # Use population std (not sample std) + epsilon to match TensorFlow's reduce_std
        feat_var = ((all_features - feat_mean.unsqueeze(0)) ** 2).mean(dim=0)
        feat_std = torch.sqrt(feat_var + 1e-6)
        
        # Concatenate with coordinates
        proj_feat = torch.cat([coords, feat_max, feat_mean, feat_std], dim=1)
        
        return proj_feat
    
    def _get_camera_matrix(self, camera, device):
        """Compute camera rotation matrix and origin"""
        theta = camera[0] * np.pi / 180.0
        elevation = camera[1] * np.pi / 180.0
        distance = camera[3]
        
        # Camera position (origin)
        camy = distance * np.sin(elevation)
        lens = distance * np.cos(elevation)
        camx = lens * np.cos(theta)
        camz = lens * np.sin(theta)
        Z = np.array([camx, camy, camz])
        
        # Camera axes
        x = camy * np.cos(theta + np.pi)
        z = camy * np.sin(theta + np.pi)
        Y = np.array([x, lens, z])
        X = np.cross(Y, Z)
        
        def normalize(v):
            norm = np.linalg.norm(v)
            if norm == 0:
                return v
            return v / norm
        
        c = np.stack([normalize(X), normalize(Y), normalize(Z)])
        c_t = torch.from_numpy(c.astype(np.float32)).to(device)
        o_t = torch.from_numpy(Z.astype(np.float32)).to(device)
        
        return c_t, o_t
    
    def _camera_trans(self, coords, camera, device):
        """Transform world coordinates to camera coordinates"""
        c, o = self._get_camera_matrix(camera, device)
        pt_trans = coords - o
        pt_trans = pt_trans @ c.T
        return pt_trans
    
    def _camera_trans_inv(self, coords, camera, device):
        """Transform camera coordinates to world coordinates (inverse transform)"""
        c, o = self._get_camera_matrix(camera, device)
        # inv_xyz = (xyz @ inv(c.T)) + o = (xyz @ inv(c).T) + o = (xyz @ c) + o
        inv_c = torch.inverse(c.T)
        inv_xyz = coords @ inv_c + o
        return inv_xyz
    
    def _coords_to_image(self, coords_cam, device):
        """Project camera-space coordinates to image coordinates"""
        X = coords_cam[:, 0]
        Y = coords_cam[:, 1]
        Z = coords_cam[:, 2]
        
        # Perspective projection (same as TensorFlow version)
        h = 248.0 * (-Y / -Z) + 112.0
        w = 248.0 * (X / -Z) + 112.0
        
        h = torch.clamp(h, 0, 223)
        w = torch.clamp(w, 0, 223)
        
        return h, w
