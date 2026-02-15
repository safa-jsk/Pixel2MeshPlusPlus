#!/usr/bin/env python3
"""
Maximum Speed PyTorch Inference for Pixel2Mesh++ (Design B v4) with Metrics

Includes integrated quality metrics computation:
- Chamfer Distance (CD)
- F1@tau
- F1@2tau

Safe optimizations that don't affect accuracy:
1. cuDNN benchmark mode for faster convolutions
2. TF32 tensor cores for faster matmul
3. Contiguous memory layout
4. Pre-allocated buffers
5. Extended warmup
6. torch.inference_mode()
7. AMP autocast (optional, enabled by default)
8. torch.compile (optional, disabled by default)
9. CUDA Chamfer extension (preferred over PyTorch fallback)
"""

import argparse
import numpy as np
import pickle
import os
import sys
import time
import csv
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

# Enable cuDNN autotuner for faster convolutions
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.models_mvp2m_pytorch import MVP2MNet
from modules.models_p2mpp_exact import MeshNetPyTorch


# [DESIGN.B][CAMFM.A3_METRICS] CUDA Chamfer Distance Extension (with safe fallback)
# ============================================================================

# Global flag to track which backend is in use
_CHAMFER_BACKEND = None
_CHAMFER_CUDA_MODULE = None


def _try_load_cuda_chamfer():
    """
    Attempt to load the CUDA Chamfer extension.
    Returns the module if successful, None otherwise.
    
    Note: The .so file must match the Python version. If compiled for Python 3.8
    but running Python 3.10, import will fail. Rebuild with:
        cd external/chamfer && python setup.py build_ext --inplace
    """
    global _CHAMFER_CUDA_MODULE
    if _CHAMFER_CUDA_MODULE is not None:
        return _CHAMFER_CUDA_MODULE
    
    # Try to import pre-built extension
    try:
        # Find the project root (parent of pytorch_impl)
        this_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(this_dir)  # Pixel2MeshPlusPlus root
        chamfer_dir = os.path.join(project_root, 'external', 'chamfer')
        
        # Also try workspace-relative path (for Docker)
        if not os.path.exists(chamfer_dir):
            chamfer_dir = '/workspace/external/chamfer'
        
        if os.path.exists(chamfer_dir):
            # Look for .so file in the chamfer directory
            so_files = [f for f in os.listdir(chamfer_dir) if f.endswith('.so')]
            if so_files:
                sys.path.insert(0, chamfer_dir)
            
            # Also check build directory
            build_dir = os.path.join(chamfer_dir, 'build')
            if os.path.exists(build_dir):
                for subdir in os.listdir(build_dir):
                    if subdir.startswith('lib'):
                        sys.path.insert(0, os.path.join(build_dir, subdir))
        
        import chamfer as chamfer_cuda
        _CHAMFER_CUDA_MODULE = chamfer_cuda
        return chamfer_cuda
    except ImportError as e:
        # Common case: .so compiled for different Python version
        return None
    except Exception as e:
        return None


def chamfer_distance_cuda(pred_pts, gt_pts):
    """
    Compute Chamfer Distance using CUDA extension.
    
    Args:
        pred_pts: (N, 3) predicted point cloud (CUDA tensor)
        gt_pts: (M, 3) ground truth point cloud (CUDA tensor)
    
    Returns:
        cd: scalar Chamfer distance
        d1: (N,) squared distances from pred to gt
        d2: (M,) squared distances from gt to pred
    """
    chamfer_cuda = _try_load_cuda_chamfer()
    if chamfer_cuda is None:
        raise RuntimeError("CUDA Chamfer extension not available")
    
    # The CUDA kernel expects (B, N, 3) format
    pred_batch = pred_pts.unsqueeze(0).contiguous()  # (1, N, 3)
    gt_batch = gt_pts.unsqueeze(0).contiguous()      # (1, M, 3)
    
    n = pred_pts.shape[0]
    m = gt_pts.shape[0]
    
    # Pre-allocate output tensors
    dist1 = torch.zeros(1, n, device=pred_pts.device, dtype=torch.float32)
    dist2 = torch.zeros(1, m, device=pred_pts.device, dtype=torch.float32)
    idx1 = torch.zeros(1, n, device=pred_pts.device, dtype=torch.int32)
    idx2 = torch.zeros(1, m, device=pred_pts.device, dtype=torch.int32)
    
    # Call CUDA kernel
    chamfer_cuda.forward(pred_batch, gt_batch, dist1, dist2, idx1, idx2)
    
    # Remove batch dimension
    d1 = dist1.squeeze(0)  # (N,)
    d2 = dist2.squeeze(0)  # (M,)
    
    # Chamfer distance is mean of both directions
    cd = d1.mean() + d2.mean()
    
    return cd, d1, d2


def chamfer_distance_pytorch_fallback(pred_pts, gt_pts):
    """
    Compute Chamfer Distance using pure PyTorch (fallback).
    
    Args:
        pred_pts: (N, 3) predicted point cloud
        gt_pts: (M, 3) ground truth point cloud
    
    Returns:
        cd: scalar Chamfer distance
        d1: (N,) distances from pred to gt
        d2: (M,) distances from gt to pred
    """
    # pred_pts: (N, 3), gt_pts: (M, 3)
    # Compute pairwise squared distances
    # d[i,j] = ||pred[i] - gt[j]||^2
    
    pred_sq = (pred_pts ** 2).sum(dim=1, keepdim=True)  # (N, 1)
    gt_sq = (gt_pts ** 2).sum(dim=1, keepdim=True)       # (M, 1)
    
    # (N, 3) @ (3, M) = (N, M)
    cross = torch.mm(pred_pts, gt_pts.t())
    
    # dist[i,j] = pred_sq[i] + gt_sq[j] - 2*cross[i,j]
    dist_sq = pred_sq + gt_sq.t() - 2 * cross  # (N, M)
    
    # Clamp to avoid numerical issues
    dist_sq = torch.clamp(dist_sq, min=0.0)
    
    # d1: min distance from each pred point to gt
    d1, _ = dist_sq.min(dim=1)  # (N,)
    
    # d2: min distance from each gt point to pred
    d2, _ = dist_sq.min(dim=0)  # (M,)
    
    # Chamfer distance is mean of both directions
    cd = d1.mean() + d2.mean()
    
    return cd, d1, d2


def chamfer_distance_auto(pred_pts, gt_pts, force_cuda=True):
    """
    Compute Chamfer Distance with automatic backend selection.
    
    Args:
        pred_pts: (N, 3) predicted point cloud
        gt_pts: (M, 3) ground truth point cloud
        force_cuda: If True and CUDA extension unavailable, raise error
    
    Returns:
        cd: scalar Chamfer distance
        d1: (N,) distances from pred to gt
        d2: (M,) distances from gt to pred
    """
    global _CHAMFER_BACKEND
    
    # Check if CUDA extension is available
    cuda_available = _try_load_cuda_chamfer() is not None
    tensors_on_cuda = pred_pts.is_cuda and gt_pts.is_cuda
    
    if cuda_available and tensors_on_cuda:
        if _CHAMFER_BACKEND != 'CUDA':
            _CHAMFER_BACKEND = 'CUDA'
        return chamfer_distance_cuda(pred_pts.float(), gt_pts.float())
    else:
        if force_cuda and tensors_on_cuda:
            raise RuntimeError(
                "CUDA Chamfer extension not available but --force-chamfer-cuda is True.\n"
                "To build the extension, run:\n"
                "  cd external/chamfer && python setup.py build_ext --inplace\n"
                "Or set --no-force-chamfer-cuda to use PyTorch fallback."
            )
        if _CHAMFER_BACKEND != 'PyTorch':
            _CHAMFER_BACKEND = 'PyTorch'
        return chamfer_distance_pytorch_fallback(pred_pts.float(), gt_pts.float())


def get_chamfer_backend():
    """Return the currently active Chamfer backend name."""
    global _CHAMFER_BACKEND
    return _CHAMFER_BACKEND or 'uninitialized'


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
    precision = 100.0 * ((d1 <= threshold).sum().item() / len(d1))
    recall = 100.0 * ((d2 <= threshold).sum().item() / len(d2))
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)
    return f1, precision, recall


# ============================================================================
# Inference Engine
# ============================================================================

class MaxSpeedInferenceEngine:
    """Maximum speed inference with safe optimizations
    
    Supports:
      - AMP autocast (FP16 mixed precision) via use_amp flag
      - torch.compile optimization (optional) via use_compile flag
      - CUDA Chamfer extension (external/chamfer/)
      - Configurable warmup iterations
    """
    
    def __init__(self, stage1_checkpoint, stage2_checkpoint, mesh_data_path, device='cuda',
                 use_amp=True, use_compile=False, compile_mode='reduce-overhead', warmup_iters=15):
        self.device = torch.device(device)
        self.use_amp = use_amp
        self.use_compile = use_compile
        self.compile_mode = compile_mode
        self.warmup_iters = warmup_iters
        
        print('Loading mesh data...')
        with open(mesh_data_path, 'rb') as f:
            pkl = pickle.load(f)
        
        # Pre-load all data to GPU with optimal memory layout
        self.initial_coord = torch.from_numpy(pkl['coord'].astype(np.float32)).to(self.device).contiguous()
        self.supports1 = [self._sparse_to_torch(t, pkl['stage1']) for t in range(2)]
        self.supports2 = [self._sparse_to_torch(t, pkl['stage2']) for t in range(2)]
        self.supports3 = [self._sparse_to_torch(t, pkl['stage3']) for t in range(2)]
        self.pool_idx1 = torch.from_numpy(pkl['pool_idx'][0].astype(np.int64)).to(self.device)
        self.pool_idx2 = torch.from_numpy(pkl['pool_idx'][1].astype(np.int64)).to(self.device)
        
        self.sample_coord = torch.from_numpy(pkl['sample_coord'].astype(np.float32)).to(self.device).contiguous()
        self.sample_adj = [torch.from_numpy(np.array(adj).astype(np.float32)).to(self.device).contiguous()
                          for adj in pkl['sample_cheb_dense']]
        self.faces = pkl['faces_triangle'][0]
        self.S = 43
        
        # Load Stage 1 model
        print('Loading Stage 1 model...')
        self.stage1_model = MVP2MNet(feat_dim=2883, hidden_dim=192, coord_dim=3)
        stage1_data = np.load(stage1_checkpoint)
        stage1_state = {}
        model_dict = self.stage1_model.state_dict()
        for key in stage1_data.files:
            arr = stage1_data[key].copy()
            if key in model_dict and model_dict[key].shape == arr.shape:
                stage1_state[key] = torch.from_numpy(arr)
        model_dict.update(stage1_state)
        self.stage1_model.load_state_dict(model_dict, strict=False)
        self.stage1_model = self.stage1_model.to(self.device)
        self.stage1_model.eval()
        
        # Load Stage 2 model
        print('Loading Stage 2 model...')
        self.stage2_model = MeshNetPyTorch(stage2_feat_dim=339)
        stage2_data = np.load(stage2_checkpoint)
        stage2_state = {k: torch.from_numpy(stage2_data[k].copy()) for k in stage2_data.files}
        self.stage2_model.load_state_dict(stage2_state)
        self.stage2_model = self.stage2_model.to(self.device)
        self.stage2_model.eval()
        
        # Apply torch.compile if requested (PyTorch 2.0+ feature)
        # Note: Only apply to Stage 2 - Stage 1 uses sparse tensors which aren't supported
        if self.use_compile:
            if hasattr(torch, 'compile'):
                print(f'Applying torch.compile to Stage 2 only (mode={self.compile_mode})...')
                print('  (Stage 1 uses sparse tensors, incompatible with torch.compile)')
                self.stage2_model = torch.compile(self.stage2_model, mode=self.compile_mode)
            else:
                print('Warning: torch.compile not available (requires PyTorch 2.0+)')
                self.use_compile = False
        
        # Pre-allocate delta_coord
        N = 2466
        self.delta_coord = self.sample_coord.unsqueeze(0).expand(N, -1, -1).contiguous()
        
        # Extended warmup for cuDNN autotuner
        print(f'Warming up GPU ({self.warmup_iters} iterations)...')
        self._warmup()
        
        print(f'Stage 1 params: {sum(p.numel() for p in self.stage1_model.parameters()):,}')
        print(f'Stage 2 params: {sum(p.numel() for p in self.stage2_model.parameters()):,}')
        
        # Print final status
        amp_status = "AMP autocast enabled" if self.use_amp else "AMP disabled"
        compile_status = f"torch.compile ({self.compile_mode})" if self.use_compile else "no compile"
        print(f'Configuration: {amp_status}, {compile_status}')
        print('Ready for maximum speed inference!')
    
    def _sparse_to_torch(self, idx, stage_data):
        indices, values, dense_shape = stage_data[idx]
        coo = sp.coo_matrix((values.astype(np.float32), (indices[:, 0], indices[:, 1])), shape=dense_shape)
        indices_t = torch.LongTensor(np.vstack([coo.row, coo.col]))
        values_t = torch.FloatTensor(coo.data)
        sparse = torch.sparse_coo_tensor(indices_t, values_t, torch.Size(coo.shape))
        return sparse.to(self.device).coalesce()
    
    def _warmup(self):
        """Warmup GPU for stable timing measurements.
        
        Uses self.warmup_iters iterations and respects AMP setting.
        Extra iterations recommended when torch.compile is enabled.
        
        Note: Stage 1 uses sparse ops (no AMP), Stage 2 uses AMP.
        """
        dummy_img = torch.randn(3, 3, 224, 224, device=self.device)
        dummy_cam = np.array([[0, 25, 0, 1.9, 25], [162, 25, 0, 1.9, 25], [198, 25, 0, 1.9, 25]])
        
        # torch.compile requires more warmup for graph compilation
        iters = self.warmup_iters * 2 if self.use_compile else self.warmup_iters
        
        with torch.inference_mode():
            for _ in range(iters):
                # Stage 1: No AMP (sparse ops don't support FP16)
                _ = self.stage1_model(
                    dummy_img, self.initial_coord,
                    self.supports1, self.supports2, self.supports3,
                    self.pool_idx1, self.pool_idx2,
                    dummy_cam, self.device
                )
                # Stage 2: Use AMP if enabled
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    _ = self.stage2_model.cnn(dummy_img)
                    x = torch.randn(2466, 43, 339, device=self.device)
                    _ = self.stage2_model.drb1.local_conv1(x, self.sample_adj)
        torch.cuda.synchronize()
    
    def _project_features(self, coord, img_feat, cameras):
        N = coord.shape[0]
        S = self.S
        device = self.device
        
        sample_points = coord.unsqueeze(1) + self.sample_coord.unsqueeze(0)
        sample_points_flat = sample_points.reshape(N * S, 3)
        
        x0, x1, x2 = img_feat
        all_features = []
        
        for view_idx in range(3):
            cam = cameras[view_idx]
            theta = float(cam[0]) * np.pi / 180.0
            elevation = float(cam[1]) * np.pi / 180.0
            distance = float(cam[3])
            
            camy = distance * np.sin(elevation)
            lens = distance * np.cos(elevation)
            camx = lens * np.cos(theta)
            camz = lens * np.sin(theta)
            
            Z = torch.tensor([camx, camy, camz], device=device, dtype=torch.float32)
            x_val = camy * np.cos(theta + np.pi)
            z_val = camy * np.sin(theta + np.pi)
            Y = torch.tensor([x_val, lens, z_val], device=device, dtype=torch.float32)
            X = torch.cross(Y, Z)
            
            X = X / (torch.norm(X) + 1e-8)
            Y = Y / (torch.norm(Y) + 1e-8)
            Z = Z / (torch.norm(Z) + 1e-8)
            
            c_mat = torch.stack([X, Y, Z])
            cam_points = (sample_points_flat - Z) @ c_mat.T
            X_p, Y_p, Z_p = cam_points[:, 0], cam_points[:, 1], cam_points[:, 2]
            Z_p = torch.clamp(Z_p, max=-0.1)
            
            h = 248.0 * (-Y_p / -Z_p) + 112.0
            w = 248.0 * (X_p / -Z_p) + 112.0
            
            def sample_feat(feat, h, w, size):
                h_norm = 2.0 * h / (size - 1) - 1.0
                w_norm = 2.0 * w / (size - 1) - 1.0
                grid = torch.stack([w_norm, h_norm], dim=-1).unsqueeze(0).unsqueeze(2)
                out = F.grid_sample(feat.unsqueeze(0), grid, mode='bilinear', 
                                   padding_mode='border', align_corners=True)
                return out.squeeze(0).squeeze(-1).T
            
            feat1 = sample_feat(x0[view_idx], h, w, 224)
            feat2 = sample_feat(x1[view_idx], h/2, w/2, 112)
            feat3 = sample_feat(x2[view_idx], h/4, w/4, 56)
            
            view_feat = torch.cat([feat1, feat2, feat3], dim=1)
            all_features.append(view_feat)
        
        all_features = torch.stack(all_features, dim=0)
        feat_max = all_features.max(dim=0)[0]
        feat_mean = all_features.mean(dim=0)
        feat_var = all_features.var(dim=0, unbiased=False)
        feat_std = torch.sqrt(feat_var + 1e-6)
        
        proj_feat = torch.cat([sample_points_flat, feat_max, feat_mean, feat_std], dim=1)
        return proj_feat
    
    def _run_drb(self, drb, proj_feat, prev_coord, delta_coord):
        N = prev_coord.shape[0]
        S = self.S
        
        x = proj_feat.view(N, S, -1)
        x1 = drb.local_conv1(x, self.sample_adj)
        x2 = drb.local_conv2(x1, self.sample_adj)
        x3 = drb.local_conv3(x2, self.sample_adj) + x1
        x4 = drb.local_conv4(x3, self.sample_adj)
        x5 = drb.local_conv5(x4, self.sample_adj) + x3
        x6 = drb.local_conv6(x5, self.sample_adj)
        
        score = F.softmax(x6, dim=1)
        weighted_delta = score * delta_coord
        next_coord = weighted_delta.sum(dim=1) + prev_coord
        
        return next_coord
    
    @torch.inference_mode()
    def infer(self, imgs, cameras):
        """Run inference with optional AMP autocast.
        
        AMP (Automatic Mixed Precision) uses FP16 for compute while
        preserving FP32 for numerically sensitive operations.
        
        Note: Stage 1 uses sparse matrix operations which don't support FP16,
        so AMP is only applied to Stage 2 (the CNN and DRB refinement).
        """
        # Stage 1: Sparse GCN operations - must stay in FP32
        output = self.stage1_model(
            imgs, self.initial_coord,
            self.supports1, self.supports2, self.supports3,
            self.pool_idx1, self.pool_idx2,
            cameras, self.device
        )
        coarse_mesh = output['coords3']
        
        # Stage 2: CNN + DRB - can use AMP
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            img_feat = self.stage2_model.cnn(imgs)
            
            proj_feat1 = self._project_features(coarse_mesh, img_feat, cameras)
            blk1_coord = self._run_drb(self.stage2_model.drb1, proj_feat1, coarse_mesh, self.delta_coord)
            
            proj_feat2 = self._project_features(blk1_coord, img_feat, cameras)
            blk2_coord = self._run_drb(self.stage2_model.drb2, proj_feat2, blk1_coord, self.delta_coord)
        
        # Return in FP32 for metric computation (always cast back)
        return blk2_coord.float()


def load_sample(image_root, sample_id):
    ids = sample_id.split('_')
    category, item_id = ids[0], ids[1]
    
    img_path = os.path.join(image_root, category, item_id, 'rendering')
    if not os.path.exists(img_path):
        img_path = os.path.join(image_root, category, item_id)
    
    camera_meta = np.loadtxt(os.path.join(img_path, 'rendering_metadata.txt'))
    
    imgs = np.zeros((3, 224, 224, 3), dtype=np.float32)
    poses = np.zeros((3, 5), dtype=np.float32)
    
    for idx, view in enumerate([0, 6, 7]):
        img = Image.open(os.path.join(img_path, f'{view:02d}.png'))
        if img.mode == 'RGBA':
            bg = Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32) / 255.0
        imgs[idx] = img_np[:, :, ::-1]
        poses[idx] = camera_meta[view]
    
    return imgs, poses


def load_ground_truth(gt_root, sample_id):
    """
    Load ground truth point cloud from the p2mppdata.
    
    Args:
        gt_root: Path to p2mppdata/test directory
        sample_id: Sample identifier (e.g., '02691156_xxx_00.dat')
    
    Returns:
        gt_pts: (N, 3) ground truth point cloud
    """
    gt_path = os.path.join(gt_root, sample_id)
    if not os.path.exists(gt_path):
        return None
    
    # Load Python 2 pickle with latin1 encoding
    with open(gt_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    # data is a tuple: (image, points)
    # points shape: (1155, 6) - first 3 columns are XYZ
    gt_pts = data[1][:, :3].astype(np.float32)
    
    return gt_pts


def save_mesh_obj(vertices, faces, filepath):
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for face in faces:
            if len(face) == 3:
                f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')


# Category names for ShapeNet
CATEGORY_NAMES = {
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1_checkpoint', default='pytorch_impl/checkpoints/mvp2m_converted.npz')
    parser.add_argument('--stage2_checkpoint', default='pytorch_impl/checkpoints/meshnet_converted.npz')
    parser.add_argument('--mesh_data', default='data/iccv_p2mpp.dat')
    parser.add_argument('--test_file', default='data/designB_eval_full.txt')
    parser.add_argument('--image_root', default='data/designA_subset/ShapeNetRendering/rendering_only')
    parser.add_argument('--gt_root', default='data/designA_subset/p2mppdata/test')
    parser.add_argument('--output_dir', default='outputs/designB/eval_meshes_v4')
    parser.add_argument('--tau', type=float, default=0.0001, help='Threshold for F1-score')
    
    # Acceleration options (matching documentation)
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Enable AMP autocast for FP16 inference (default: True)')
    parser.add_argument('--no-amp', action='store_false', dest='amp',
                        help='Disable AMP autocast')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Enable torch.compile (default: False)')
    parser.add_argument('--compile-mode', type=str, default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode (default: reduce-overhead)')
    parser.add_argument('--force-chamfer-cuda', action='store_true', default=False,
                        help='Require CUDA Chamfer extension (default: False, uses best available)')
    parser.add_argument('--no-force-chamfer-cuda', action='store_false', dest='force_chamfer_cuda',
                        help='Allow PyTorch fallback for Chamfer (default)')
    parser.add_argument('--warmup-iters', type=int, default=15,
                        help='Number of warmup iterations (default: 15)')
    
    args = parser.parse_args()
    
    tau = args.tau
    tau_2 = 2 * tau
    
    # Print feature banner (matching documentation claims)
    print(f'\n{"="*70}')
    print('DESIGN B v4 - ACCELERATION FEATURE STATUS')
    print(f'{"="*70}')
    print(f'  AMP Autocast:        {"ENABLED" if args.amp else "DISABLED"}')
    print(f'  torch.compile:       {"ENABLED (mode={})".format(args.compile_mode) if args.compile else "DISABLED"}')
    print(f'  Force CUDA Chamfer:  {"YES" if args.force_chamfer_cuda else "NO (allow PyTorch fallback)"}')
    print(f'  Warmup iterations:   {args.warmup_iters}')
    print(f'{"="*70}')
    
    engine = MaxSpeedInferenceEngine(
        args.stage1_checkpoint, args.stage2_checkpoint, args.mesh_data, device='cuda',
        use_amp=args.amp, use_compile=args.compile, compile_mode=args.compile_mode,
        warmup_iters=args.warmup_iters
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    benchmark_dir = args.output_dir.replace('eval_meshes', 'benchmark')
    os.makedirs(benchmark_dir, exist_ok=True)
    
    with open(args.test_file, 'r') as f:
        test_list = [line.strip() for line in f if line.strip()]
    
    print(f'\n{"="*70}')
    print('Design B v4 - Inference with Quality Metrics')
    print(f'{"="*70}')
    print(f'Test file: {args.test_file}')
    print(f'Ground truth: {args.gt_root}')
    print(f'Threshold tau: {tau}')
    print(f'{"="*70}')
    print(f'\nProcessing {len(test_list)} samples...')
    print('-' * 90)
    print(f'{"Sample":<45} {"Time(ms)":>10} {"CD(×10⁻³)":>12} {"F1@τ(%)":>10} {"F1@2τ(%)":>10}')
    print('-' * 90)
    
    device = torch.device('cuda')
    times = []
    all_results = []
    category_metrics = {}
    
    for idx, sample_id in enumerate(test_list):
        # Load input images
        imgs, poses = load_sample(args.image_root, sample_id)
        imgs_tensor = torch.from_numpy(imgs.transpose(0, 3, 1, 2)).to(device)
        
        # Run inference with timing
        torch.cuda.synchronize()
        start = time.time()
        mesh_gpu = engine.infer(imgs_tensor, poses)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        
        mesh = mesh_gpu.cpu().numpy()
        sample_base = sample_id.replace('.dat', '')
        
        # Save mesh outputs
        save_mesh_obj(mesh, engine.faces, os.path.join(args.output_dir, f'{sample_base}_predict.obj'))
        np.savetxt(os.path.join(args.output_dir, f'{sample_base}_predict.xyz'), mesh)
        
        # Load ground truth and compute metrics
        gt_pts = load_ground_truth(args.gt_root, sample_id)
        
        if gt_pts is not None:
            # Convert to tensors
            pred_tensor = torch.from_numpy(mesh).to(device)
            gt_tensor = torch.from_numpy(gt_pts).to(device)
            
            # Save ground truth
            np.savetxt(os.path.join(args.output_dir, f'{sample_base}_ground.xyz'), gt_pts)
            
            # Compute Chamfer Distance using auto-selected backend (CUDA preferred)
            cd, d1, d2 = chamfer_distance_auto(pred_tensor, gt_tensor, force_cuda=args.force_chamfer_cuda)
            cd_val = cd.item()
            
            # Compute F1 scores
            f1_tau, prec_tau, rec_tau = compute_f1_score(d1, d2, tau)
            f1_2tau, prec_2tau, rec_2tau = compute_f1_score(d1, d2, tau_2)
            
            # Store result
            category_id = sample_base.split('_')[0]
            result = {
                'sample_id': sample_id,
                'inference_time_ms': elapsed * 1000,
                'chamfer_distance': cd_val,
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
                category_metrics[category_id] = {'cd': [], 'f1_tau': [], 'f1_2tau': [], 'time': []}
            category_metrics[category_id]['cd'].append(cd_val)
            category_metrics[category_id]['f1_tau'].append(f1_tau)
            category_metrics[category_id]['f1_2tau'].append(f1_2tau)
            category_metrics[category_id]['time'].append(elapsed * 1000)
            
            # Print progress
            print(f'{sample_base[:45]:<45} {elapsed*1000:>10.1f} {cd_val*1000:>12.4f} {f1_tau:>10.2f} {f1_2tau:>10.2f}')
        else:
            print(f'{sample_base[:45]:<45} {elapsed*1000:>10.1f} {"N/A":>12} {"N/A":>10} {"N/A":>10}')
    
    print('-' * 90)
    
    # =========================================================================
    # Summary Statistics
    # =========================================================================
    
    total_time_ms = sum(times) * 1000
    total_time_s = sum(times)
    
    print('')
    print(f'{"="*70}')
    print('DESIGN B v4 - PERFORMANCE SUMMARY')
    print(f'{"="*70}')
    print(f'Total samples: {len(times)}')
    print(f'Mean inference time: {np.mean(times)*1000:.1f}ms/sample')
    print(f'Std: {np.std(times)*1000:.1f}ms')
    print(f'Min: {np.min(times)*1000:.1f}ms')
    print(f'Max: {np.max(times)*1000:.1f}ms')
    print(f'Total time: {total_time_ms:.1f}ms ({total_time_s:.2f}s)')
    print(f'Throughput: {len(times)/sum(times):.1f} samples/sec')
    print(f'{"="*70}')
    print(f'\nAcceleration Features Used:')
    print(f'  AMP Autocast:    {"ENABLED" if args.amp else "DISABLED"}')
    print(f'  torch.compile:   {"ENABLED" if args.compile else "DISABLED"}')
    print(f'  Chamfer Backend: {get_chamfer_backend()}')
    
    if all_results:
        avg_cd = np.mean([r['chamfer_distance'] for r in all_results])
        std_cd = np.std([r['chamfer_distance'] for r in all_results])
        avg_f1_tau = np.mean([r['f1_tau'] for r in all_results])
        avg_f1_2tau = np.mean([r['f1_2tau'] for r in all_results])
        
        print('')
        print(f'{"="*70}')
        print('QUALITY METRICS SUMMARY')
        print(f'{"="*70}')
        print(f'  Samples evaluated: {len(all_results)}')
        print(f'  Threshold tau: {tau}')
        print('')
        print(f'  Chamfer Distance: {avg_cd:.8f} ± {std_cd:.8f}  (×10⁻³: {avg_cd*1000:.4f})')
        print(f'  F1@tau:           {avg_f1_tau:.2f}%')
        print(f'  F1@2tau:          {avg_f1_2tau:.2f}%')
        print('')
        
        # Per-category breakdown
        print('PER-CATEGORY METRICS:')
        print('-' * 70)
        print(f'{"Category":<10} {"Name":<10} {"Count":>8} {"CD(×10⁻³)":>12} {"F1@τ(%)":>10} {"F1@2τ(%)":>10}')
        print('-' * 70)
        
        for cat_id in sorted(category_metrics.keys()):
            cat_data = category_metrics[cat_id]
            cat_name = CATEGORY_NAMES.get(cat_id, 'unknown')
            print(f'{cat_id:<10} {cat_name:<10} {len(cat_data["cd"]):>8} '
                  f'{np.mean(cat_data["cd"])*1000:>12.4f} '
                  f'{np.mean(cat_data["f1_tau"]):>10.2f} '
                  f'{np.mean(cat_data["f1_2tau"]):>10.2f}')
        print('-' * 70)
        
        # =====================================================================
        # Save detailed CSV
        # =====================================================================
        metrics_csv = os.path.join(benchmark_dir, 'metrics_results.csv')
        with open(metrics_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id', 'inference_time_ms', 'chamfer_distance', 
                           'f1_tau', 'f1_2tau', 'precision_tau', 'recall_tau', 
                           'precision_2tau', 'recall_2tau'])
            for r in all_results:
                writer.writerow([
                    r['sample_id'],
                    f'{r["inference_time_ms"]:.2f}',
                    f'{r["chamfer_distance"]:.8f}',
                    f'{r["f1_tau"]:.4f}',
                    f'{r["f1_2tau"]:.4f}',
                    f'{r["precision_tau"]:.4f}',
                    f'{r["recall_tau"]:.4f}',
                    f'{r["precision_2tau"]:.4f}',
                    f'{r["recall_2tau"]:.4f}',
                ])
        print(f'\n✓ Saved detailed metrics to: {metrics_csv}')
        
        # =====================================================================
        # Save summary
        # =====================================================================
        summary_file = os.path.join(benchmark_dir, 'metrics_summary.txt')
        with open(summary_file, 'w') as f:
            f.write('Design B v4 Quality Metrics Summary\n')
            f.write('=' * 60 + '\n')
            f.write(f'Date: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'Samples: {len(all_results)}\n')
            f.write(f'Threshold tau: {tau}\n')
            f.write('\n')
            f.write('Performance:\n')
            f.write('-' * 60 + '\n')
            f.write(f'  Mean inference time: {np.mean(times)*1000:.1f}ms/sample\n')
            f.write(f'  Std: {np.std(times)*1000:.1f}ms\n')
            f.write(f'  Total time: {total_time_ms:.1f}ms ({total_time_s:.2f}s)\n')
            f.write(f'  Throughput: {len(times)/sum(times):.1f} samples/sec\n')
            f.write('\n')
            f.write('Overall Metrics:\n')
            f.write('-' * 60 + '\n')
            f.write(f'  Chamfer Distance: {avg_cd:.8f} ± {std_cd:.8f}\n')
            f.write(f'  F1@tau:           {avg_f1_tau:.2f}%\n')
            f.write(f'  F1@2tau:          {avg_f1_2tau:.2f}%\n')
            f.write('\n')
            f.write('Per-Category Metrics:\n')
            f.write('-' * 60 + '\n')
            for cat_id in sorted(category_metrics.keys()):
                cat_data = category_metrics[cat_id]
                cat_name = CATEGORY_NAMES.get(cat_id, 'unknown')
                f.write(f'  {cat_id} ({cat_name}): CD={np.mean(cat_data["cd"]):.6f}  '
                       f'F1@tau={np.mean(cat_data["f1_tau"]):.2f}%  '
                       f'F1@2tau={np.mean(cat_data["f1_2tau"]):.2f}%  '
                       f'Time={np.mean(cat_data["time"]):.1f}ms  '
                       f'(n={len(cat_data["cd"])})\n')
            f.write('=' * 60 + '\n')
        print(f'✓ Saved summary to: {summary_file}')
        
        # =====================================================================
        # Save combined timing stats
        # =====================================================================
        timing_file = os.path.join(benchmark_dir, 'combined_timings.txt')
        with open(timing_file, 'w') as f:
            f.write('Design B v4 Timing Statistics\n')
            f.write('=' * 40 + '\n')
            f.write(f'Samples: {len(times)}\n')
            f.write(f'Mean: {np.mean(times)*1000:.1f}ms\n')
            f.write(f'Std:  {np.std(times)*1000:.1f}ms\n')
            f.write(f'Min:  {np.min(times)*1000:.1f}ms\n')
            f.write(f'Max:  {np.max(times)*1000:.1f}ms\n')
            f.write(f'Total: {total_time_ms:.1f}ms ({total_time_s:.2f}s)\n')
            f.write(f'Throughput: {len(times)/sum(times):.1f} samples/sec\n')
        print(f'✓ Saved timing stats to: {timing_file}')
    
    print('')
    print(f'{"="*70}')
    print('DESIGN B v4 - EVALUATION COMPLETE!')
    print(f'{"="*70}')
    print(f'📁 Output files:')
    print(f'   Meshes:  {args.output_dir}/')
    print(f'   Metrics: {benchmark_dir}/metrics_results.csv')
    print(f'   Summary: {benchmark_dir}/metrics_summary.txt')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
