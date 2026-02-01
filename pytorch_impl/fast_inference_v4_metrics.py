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


# ============================================================================
# Pure PyTorch Chamfer Distance (GPU-accelerated, no custom CUDA kernel needed)
# ============================================================================

def chamfer_distance_pytorch(pred_pts, gt_pts):
    """
    Compute Chamfer Distance using pure PyTorch.
    
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
    """Maximum speed inference with safe optimizations"""
    
    def __init__(self, stage1_checkpoint, stage2_checkpoint, mesh_data_path, device='cuda'):
        self.device = torch.device(device)
        
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
        
        # Pre-allocate delta_coord
        N = 2466
        self.delta_coord = self.sample_coord.unsqueeze(0).expand(N, -1, -1).contiguous()
        
        # Extended warmup for cuDNN autotuner
        print('Warming up GPU (extended)...')
        self._warmup()
        
        print(f'Stage 1 params: {sum(p.numel() for p in self.stage1_model.parameters()):,}')
        print(f'Stage 2 params: {sum(p.numel() for p in self.stage2_model.parameters()):,}')
        print('Ready for maximum speed inference!')
    
    def _sparse_to_torch(self, idx, stage_data):
        indices, values, dense_shape = stage_data[idx]
        coo = sp.coo_matrix((values.astype(np.float32), (indices[:, 0], indices[:, 1])), shape=dense_shape)
        indices_t = torch.LongTensor(np.vstack([coo.row, coo.col]))
        values_t = torch.FloatTensor(coo.data)
        sparse = torch.sparse_coo_tensor(indices_t, values_t, torch.Size(coo.shape))
        return sparse.to(self.device).coalesce()
    
    def _warmup(self):
        dummy_img = torch.randn(3, 3, 224, 224, device=self.device)
        dummy_cam = np.array([[0, 25, 0, 1.9, 25], [162, 25, 0, 1.9, 25], [198, 25, 0, 1.9, 25]])
        
        with torch.inference_mode():
            for _ in range(15):
                _ = self.stage1_model(
                    dummy_img, self.initial_coord,
                    self.supports1, self.supports2, self.supports3,
                    self.pool_idx1, self.pool_idx2,
                    dummy_cam, self.device
                )
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
        output = self.stage1_model(
            imgs, self.initial_coord,
            self.supports1, self.supports2, self.supports3,
            self.pool_idx1, self.pool_idx2,
            cameras, self.device
        )
        coarse_mesh = output['coords3']
        
        img_feat = self.stage2_model.cnn(imgs)
        
        proj_feat1 = self._project_features(coarse_mesh, img_feat, cameras)
        blk1_coord = self._run_drb(self.stage2_model.drb1, proj_feat1, coarse_mesh, self.delta_coord)
        
        proj_feat2 = self._project_features(blk1_coord, img_feat, cameras)
        blk2_coord = self._run_drb(self.stage2_model.drb2, proj_feat2, blk1_coord, self.delta_coord)
        
        return blk2_coord


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
    
    args = parser.parse_args()
    
    tau = args.tau
    tau_2 = 2 * tau
    
    engine = MaxSpeedInferenceEngine(
        args.stage1_checkpoint, args.stage2_checkpoint, args.mesh_data, device='cuda'
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
            
            # Compute Chamfer Distance
            cd, d1, d2 = chamfer_distance_pytorch(pred_tensor, gt_tensor)
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
