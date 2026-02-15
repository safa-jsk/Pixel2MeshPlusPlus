#!/usr/bin/env python3
"""
Maximum Speed PyTorch Inference for Pixel2Mesh++ (Design B v4)

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
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

# [DESIGN.B][CAMFM.A2b_STEADY_STATE] cuDNN autotune + TF32 tensor cores
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.models_mvp2m_pytorch import MVP2MNet
from modules.models_p2mpp_exact import MeshNetPyTorch


class MaxSpeedInferenceEngine:
    """Maximum speed inference with safe optimizations"""
    
    def __init__(self, stage1_checkpoint, stage2_checkpoint, mesh_data_path, device='cuda'):
        self.device = torch.device(device)
        
        print('Loading mesh data...')
        with open(mesh_data_path, 'rb') as f:
            pkl = pickle.load(f)
        
        # [DESIGN.B][CAMFM.A2a_GPU_RESIDENCY] Pre-load all data to GPU with optimal memory layout
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
        
        # [DESIGN.B][CAMFM.A2c_MEM_LAYOUT] Pre-allocate delta_coord buffer
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
        # [DESIGN.B][CAMFM.A2b_STEADY_STATE] Extended warmup for cuDNN autotuner
        dummy_img = torch.randn(3, 3, 224, 224, device=self.device)
        dummy_cam = np.array([[0, 25, 0, 1.9, 25], [162, 25, 0, 1.9, 25], [198, 25, 0, 1.9, 25]])
        
        with torch.inference_mode():
            for _ in range(15):  # [CAMFM.A2b] 15 warmup iterations
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
    
    # [DESIGN.B][CAMFM.A2d_OPTIONAL_ACCEL] torch.inference_mode disables autograd
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


def save_mesh_obj(vertices, faces, filepath):
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for face in faces:
            if len(face) == 3:
                f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1_checkpoint', default='pytorch_impl/checkpoints/mvp2m_converted.npz')
    parser.add_argument('--stage2_checkpoint', default='pytorch_impl/checkpoints/meshnet_converted.npz')
    parser.add_argument('--mesh_data', default='data/iccv_p2mpp.dat')
    parser.add_argument('--test_file', default='data/designB_eval_full.txt')
    parser.add_argument('--image_root', default='data/designA_subset/ShapeNetRendering/rendering_only')
    parser.add_argument('--output_dir', default='outputs/designB/eval_meshes_v4')
    
    args = parser.parse_args()
    
    engine = MaxSpeedInferenceEngine(
        args.stage1_checkpoint, args.stage2_checkpoint, args.mesh_data, device='cuda'
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.test_file, 'r') as f:
        test_list = [line.strip() for line in f if line.strip()]
    
    print(f'\nProcessing {len(test_list)} samples...')
    
    times = []
    device = torch.device('cuda')
    
    for idx, sample_id in enumerate(test_list):
        imgs, poses = load_sample(args.image_root, sample_id)
        imgs_tensor = torch.from_numpy(imgs.transpose(0, 3, 1, 2)).to(device)
        
        torch.cuda.synchronize()
        start = time.time()
        mesh_gpu = engine.infer(imgs_tensor, poses)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        
        mesh = mesh_gpu.cpu().numpy()
        sample_base = sample_id.replace('.dat', '')
        save_mesh_obj(mesh, engine.faces, os.path.join(args.output_dir, f'{sample_base}_predict.obj'))
        np.savetxt(os.path.join(args.output_dir, f'{sample_base}_predict.xyz'), mesh)
        
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f'[{idx+1}/{len(test_list)}] {sample_id}: {elapsed*1000:.1f}ms')
    
    print(f'\n{"="*60}')
    print(f'DESIGN B v4 - MAXIMUM SPEED RESULTS')
    print(f'{"="*60}')
    print(f'Total samples: {len(times)}')
    print(f'Mean inference time: {np.mean(times)*1000:.1f}ms/sample')
    print(f'Std: {np.std(times)*1000:.1f}ms')
    print(f'Min: {np.min(times)*1000:.1f}ms')
    print(f'Max: {np.max(times)*1000:.1f}ms')
    print(f'Throughput: {len(times)/sum(times):.1f} samples/sec')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
