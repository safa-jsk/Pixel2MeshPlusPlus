"""
Simple PyTorch implementation of Chamfer Distance
Fallback if PyTorch3D is not available
"""

import torch


def chamfer_distance_naive(pred_points, gt_points):
    """
    Compute chamfer distance between two point clouds
    
    Args:
        pred_points: [B, N, 3] predicted points
        gt_points: [B, M, 3] ground truth points
        
    Returns:
        loss: scalar chamfer distance
    """
    # pred_points: [B, N, 3]
    # gt_points: [B, M, 3]
    
    # Expand dimensions for broadcasting
    pred_exp = pred_points.unsqueeze(2)  # [B, N, 1, 3]
    gt_exp = gt_points.unsqueeze(1)      # [B, 1, M, 3]
    
    # Compute pairwise distances: [B, N, M]
    distances = torch.sum((pred_exp - gt_exp) ** 2, dim=-1)
    
    # Forward direction: for each predicted point, find nearest GT point
    forward_dist = torch.min(distances, dim=2)[0]  # [B, N]
    forward_loss = forward_dist.mean()
    
    # Backward direction: for each GT point, find nearest predicted point
    backward_dist = torch.min(distances, dim=1)[0]  # [B, M]
    backward_loss = backward_dist.mean()
    
    # Total chamfer distance
    chamfer_loss = forward_loss + backward_loss
    
    return chamfer_loss


def chamfer_distance(pred_points, gt_points):
    """
    Compute chamfer distance - uses PyTorch3D if available, otherwise fallback
    """
    try:
        from pytorch3d.loss import chamfer_distance as p3d_chamfer
        loss, _ = p3d_chamfer(pred_points, gt_points)
        return loss
    except ImportError:
        print("[INFO] Using custom chamfer distance implementation")
        return chamfer_distance_naive(pred_points, gt_points)
