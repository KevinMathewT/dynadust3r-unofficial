"""
Author: DynaDUSt3R Project
Date: 2025-04-12
"""

import torch


def torch_cam_pc_to_cam_pm(cam_pc, intrinsics, image_shape):
    """
    Projects camera-space point cloud to camera point map using fully vectorized operations.
    
    Args:
        cam_pc (torch.Tensor): Camera point cloud tensor of shape (B, N, 3/4)
        intrinsics (torch.Tensor): Camera intrinsic matrix of shape (3, 3)
        image_shape (tuple): Target output shape (H, W)
    
    Returns:
        torch.Tensor: Camera point map of shape (B, H, W, 4)
    """
    batch_size, num_points = cam_pc.shape[0], cam_pc.shape[1]
    h, w = image_shape
    device = cam_pc.device
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    
    x, y, z = cam_pc[..., 0], cam_pc[..., 1], cam_pc[..., 2]
    valid = cam_pc[..., 3] > 0 if cam_pc.shape[2] == 4 else torch.ones_like(z, dtype=torch.bool)
    z_valid = z > 0
    valid = valid & z_valid
    
    safe_z = torch.where(valid, z, torch.ones_like(z))
    u = (x * fx / safe_z + cx)
    v = (y * fy / safe_z + cy)
    
    u = torch.round(u).long()
    v = torch.round(v).long()
    
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h) & valid
    
    batch_indices = torch.arange(batch_size, device=device).view(batch_size, 1).expand(batch_size, num_points)
    
    indices = torch.stack([
        batch_indices[in_bounds],
        v[in_bounds],
        u[in_bounds]
    ], dim=1)
    
    points = cam_pc[in_bounds, :3]
    
    cam_pm = torch.zeros((batch_size, h, w, 4), device=device)
    
    for i in range(3):
        cam_pm[indices[:, 0], indices[:, 1], indices[:, 2], i] = points[:, i]
    
    cam_pm[indices[:, 0], indices[:, 1], indices[:, 2], 3] = 1.0
    
    return cam_pm


def torch_cam_pc_to_cam_pm_batch(cam_pc, intrinsics, image_shape):
    """
    Alternative implementation that processes each batch item separately.
    Use when dealing with memory constraints.
    
    Args:
        cam_pc (torch.Tensor): Camera point cloud tensor of shape (B, N, 3/4)
        intrinsics (torch.Tensor): Camera intrinsic matrix of shape (3, 3)
        image_shape (tuple): Target output shape (H, W)
    
    Returns:
        torch.Tensor: Camera point map of shape (B, H, W, 4)
    """
    batch_size = cam_pc.shape[0]
    h, w = image_shape
    device = cam_pc.device
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    
    cam_pm = torch.zeros((batch_size, h, w, 4), device=device)
    
    for b in range(batch_size):
        pts = cam_pc[b]
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        validity = pts[:, 3] > 0 if pts.shape[1] == 4 else torch.ones_like(z, dtype=torch.bool)
        valid_pts = (z > 0) & validity
        
        if not valid_pts.any():
            continue
            
        x_valid, y_valid, z_valid = x[valid_pts], y[valid_pts], z[valid_pts]
        u = (x_valid * fx / z_valid + cx).round().long()
        v = (y_valid * fy / z_valid + cy).round().long()
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        
        if in_bounds.any():
            valid_pts_3d = pts[valid_pts][in_bounds]
            cam_pm[b, v[in_bounds], u[in_bounds], :3] = valid_pts_3d[:, :3]
            cam_pm[b, v[in_bounds], u[in_bounds], 3] = 1.0
    
    return cam_pm


def project_points_to_pixels(points_3d, intrinsics):
    """
    Projects 3D points to 2D pixel coordinates using camera intrinsics.
    
    Args:
        points_3d (torch.Tensor): 3D points in camera coordinates of shape (B, H, W, 3)
        intrinsics (torch.Tensor): Camera intrinsic matrix of shape (3, 3)
    
    Returns:
        tuple: (u, v, z_valid, in_bounds) - Pixel coordinates and validity masks
    """
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    x, y, z = points_3d[..., 0], points_3d[..., 1], points_3d[..., 2]
    z_valid = z > 0
    u, v = torch.zeros_like(x), torch.zeros_like(y)
    
    u[z_valid] = (x[z_valid] * fx / z[z_valid] + cx)
    v[z_valid] = (y[z_valid] * fy / z[z_valid] + cy)
    u, v = torch.round(u).long(), torch.round(v).long()
    
    h, w = points_3d.shape[1:3]
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h) & z_valid
    
    return u, v, z_valid, in_bounds


def create_point_maps_from_projections(points_3d, u, v, in_bounds, image_shape, device):
    """
    Creates point maps from 3D points and their projected 2D coordinates.
    
    Args:
        points_3d (torch.Tensor): 3D points of shape (B, H, W, 3)
        u (torch.Tensor): Projected x-coordinates of shape (B, H, W)
        v (torch.Tensor): Projected y-coordinates of shape (B, H, W)
        in_bounds (torch.Tensor): Mask for valid and in-bounds points of shape (B, H, W)
        image_shape (tuple): Output image shape (H, W)
        device: PyTorch device
    
    Returns:
        torch.Tensor: Point map with validity mask of shape (B, H, W, 4)
    """
    batch_size = points_3d.shape[0]
    h, w = image_shape
    point_map = torch.zeros((batch_size, h, w, 4), device=device)
    
    for b in range(batch_size):
        valid_idx = torch.nonzero(in_bounds[b], as_tuple=True)
        if len(valid_idx[0]) > 0:
            src_y, src_x = valid_idx
            tgt_y, tgt_x = v[b, src_y, src_x], u[b, src_y, src_x]
            valid_points = points_3d[b, src_y, src_x]
            point_map[b, tgt_y, tgt_x, :3] = valid_points
            point_map[b, tgt_y, tgt_x, 3] = 1.0
    
    return point_map
