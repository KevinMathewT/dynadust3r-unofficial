"""
Author: Kevin Mathew T
Date: 2025-08-17
LinkedIn: https://www.linkedin.com/in/kevinmathewt/

Geometry and camera utilities for stereo and motion learning.

This module centralizes a broad set of geometry helpers used across the
training and evaluation pipeline. It focuses on lightweight, explicit
operations that map cleanly to common computer vision conventions
without hiding important semantics. The functions are intentionally
minimal and composable so they can be applied to large WebDataset-driven
workloads and multi‑GPU training without unnecessary overhead.

What this module provides
- Coordinate transforms:
  - World ↔ Camera conversions using standard 4x4 or 3x4 extrinsics
    where extrinsics encode world→cam (P_c = R P_w + t).
  - Depth map to point cloud conversions in camera/world frames.
  - Point map creation where per‑pixel XYZ lives in a chosen reference
    camera while pixels index a source image (useful for stereo).

- Projection helpers:
  - Project 3D points to image plane via intrinsics with proper handling
    of visibility (z > 0) and image bounds checks.

- Motion map construction:
  - Direct 3D motion in a chosen reference frame for pairs of frames
    (e.g., left→mid, right→mid, left→right, right→left), stored sparsely
    at the projection locations of source points.

- Camera intrinsics updates under image transforms:
  - Crop/resize utilities that keep intrinsics consistent when images are
    cropped or rescaled, preserving geometric correctness.

Design assumptions and conventions
- Extrinsics are world→cam. Inverting a 3x4 [R|t] or 4x4 pose recovers
  cam→world when needed.
- Depth values are measured along the optical z axis in camera space.
- Validity masks (stored in the last channel for point/motion maps) are
  1 for valid and 0 for invalid.
- Numpy is used for CPU paths and Torch for GPU‑friendly tensor paths; we
  avoid mixing without explicit conversion.

Numerical safety notes
- Projections skip points with non‑positive or non‑finite depth to avoid
  division errors and NaNs.
- Bounds checks are applied before writing into image‑shaped outputs.

Performance notes
- Functions are written to minimize temporary allocations and to permit
  batched execution in data loaders. Where appropriate, vectorized numpy
  or torch operations are preferred over Python loops.

This file intentionally avoids any side effects beyond importing standard
libraries. Callers are expected to manage devices, dtypes, and casting
explicitly when moving between numpy and torch tensors.
"""

# Standard imports are centralized here to keep the module import order
# clear and to avoid scattered imports throughout the file.
import sys as _sys
import numpy as np
import torch
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation

from utils.misc import invalid_to_zeros, invalid_to_nans

# self-alias so calls to geo.* refer to functions in this module
geo = _sys.modules[__name__]


def compute_motion_map_direct(
    source_world_pc: np.ndarray,
    target_world_pc: np.ndarray,
    source_cam: tuple,
    target_cam: tuple,
    reference_cam: tuple,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """
    Compute motion map from source to target points, expressing motion in reference camera frame.
    
    This function computes motion vectors (target - source) for corresponding 3D points,
    stores them at the pixel locations where source points project in the source camera's image,
    and expresses the motion vectors in the reference camera's coordinate frame.
    
    Args:
        source_world_pc (np.ndarray): (N, 4) source points in world coords with validity
        target_world_pc (np.ndarray): (N, 4) target points in world coords with validity
        source_cam (tuple): (intrinsics, extrinsics) for source camera
        target_cam (tuple): (intrinsics, extrinsics) for target camera
        reference_cam (tuple): (intrinsics, extrinsics) for reference frame
        image_shape (tuple): (H, W) output image dimensions
        
    Returns:
        np.ndarray: (H, W, 4) motion map with motion vectors in reference frame
    """
    H, W = image_shape
    motion_map = np.zeros((H, W, 4), dtype=np.float32)
    
    # Extract valid points
    valid_mask = (source_world_pc[:, 3] > 0) & (target_world_pc[:, 3] > 0)
    if not np.any(valid_mask):
        return motion_map
        
    source_world = source_world_pc[valid_mask, :3]  # (M, 3)
    target_world = target_world_pc[valid_mask, :3]  # (M, 3)
    
    # Transform to reference camera coordinates
    source_ref = geo.world_pc_to_cam_pc(source_world, reference_cam)  # (M, 3)
    target_ref = geo.world_pc_to_cam_pc(target_world, reference_cam)  # (M, 3)
    
    # Compute motion in reference frame
    motion_3d = target_ref - source_ref  # (M, 3)
    
    # Project source points to source camera image
    source_cam_coords = geo.world_pc_to_cam_pc(source_world, source_cam)  # (M, 3)
    
    # Extract camera parameters
    intrinsics = source_cam[0]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Project to image coordinates
    x, y, z = source_cam_coords[:, 0], source_cam_coords[:, 1], source_cam_coords[:, 2]
    z_pos = z > 0
    if not np.any(z_pos):
        return motion_map
        
    x, y, z = x[z_pos], y[z_pos], z[z_pos]
    motion_3d = motion_3d[z_pos]
    
    u = np.round(fx * x / z + cx).astype(int)
    v = np.round(fy * y / z + cy).astype(int)
    
    # Bounds check
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(in_bounds):
        return motion_map
        
    u, v = u[in_bounds], v[in_bounds]
    motion_valid = motion_3d[in_bounds]
    
    # Write to motion map
    motion_map[v, u, :3] = motion_valid
    motion_map[v, u, 3] = 1
    
    return motion_map


def compute_all_motion_maps(
    left_world_pc: np.ndarray,
    mid_world_pc: np.ndarray, 
    right_world_pc: np.ndarray,
    left_cam: tuple,
    mid_cam: tuple,
    right_cam: tuple,
    image_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    """
    Compute all required motion maps with clear semantics.
    All motions are expressed in the left camera (reference) frame.
    
    Returns:
        dict with keys: "l2m", "r2m", "l2r", "r2l"
    """
    # Use left camera as reference for all motions
    reference_cam = left_cam
    
    motion_maps = {}
    
    # Left to Mid
    motion_maps["l2m"] = compute_motion_map_direct(
        left_world_pc, mid_world_pc,
        left_cam, mid_cam, reference_cam,
        image_shape
    )
    
    # Right to Mid
    motion_maps["r2m"] = compute_motion_map_direct(
        right_world_pc, mid_world_pc,
        right_cam, mid_cam, reference_cam,
        image_shape
    )
    
    # Left to Right
    motion_maps["l2r"] = compute_motion_map_direct(
        left_world_pc, right_world_pc,
        left_cam, right_cam, reference_cam,
        image_shape
    )
    
    # Right to Left
    motion_maps["r2l"] = compute_motion_map_direct(
        right_world_pc, left_world_pc,
        right_cam, left_cam, reference_cam,
        image_shape
    )
    
    return motion_maps

"""
Author: Kevin Mathew T
Date: 2025-03-10
"""

 


def inv(mat):
    """
    Invert a pose matrix.

    Supports:
      - 4×4 homogeneous matrices
      - 3×4 [R|t] matrices

    Args:
        mat (np.ndarray or torch.Tensor): shape (...,4,4) or (...,3,4)

    Returns:
        np.ndarray or torch.Tensor: inverted 4×4 matrix
    """
    is_torch = isinstance(mat, torch.Tensor)
    h, w = mat.shape[-2:]
    if h == 4 and w == 4:
        return torch.linalg.inv(mat) if is_torch else np.linalg.inv(mat)
    if h == 3 and w == 4:
        if is_torch:
            R, t = mat[:, :3], mat[:, 3]
            Ri = R.transpose(-2, -1)
            ti = -Ri @ t.unsqueeze(-1)
            E = torch.eye(4, dtype=mat.dtype, device=mat.device)
            E[:3, :3], E[:3, 3] = Ri, ti.squeeze(-1)
            return E
        R, t = mat[:, :3], mat[:, 3]
        Ri, ti = R.T, -R.T @ t
        E = np.eye(4, dtype=mat.dtype)
        E[:3, :3], E[:3, 3] = Ri, ti
        return E
    raise ValueError(f"Can't invert matrix of shape {mat.shape}")


def recolor(pm, cam1, cam2, img2):
    """
    Recolor a point map (wrt cam1) using colors from cam2's image.

    Args:
        pm (np.ndarray): (H, W, 4) point map in cam1 coords (X,Y,Z,valid)
        cam1 (tuple): (intrinsics, extrinsics) of cam1
        cam2 (tuple): (intrinsics, extrinsics) of cam2
        img2 (np.ndarray): (H2, W2, 3) image from cam2

    Returns:
        np.ndarray: (H, W, 3) RGB image in cam1 pixel grid
    """
    intrinsics1, extrinsics1 = cam1
    intrinsics2, extrinsics2 = cam2
    H, W = pm.shape[:2]

    mask = pm[..., 3] > 0
    pts_cam1 = pm[mask, :3]

    R1, t1 = extrinsics1[:3, :3], extrinsics1[:3, 3]
    # cam1 -> world
    pts_world = (R1.T @ (pts_cam1 - t1).T).T

    R2, t2 = extrinsics2[:3, :3], extrinsics2[:3, 3]
    # world -> cam2
    pts_cam2 = (R2 @ pts_world.T).T + t2

    # project into cam2 image
    uvw = (intrinsics2 @ pts_cam2.T).T
    u = (uvw[:, 0] / uvw[:, 2]).astype(int)
    v = (uvw[:, 1] / uvw[:, 2]).astype(int)

    valid = (uvw[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    out = np.zeros((H * W, 3), dtype=img2.dtype)

    idx = np.flatnonzero(mask.ravel())[valid]
    out[idx] = img2[v[valid], u[valid]]

    return out.reshape(H, W, 3)


def decompose_extrinsics(extrinsics):
    """
    Decomposes a 4x4 extrinsic matrix into translation and rotation (as quaternion).
    """
    translation = extrinsics[:3, 3]
    rotation_matrix = extrinsics[:3, :3]
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat()
    return translation, quaternion


def dm_to_cam_pc(dm, cam):
    """
    Converts a dm to a cam point cloud.

    Args:
        dm (numpy.ndarray): (H, W) - Depthmap with per-pixel depth values.
        cam (tuple): (intrinsics, extrinsics) - Camera parameters.

    Returns:
        numpy.ndarray: (N, 3) - Camera point cloud in cam coordinates.
    """
    intrinsics, _ = cam
    H, W = dm.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))  # (H, W)
    x = (u - cx) / fx  # (H, W)
    y = (v - cy) / fy  # (H, W)
    z = dm  # (H, W)

    mask = z > 0  # mask for negative/zero z
    x, y, z = x[mask], y[mask], z[mask]

    cam_pc = np.stack([x * z, y * z, z], axis=-1).reshape(-1, 3)  # (N, 3)
    return cam_pc  # (N, 3)


def cam_pc_to_world_pc(cam_pc, cam):
    """
    Converts a cam point cloud to world coordinates using the extrinsic matrix.

    The provided extrinsics represent the transformation from **world to cam**.
    To convert cam points back to world coordinates, we apply:

        P_w = R^T (P_c - t)

    Args:
        cam_pc (numpy.ndarray): (N, 3) - Camera point cloud in cam coordinates.
        cam (tuple): (intrinsics, extrinsics) - Camera parameters.

    Returns:
        numpy.ndarray: (N, 3) - World point cloud in world coordinates.
    """
    _, extrinsics = cam
    R = extrinsics[:3, :3]  # (3, 3)
    t = extrinsics[:3, 3]  # (3,)
    world_pc = (R.T @ (cam_pc - t).T).T  # (N, 3)
    return world_pc  # (N, 3)


def world_pc_to_cam_pc(world_pc, cam):
    """
    Converts a world point cloud to cam coordinates using the extrinsic matrix.

    The provided extrinsics represent the transformation from **world to cam**:

        P_c = R P_w + t

    Args:
        world_pc (numpy.ndarray): (N, 3) - World point cloud in world coordinates.
        cam (tuple): (intrinsics, extrinsics) - Camera parameters.

    Returns:
        numpy.ndarray: (N, 3) - Camera point cloud in cam coordinates.
    """
    _, extrinsics = cam
    R = extrinsics[:3, :3]  # (3, 3)
    t = extrinsics[:3, 3]  # (3,)
    cam_pc = (R @ world_pc.T).T + t  # (N, 3)
    return cam_pc  # (N, 3)


def dm_to_world_pc(dm, cam):
    """
    Converts a dm directly to a world point cloud.

    This function combines:
        - dm_to_cam_pc
        - cam_pc_to_world_pc

    Args:
        dm (numpy.ndarray): (H, W) - Depthmap with per-pixel depth values.
        cam (tuple): (intrinsics, extrinsics) - Camera parameters.

    Returns:
        numpy.ndarray: (N, 3) - World point cloud in world coordinates.
    """
    cam_pc = dm_to_cam_pc(dm, cam)
    world_pc = cam_pc_to_world_pc(cam_pc, cam)
    return world_pc  # (N, 3)


def dm_to_cam_pm(dm, cam):
    """
    Converts a dm to a cam point map (H, W, 3), where each pixel contains a 3D point.

    Args:
        dm (numpy.ndarray): (H, W) - Depth map with per-pixel depth values.
        cam (tuple): (intrinsics, extrinsics) - Camera parameters.

    Returns:
        numpy.ndarray: (H, W, 4) - Camera point map in cam coordinates.
    """
    intrinsics, _ = cam
    H, W = dm.shape  # (H, W)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]  # scalar
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]  # scalar
    u, v = np.meshgrid(np.arange(W), np.arange(H))  # (H, W)
    x = (u - cx) / fx  # (H, W)
    y = (v - cy) / fy  # (H, W)
    z = dm  # (H, W)
    cam_pm = np.stack([x * z, y * z, z], axis=-1)  # (H, W, 3)
    mask = (z > 0).astype(np.uint8)  # (H, W)
    cam_pm = np.concatenate([cam_pm, mask[..., None]], axis=-1)  # (H, W, 4)
    return cam_pm  # (H, W, 4)


def cam_pc_to_cam_pm_with_torch(cam_pc, cam, image_shape, valid=False):
    """
    Projects a cam-space point cloud (N,3) or (N,4) onto an image plane and
    stores the corresponding 3D points in a (H, W, 4) tensor at their respective pixel locations.
    The last channel in the output (H, W, 4) indicates whether a pixel is valid (1) or invalid (0).

    Args:
        cam_pc (torch.Tensor): (N, 3) or (N, 4) - 3D points in cam coordinates.
        cam (tuple): (intrinsics, extrinsics) - Camera parameters (extrinsics ignored).
        image_shape (tuple): (H, W) - Shape of the target image.
        valid (bool): If True, uses the fourth channel of cam_pc for validity.

    Returns:
        torch.Tensor: (H, W, 4) - Camera point map with validity mask in the last channel.
    """
    intrinsics, _ = cam
    height, width = image_shape

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x, y, z = cam_pc[:, 0], cam_pc[:, 1], cam_pc[:, 2]
    u = (x * fx / z + cx).int()
    v = (y * fy / z + cy).int()

    cam_pm = torch.zeros((height, width, 4), dtype=cam_pc.dtype, device=cam_pc.device)

    # negative z
    validity_mask = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)

    if valid and cam_pc.shape[1] == 4:
        validity_mask = validity_mask & (cam_pc[:, 3] > 0)

    cam_pm[v[validity_mask], u[validity_mask], :3] = cam_pc[validity_mask, :3]
    cam_pm[v[validity_mask], u[validity_mask], 3] = 1  # mark valid pixels

    return cam_pm


def cam_pc_to_cam_pm(cam_pc, cam, image_shape, valid=False):
    """
    Projects a cam-space point cloud (N,3) or (N,4) onto an image plane and
    stores the corresponding 3D points in a (H, W, 4) array at their respective pixel locations.
    The last channel in the output (H, W, 4) indicates whether a pixel is valid (1) or invalid (0).

    Args:
        cam_pc (np.ndarray): (N, 3) or (N, 4) - 3D points in cam coordinates.
        cam (tuple): (intrinsics, extrinsics) - Camera parameters (extrinsics ignored).
        image_shape (tuple): (H, W) - Shape of the target image.
        valid (bool): If True, uses the fourth channel of cam_pc for validity.

    Returns:
        np.ndarray: (H, W, 4) - Camera point map with validity mask in the last channel.
    """
    intrinsics, _ = cam
    height, width = image_shape

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x, y, z = cam_pc[:, 0], cam_pc[:, 1], cam_pc[:, 2]
    
    # Pre-filter valid z values to avoid division by zero/negative values
    z_valid = (z > 1e-6) & np.isfinite(z)  # Use small epsilon instead of 0
    
    # Initialize with invalid values
    u = np.full_like(x, -1, dtype=int)
    v = np.full_like(y, -1, dtype=int)
    
    # Only compute projections for valid z values
    if np.any(z_valid):
        u_valid = x[z_valid] * fx / z[z_valid] + cx
        v_valid = y[z_valid] * fy / z[z_valid] + cy
        
        # Check for finite values before casting
        finite_mask = np.isfinite(u_valid) & np.isfinite(v_valid)
        
        if np.any(finite_mask):
            u[z_valid] = np.where(finite_mask, u_valid.astype(int), -1)
            v[z_valid] = np.where(finite_mask, v_valid.astype(int), -1)

    cam_pm = np.zeros((height, width, 4), dtype=cam_pc.dtype)

    # Create comprehensive validity mask
    validity_mask = (
        z_valid &  # z is positive and finite
        (u >= 0) & (u < width) & 
        (v >= 0) & (v < height)
    )

    if valid and cam_pc.shape[1] == 4:
        validity_mask &= cam_pc[:, 3] > 0

    # Only process valid points
    if np.any(validity_mask):
        valid_indices = np.where(validity_mask)[0]
        cam_pm[v[valid_indices], u[valid_indices], :3] = cam_pc[valid_indices, :3]
        cam_pm[v[valid_indices], u[valid_indices], 3] = 1  # mark valid pixels

    return cam_pm


def create_pm_in_ref_frame(world_pc, validity, cam_source, cam_reference, 
                                       image_shape, pm_source="3d_tracks"):
    """
    Creates a point map where pixels correspond to cam_source's image space, 
    but 3D coordinates are expressed in cam_reference's coordinate frame.
    
    Args:
        world_pc (np.ndarray): (N, 3) - 3D points in world coordinates
        validity (np.ndarray): (N, 1) - Validity mask for each point
        cam_source (tuple): (intrinsics, extrinsics) - Camera to project points onto
        cam_reference (tuple): (intrinsics, extrinsics) - Camera whose coordinate frame to use
        image_shape (tuple): (H, W) - Shape of the output point map
        pm_source (str): Either "3d_tracks" (sparse) or "dm" (dense)
        
    Returns:
        np.ndarray: (H, W, 4) - Point map with pixels in source image space and 
                                 3D coords in reference frame
    """
    h, w = image_shape
    
    # Transform points to reference camera coordinates (for storage)
    cam_pc_in_ref = world_pc_to_cam_pc(world_pc, cam_reference)  # (N, 3)
    
    # Transform points to source camera coordinates (for projection)
    cam_pc_in_source = world_pc_to_cam_pc(world_pc, cam_source)  # (N, 3)
    
    # Initialize output point map
    pm = np.zeros((h, w, 4), dtype=np.float32)
    
    if pm_source == "3d_tracks":
        # Vectorized sparse tracks projection
        intrinsics = cam_source[0]
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # Flatten validity for easier indexing
        validity_flat = validity.flatten() > 0
        
        # Extract valid points
        valid_cam_pc_source = cam_pc_in_source[validity_flat]
        valid_cam_pc_ref = cam_pc_in_ref[validity_flat]
        
        if len(valid_cam_pc_source) > 0:
            # Extract coordinates
            x, y, z = valid_cam_pc_source[:, 0], valid_cam_pc_source[:, 1], valid_cam_pc_source[:, 2]
            
            # Filter points with positive depth
            depth_valid = z > 0
            x = x[depth_valid]
            y = y[depth_valid]
            z = z[depth_valid]
            valid_cam_pc_ref = valid_cam_pc_ref[depth_valid]
            
            if len(x) > 0:
                # Project all points at once
                u = (x * fx / z + cx).astype(np.int32)
                v = (y * fy / z + cy).astype(np.int32)
                
                # Bounds check
                in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
                u = u[in_bounds]
                v = v[in_bounds]
                z = z[in_bounds]  # keep z aligned with u/v
                valid_cam_pc_ref = valid_cam_pc_ref[in_bounds]
                
                # Handle potential duplicate projections by keeping the closest point
                if len(u) > 0:
                    # Create a unique key for each pixel
                    pixel_keys = v * w + u

                    # sort first by pixel id, then by depth z (ascending → nearest first)
                    order = np.lexsort((z, pixel_keys))         # stable, O(N log N)
                    pixel_keys_sorted      = pixel_keys[order]
                    u_sorted, v_sorted     = u[order], v[order]
                    ref_points_sorted      = valid_cam_pc_ref[order]
                    
                    # Find unique pixels and their first occurrence
                    unique_pixels, unique_indices = np.unique(pixel_keys_sorted, return_index=True)
                    
                    # Extract unique u, v coordinates (FIX: use sorted arrays)
                    u_unique = u_sorted[unique_indices]
                    v_unique = v_sorted[unique_indices]
                    pts_unique = ref_points_sorted[unique_indices]
                    
                    # Update point map
                    pm[v_unique, u_unique, :3] = pts_unique
                    pm[v_unique, u_unique, 3] = 1
                        
    else:  # Dense point map
        # First create point map in source camera coordinates to get pixel positions
        cam_pc_in_source_valid = np.concatenate([cam_pc_in_source, validity], axis=1)
        pm_source = cam_pc_to_cam_pm(cam_pc_in_source_valid, (cam_source[0], None), 
                                     image_shape, valid=True)
        
        # Now fill in the reference camera coordinates
        valid_mask = pm_source[..., 3] > 0
        if np.any(valid_mask):
            # Extract valid points and transform
            valid_points_source = pm_source[valid_mask, :3]
            # Transform from source to world to reference
            valid_points_world = cam_pc_to_world_pc(valid_points_source, cam_source)
            valid_points_ref = world_pc_to_cam_pc(valid_points_world, cam_reference)
            # Put back into point map
            pm[valid_mask, :3] = valid_points_ref
            pm[valid_mask, 3] = 1
            
    return pm


def create_pm_from_dm_in_ref_frame(depthmap, cam_source, cam_reference):
    """
    Creates a point map from a depth map where pixels correspond to the source camera's 
    image space, but 3D coordinates are expressed in the reference camera's frame.
    
    Args:
        depthmap (np.ndarray): (H, W) - Depth map from source camera
        cam_source (tuple): (intrinsics, extrinsics) - Camera that captured the depth map
        cam_reference (tuple): (intrinsics, extrinsics) - Camera whose coordinate frame to use
        
    Returns:
        np.ndarray: (H, W, 4) - Point map with 3D coords in reference frame
    """
    # First create point map in source camera's coordinate system
    pm_in_source = dm_to_cam_pm(depthmap, cam_source)  # (H, W, 4)
    
    # Transform the 3D coordinates to reference camera frame
    h, w = depthmap.shape
    pm_in_reference = np.zeros((h, w, 4), dtype=np.float32)
    valid_mask = pm_in_source[..., 3] > 0
    
    if np.any(valid_mask):
        # Extract valid 3D points
        valid_points_source = pm_in_source[valid_mask, :3]  # (N, 3) in source camera coords
        
        # Transform to world then to reference camera
        valid_points_world = cam_pc_to_world_pc(valid_points_source, cam_source)
        valid_points_ref = world_pc_to_cam_pc(valid_points_world, cam_reference)
        
        # Put back into point map maintaining pixel locations
        pm_in_reference[valid_mask, :3] = valid_points_ref
        pm_in_reference[valid_mask, 3] = 1
        
    return pm_in_reference


def compute_scale_difference(approx_pm, actual_pm):
    """
    Computes the scale difference between two cam point maps (H, W, 4).
    The first map is an approximation from 3D tracks (subset of actual points).

    Args:
        approx_pm (np.ndarray): (H, W, 4) - Approximate cam point map.
        actual_pm (np.ndarray): (H, W, 4) - Actual cam point map.

    Returns:
        float: Estimated scale difference.
    """
    valid_approx = approx_pm[..., 3] > 0
    valid_actual = actual_pm[..., 3] > 0
    valid_mask = valid_approx & valid_actual

    approx_points = approx_pm[valid_mask, :3]
    actual_points = actual_pm[valid_mask, :3]

    if len(approx_points) == 0:
        return None  # no valid points

    scale_ratios = np.linalg.norm(actual_points, axis=1) / np.linalg.norm(
        approx_points, axis=1
    )
    print(f"Scale Ratios: {scale_ratios}")
    return np.median(scale_ratios)


def get_motion_map_from_world_pc(
    world_pc_valid_list: list[np.ndarray],
    cam_list: list[tuple[np.ndarray, np.ndarray]],
    image_dimensions: tuple[int, int],
) -> np.ndarray:
    """
    build per-pixel 3-d motion maps **from every source frame to the *last* frame**  
    while expressing  ΔP  in the **reference camera** (= `cam_list[0]`) coordinates.

    for each k ∈ [0, T-2] we store  

        ΔP = P_last^ref − P_k^ref         # (M, 3) in ref-cam coords  

    at the pixel (u,v) where P_k projects in frame-k’s image grid.

    Args
    ----
    world_pc_valid_list : list[np.ndarray]
        length T list of arrays shaped (N, 4): xyz (world) + validity flag.
        all arrays share track ordering, so row-i is the same point across time.
    cam_list : list[tuple[np.ndarray, np.ndarray]]
        length T list of (intrinsics (3,3), extrinsics (4,4)) for each frame.
        `cam_list[0]` is chosen as the *reference* camera.
    image_dimensions : (H, W)
        height and width of the images (assumed identical for every frame).

    Returns
    -------
    np.ndarray
        motion volume of shape (T-1, H, W, 4).  
        for k ∈ [0, T-2]:

            motion_map[k, v, u, :3]  – ΔP in reference-cam coords  
            motion_map[k, v, u,  3]  – 1 if the pixel is valid, else 0
    """
    import numpy as np

    H, W = image_dimensions                     # image size
    T = len(world_pc_valid_list)                # number of frames
    assert T == len(cam_list), "list lengths must match"

    ref_cam = cam_list[0]                       # ((3,3), (4,4))

    # data for last (target) frame
    world_pc_last   = world_pc_valid_list[-1]   # (N,4)
    last_valid_mask = world_pc_last[:, 3] > 0   # (N,)

    motion_map = np.zeros((T - 1, H, W, 4), dtype=np.float32)

    for k in range(T - 1):
        intr_k, _ = cam_list[k]                 # intrinsics of source frame-k
        world_pc_k = world_pc_valid_list[k]     # (N,4)

        # tracks visible in both k and last
        valid_mask = (world_pc_k[:, 3] > 0) & last_valid_mask  # (N,)
        if not np.any(valid_mask):
            continue                            # no overlapping tracks

        world_k    = world_pc_k  [valid_mask, :3]              # (M,3)
        world_last = world_pc_last[valid_mask, :3]             # (M,3)

        # transform to reference camera coordinates
        cam_ref_k    = geo.world_pc_to_cam_pc(world_k,    ref_cam)  # (M,3)
        cam_ref_last = geo.world_pc_to_cam_pc(world_last, ref_cam)  # (M,3)

        motion_3d = cam_ref_last - cam_ref_k                 # (M,3)

        # project world_k to pixel grid of frame-k
        cam_k = geo.world_pc_to_cam_pc(world_k, cam_list[k]) # (M,3)

        x, y, z = cam_k[:, 0], cam_k[:, 1], cam_k[:, 2]      # (M,)
        fx, fy = intr_k[0, 0], intr_k[1, 1]
        cx, cy = intr_k[0, 2], intr_k[1, 2]

        z_pos = z > 0                                        # filter behind-cam
        if not np.any(z_pos):
            continue

        x, y, z   = x[z_pos], y[z_pos], z[z_pos]             # (M',)  (M' ≤ M)
        motion_3d = motion_3d[z_pos]                         # (M',3)

        u = np.round(fx * x / z + cx).astype(int)            # (M',)
        v = np.round(fy * y / z + cy).astype(int)            # (M',)

        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(in_bounds):
            continue

        u, v        = u[in_bounds], v[in_bounds]             # (M_valid,)
        motion_valid = motion_3d[in_bounds]                  # (M_valid,3)

        # write to motion map
        motion_map[k, v, u, :3] = motion_valid               # xyz displacements
        motion_map[k, v, u,  3] = 1                          # validity flag

    return motion_map



def get_motion_map_from_cam_pc(cam_pc_valid_list, ref_intrinsics, image_dimensions):
    """
    Computes a motion map between consecutive frames.

    Args:
        cam_pc_valid_list (list of numpy.ndarray): List of (N, 4) camera point clouds with validity mask.
        ref_intrinsics (numpy.ndarray): (3, 3) Camera intrinsic matrix.
        image_dimensions (tuple): (H, W) Image height and width.

    Returns:
        numpy.ndarray: (T-1, H, W, 4) Motion map where the last channel indicates validity.
    """
    H, W = image_dimensions  # (H, W)
    num_frames = len(cam_pc_valid_list)  # (T)

    motion_map = np.zeros((num_frames - 1, H, W, 4), dtype=np.float32)  # (T-1, H, W, 4)

    for i in range(num_frames - 1):  # (T-1)
        cam_pc_ref = cam_pc_valid_list[i]  # (N, 4)
        cam_pc_target = cam_pc_valid_list[i + 1]  # (N, 4)

        valid_mask = (cam_pc_ref[:, 3] > 0) & (cam_pc_target[:, 3] > 0)  # (N,)
        cam_pc_ref_valid = cam_pc_ref[valid_mask, :3]  # (M, 3)
        cam_pc_target_valid = cam_pc_target[valid_mask, :3]  # (M, 3)

        motion_3d = cam_pc_target_valid - cam_pc_ref_valid  # (M, 3)

        x, y, z = (
            cam_pc_ref_valid[:, 0],
            cam_pc_ref_valid[:, 1],
            cam_pc_ref_valid[:, 2],
        )  # (M,)
        fx, fy = ref_intrinsics[0, 0], ref_intrinsics[1, 1]  # (scalar, scalar)
        cx, cy = ref_intrinsics[0, 2], ref_intrinsics[1, 2]  # (scalar, scalar)

        # negative z
        z_positive_mask = z > 0  # (M,)
        x, y, z = x[z_positive_mask], y[z_positive_mask], z[z_positive_mask]  # (M',)
        motion_3d = motion_3d[z_positive_mask]  # (M', 3)

        u = (fx * x / z + cx).astype(int)  # (M')
        v = (fy * y / z + cy).astype(int)  # (M')

        in_bounds_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)  # (M')
        u_valid = u[in_bounds_mask]  # (M_valid,)
        v_valid = v[in_bounds_mask]  # (M_valid,)
        motion_valid = motion_3d[in_bounds_mask]  # (M_valid, 3)

        motion_map[i, v_valid, u_valid, :3] = motion_valid  # (M_valid, 3)
        motion_map[i, v_valid, u_valid, 3] = 1  # (M_valid,)

    return motion_map  # (T-1, H, W, 4)


##################################################
#        CROPPING AND RESIZING FUNCTIONS         #
##################################################


def crop_image_depthmap(image, depthmap, intrinsics, crop_bbox):
    """
    Apply consistent crop to image, depthmap and adjust camera intrinsics.

    Args:
        image: RGB image array (H, W, 3) or PIL Image
        depthmap: Depth map array (H, W)
        intrinsics: Camera intrinsic matrix (3, 3)
        crop_bbox: Crop bounding box as (left, top, right, bottom)

    Returns:
        tuple: (cropped_image, cropped_depthmap, adjusted_intrinsics)
    """
    l, t, r, b = crop_bbox

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)  # convert to pil image

    cropped_image = image.crop((l, t, r, b))  # (new_W, new_H)
    cropped_depthmap = (
        depthmap[t:b, l:r] if depthmap is not None else None
    )  # (new_H, new_W)

    adjusted_intrinsics = intrinsics.copy()  # (3, 3)
    adjusted_intrinsics[0, 2] -= l  # adjust cx
    adjusted_intrinsics[1, 2] -= t  # adjust cy

    return cropped_image, cropped_depthmap, adjusted_intrinsics


def rescale_image_depthmap(image, depthmap, intrinsics, target_resolution, force=True):
    """
    Rescale image and depthmap to match target resolution with appropriate interpolation.

    Args:
        image: RGB image (PIL Image or numpy array)
        depthmap: Depth map array (H, W)
        intrinsics: Camera intrinsic matrix (3, 3)
        target_resolution: Target resolution as (width, height)
        force: Whether to force rescaling even if downscaling not needed

    Returns:
        tuple: (resized_image, resized_depthmap, adjusted_intrinsics)
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)  # convert to pil image

    input_resolution = np.array(image.size)  # (W, H)
    target_resolution = np.array(target_resolution)  # (W, H)

    scale_factor = max(target_resolution / input_resolution) + 1e-8  # scaling factor

    if scale_factor >= 1 and not force:
        return image, depthmap, intrinsics

    output_resolution = np.floor(input_resolution * scale_factor).astype(int)  # (W, H)

    try:
        resized_image = image.resize(
            tuple(output_resolution),
            resample=(
                Image.Resampling.LANCZOS
                if scale_factor < 1
                else Image.Resampling.BICUBIC
            ),
        )  # (new_W, new_H)
    except AttributeError:
        resized_image = image.resize(
            tuple(output_resolution),
            resample=(
                Image.Resampling.LANCZOS
                if scale_factor < 1
                else Image.Resampling.BICUBIC
            ),
        )  # (new_W, new_H)

    if depthmap is not None:
        resized_depthmap = cv2.resize(
            depthmap, tuple(output_resolution), interpolation=cv2.INTER_NEAREST
        )  # (new_H, new_W)
    else:
        resized_depthmap = None

    adjusted_intrinsics = compute_camera_matrix_for_crop(
        intrinsics, input_resolution, output_resolution, scaling=scale_factor
    )  # (3, 3)

    return resized_image, resized_depthmap, adjusted_intrinsics


def compute_camera_matrix_for_crop(
    intrinsics,
    input_resolution,
    output_resolution,
    scaling=1,
    offset_factor=0.5,
    offset=None,
):
    """
    Compute adjusted camera matrix for a crop/resize operation.

    Args:
        intrinsics: Camera intrinsic matrix (3, 3)
        input_resolution: Input image resolution (width, height)
        output_resolution: Output image resolution (width, height)
        scaling: Scaling factor for focal length
        offset_factor: Factor for determining crop offset (0.5 = center)
        offset: Explicit offset (if None, calculated from offset_factor)

    Returns:
        numpy.ndarray: Adjusted camera intrinsic matrix (3, 3)
    """
    input_resolution = np.asarray(input_resolution)  # (W, H)
    output_resolution = np.asarray(output_resolution)  # (W, H)

    margins = np.asarray(input_resolution) * scaling - output_resolution  # (W, H)
    assert np.all(
        margins >= 0.0
    ), "Output resolution cannot be larger than scaled input"

    if offset is None:
        offset = offset_factor * margins  # (W, H)

    adjusted_intrinsics = intrinsics.copy()  # (3, 3)
    adjusted_intrinsics[0, 0] *= scaling  # scale fx
    adjusted_intrinsics[1, 1] *= scaling  # scale fy
    adjusted_intrinsics[0, 2] = (
        adjusted_intrinsics[0, 2] * scaling - offset[0]
    )  # adjust cx
    adjusted_intrinsics[1, 2] = (
        adjusted_intrinsics[1, 2] * scaling - offset[1]
    )  # adjust cy

    return adjusted_intrinsics


def crop_resize_if_necessary(image, depthmap, intrinsics, output_resolution):
    """
    Apply appropriate cropping and resizing to maintain geometric consistency.

    Args:
        image: RGB image (PIL Image or numpy array)
        depthmap: Depth map array (H, W)
        intrinsics: Camera intrinsic matrix (3, 3)
        output_resolution: Target output resolution (width, height)

    Returns:
        tuple: (processed_image, processed_depthmap, adjusted_intrinsics)
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)  # convert to pil image

    W, H = image.size  # (W, H)
    cx, cy = intrinsics[:2, 2].round().astype(int)  # (cx, cy)

    min_margin_x = min(cx, W - cx)  # horizontal margin
    min_margin_y = min(cy, H - cy)  # vertical margin

    l, t = cx - min_margin_x, cy - min_margin_y  # left, top
    r, b = cx + min_margin_x, cy + min_margin_y  # right, bottom
    crop_bbox = (l, t, r, b)  # (l, t, r, b)

    # print(f"depthmap: {depthmap}")
    # import loaders.utils.viz as viz
    # viz.visualize_dm(depthmap, image=image, cam=(intrinsics, None), name="before_first_crop")

    image, depthmap, intrinsics = crop_image_depthmap(
        image, depthmap, intrinsics, crop_bbox
    )  # initial crop

    image, depthmap, intrinsics = rescale_image_depthmap(
        image, depthmap, intrinsics, output_resolution
    )  # resize to target

    final_intrinsics = compute_camera_matrix_for_crop(
        intrinsics, image.size, output_resolution
    )  # (3, 3)

    W, H = image.size  # (W, H)
    out_width, out_height = output_resolution  # (out_W, out_H)
    l, t = np.int32(np.round(intrinsics[:2, 2] - final_intrinsics[:2, 2]))  # (l, t)
    r, b = l + out_width, t + out_height  # (r, b)
    final_crop_bbox = (l, t, r, b)  # (l, t, r, b)

    image, depthmap, intrinsics = crop_image_depthmap(
        image, depthmap, intrinsics, final_crop_bbox
    )  # final crop

    # viz.visualize_dm(depthmap, image=image, cam=(intrinsics, None), name="after_final_image_crop")

    return image, depthmap, intrinsics


def bbox_from_intrinsics_in_out(input_intrinsics, output_intrinsics, output_resolution):
    """
    Compute crop bounding box from input and output intrinsics.

    Args:
        input_intrinsics: Input camera intrinsic matrix (3, 3)
        output_intrinsics: Output camera intrinsic matrix (3, 3)
        output_resolution: Output resolution (width, height)

    Returns:
        tuple: Crop bounding box (left, top, right, bottom)
    """
    out_width, out_height = output_resolution  # (W, H)
    l, t = np.int32(
        np.round(input_intrinsics[:2, 2] - output_intrinsics[:2, 2])
    )  # (l, t)
    crop_bbox = (l, t, l + out_width, t + out_height)  # (l, t, r, b)

    return crop_bbox


def normalize_pointcloud(pts1, pts2, norm_mode='avg_dis', valid1=None, valid2=None, ret_factor=False):
    """ renorm pointmaps pts1, pts2 with norm_mode
    """
    assert pts1.ndim >= 3 and pts1.shape[-1] == 3
    assert pts2 is None or (pts2.ndim >= 3 and pts2.shape[-1] == 3)
    norm_mode, dis_mode = norm_mode.split('_')

    if norm_mode == 'avg':
        # gather all points together (joint normalization)
        nan_pts1, nnz1 = invalid_to_zeros(pts1, valid1, ndim=3)
        nan_pts2, nnz2 = invalid_to_zeros(pts2, valid2, ndim=3) if pts2 is not None else (None, 0)
        all_pts = torch.cat((nan_pts1, nan_pts2), dim=1) if pts2 is not None else nan_pts1

        # compute distance to origin
        all_dis = all_pts.norm(dim=-1)
        if dis_mode == 'dis':
            pass  # do nothing
        elif dis_mode == 'log1p':
            all_dis = torch.log1p(all_dis)
        elif dis_mode == 'warp-log1p':
            # actually warp input points before normalizing them
            log_dis = torch.log1p(all_dis)
            warp_factor = log_dis / all_dis.clip(min=1e-8)
            H1, W1 = pts1.shape[1:-1]
            pts1 = pts1 * warp_factor[:, :W1 * H1].view(-1, H1, W1, 1)
            if pts2 is not None:
                H2, W2 = pts2.shape[1:-1]
                pts2 = pts2 * warp_factor[:, W1 * H1:].view(-1, H2, W2, 1)
            all_dis = log_dis  # this is their true distance afterwards
        else:
            raise ValueError(f'bad {dis_mode=}')

        norm_factor = all_dis.sum(dim=1) / (nnz1 + nnz2 + 1e-8)
    else:
        # gather all points together (joint normalization)
        nan_pts1 = invalid_to_nans(pts1, valid1, ndim=3)
        nan_pts2 = invalid_to_nans(pts2, valid2, ndim=3) if pts2 is not None else None
        all_pts = torch.cat((nan_pts1, nan_pts2), dim=1) if pts2 is not None else nan_pts1

        # compute distance to origin
        all_dis = all_pts.norm(dim=-1)

        if norm_mode == 'avg':
            norm_factor = all_dis.nanmean(dim=1)
        elif norm_mode == 'median':
            norm_factor = all_dis.nanmedian(dim=1).values.detach()
        elif norm_mode == 'sqrt':
            norm_factor = all_dis.sqrt().nanmean(dim=1)**2
        else:
            raise ValueError(f'bad {norm_mode=}')

    norm_factor = norm_factor.clip(min=1e-8)
    while norm_factor.ndim < pts1.ndim:
        norm_factor.unsqueeze_(-1)

    res = pts1 / norm_factor
    if pts2 is not None:
        res = (res, pts2 / norm_factor)
    if ret_factor:
        res = res + (norm_factor,)
    return res
