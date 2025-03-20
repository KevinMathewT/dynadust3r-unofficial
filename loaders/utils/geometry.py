"""
Author: Kevin Mathew T
Date: 2025-03-10
"""

import numpy as np
from scipy.spatial.transform import Rotation


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

    mask = z > 0 # mask for negative/zero z
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
    R = extrinsics[:3, :3] # (3, 3)
    t = extrinsics[:3, 3] # (3,)
    world_pc = (R.T @ (cam_pc - t).T).T # (N, 3)
    return world_pc # (N, 3)

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
    u = (x * fx / z + cx).astype(int)
    v = (y * fy / z + cy).astype(int)

    cam_pm = np.zeros((height, width, 4), dtype=cam_pc.dtype)

    # negative z
    validity_mask = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)

    if valid and cam_pc.shape[1] == 4:
        validity_mask &= cam_pc[:, 3] > 0

    cam_pm[v[validity_mask], u[validity_mask], :3] = cam_pc[validity_mask, :3]
    cam_pm[v[validity_mask], u[validity_mask], 3] = 1  # mark valid pixels

    return cam_pm


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

    scale_ratios = np.linalg.norm(actual_points, axis=1) / np.linalg.norm(approx_points, axis=1)
    print(f"Scale Ratios: {scale_ratios}")
    return np.median(scale_ratios)


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

        x, y, z = cam_pc_ref_valid[:, 0], cam_pc_ref_valid[:, 1], cam_pc_ref_valid[:, 2]  # (M,)
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
