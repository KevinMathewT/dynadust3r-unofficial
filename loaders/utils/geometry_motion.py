"""
Motion computation utilities with clearer semantics.
"""
import numpy as np
import loaders.utils.geometry as geo


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