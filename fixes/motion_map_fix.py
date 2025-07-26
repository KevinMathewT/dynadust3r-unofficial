"""
Fix for motion map computation in stereo_motion_base.py

The main issue is that get_motion_map_from_world_pc computes motion from 
each frame to the LAST frame, but we need specific directional motions.
"""

import numpy as np
import loaders.utils.geometry as geo

def compute_motion_maps_correctly(left_info, mid_info, right_info, reference_cam):
    """
    Compute motion maps correctly for all required directions.
    All motions are expressed in the left camera (reference) frame.
    
    Returns:
        dict: Motion maps with keys 'l2m', 'r2m', 'l2r', 'r2l'
    """
    motion_gt = {}
    
    # Helper function to compute motion between two specific frames
    def compute_pairwise_motion(world_pc_src, world_pc_dst, cam_src, cam_dst, 
                                 reference_cam, image_shape):
        """
        Compute motion from src to dst, expressed in reference frame.
        """
        # Get validity masks
        valid_src = world_pc_src[:, 3] > 0
        valid_dst = world_pc_dst[:, 3] > 0
        valid_both = valid_src & valid_dst
        
        if not np.any(valid_both):
            # Return empty motion map if no valid correspondences
            return np.zeros((*image_shape, 4), dtype=np.float32)
        
        # Extract valid world points
        world_src = world_pc_src[valid_both, :3]
        world_dst = world_pc_dst[valid_both, :3]
        
        # Transform to reference camera coordinates
        cam_ref_src = geo.world_pc_to_cam_pc(world_src, reference_cam)
        cam_ref_dst = geo.world_pc_to_cam_pc(world_dst, reference_cam)
        
        # Compute motion in reference frame
        motion_3d = cam_ref_dst - cam_ref_src
        
        # Project source points to source camera image plane
        cam_src_pts = geo.world_pc_to_cam_pc(world_src, cam_src)
        
        # Get intrinsics for projection
        intrinsics_src = cam_src[0]
        fx, fy = intrinsics_src[0, 0], intrinsics_src[1, 1]
        cx, cy = intrinsics_src[0, 2], intrinsics_src[1, 2]
        
        # Project to image coordinates
        x, y, z = cam_src_pts[:, 0], cam_src_pts[:, 1], cam_src_pts[:, 2]
        
        # Filter points behind camera
        z_valid = z > 0
        if not np.any(z_valid):
            return np.zeros((*image_shape, 4), dtype=np.float32)
        
        x, y, z = x[z_valid], y[z_valid], z[z_valid]
        motion_3d = motion_3d[z_valid]
        
        # Compute pixel coordinates
        u = np.round(fx * x / z + cx).astype(int)
        v = np.round(fy * y / z + cy).astype(int)
        
        # Filter points outside image bounds
        H, W = image_shape
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(in_bounds):
            return np.zeros((*image_shape, 4), dtype=np.float32)
        
        u, v = u[in_bounds], v[in_bounds]
        motion_valid = motion_3d[in_bounds]
        
        # Create motion map
        motion_map = np.zeros((H, W, 4), dtype=np.float32)
        motion_map[v, u, :3] = motion_valid
        motion_map[v, u, 3] = 1
        
        return motion_map
    
    # Compute all required motion maps
    # l2m: left to mid
    motion_gt["l2m"] = compute_pairwise_motion(
        left_info["world_pc_valid"], mid_info["world_pc_valid"],
        left_info["cam"], mid_info["cam"],
        reference_cam, left_info["image"].shape[:2]
    )
    
    # r2m: right to mid (project in right image space)
    motion_gt["r2m"] = compute_pairwise_motion(
        right_info["world_pc_valid"], mid_info["world_pc_valid"],
        right_info["cam"], mid_info["cam"],
        reference_cam, right_info["image"].shape[:2]
    )
    
    # l2r: left to right
    motion_gt["l2r"] = compute_pairwise_motion(
        left_info["world_pc_valid"], right_info["world_pc_valid"],
        left_info["cam"], right_info["cam"],
        reference_cam, left_info["image"].shape[:2]
    )
    
    # r2l: right to left (project in right image space)
    motion_gt["r2l"] = compute_pairwise_motion(
        right_info["world_pc_valid"], left_info["world_pc_valid"],
        right_info["cam"], left_info["cam"],
        reference_cam, right_info["image"].shape[:2]
    )
    
    return motion_gt


# Alternative: Fix the existing function calls
def fix_motion_computation(left_info, mid_info, right_info, reference_cam):
    """
    Fix the motion computation using the existing get_motion_map_from_world_pc
    by passing frames in the correct order.
    """
    motion_gt = {}
    
    # l2m: left to mid (correct as is)
    motion_maps_0 = geo.get_motion_map_from_world_pc(
        [left_info["world_pc_valid"], mid_info["world_pc_valid"]],
        [left_info["cam"], mid_info["cam"]],
        left_info["image"].shape[:2],
    )
    motion_gt["l2m"] = motion_maps_0[0]
    
    # r2m: right to mid (correct as is)
    motion_maps_1 = geo.get_motion_map_from_world_pc(
        [left_info["world_pc_valid"], right_info["world_pc_valid"], mid_info["world_pc_valid"]],
        [left_info["cam"], right_info["cam"], mid_info["cam"]],
        right_info["image"].shape[:2],
    )
    motion_gt["r2m"] = motion_maps_1[1]
    
    # l2r: left to right (correct as is)
    motion_maps_2 = geo.get_motion_map_from_world_pc(
        [left_info["world_pc_valid"], right_info["world_pc_valid"]],
        [left_info["cam"], right_info["cam"]],
        left_info["image"].shape[:2],
    )
    motion_gt["l2r"] = motion_maps_2[0]
    
    # r2l: right to left (FIXED - was incorrectly duplicating left frame)
    # We need to pass [left, right] to get motion from right to left
    motion_maps_3 = geo.get_motion_map_from_world_pc(
        [right_info["world_pc_valid"], left_info["world_pc_valid"]],  # Note: swapped order
        [right_info["cam"], left_info["cam"]],
        right_info["image"].shape[:2],
    )
    motion_gt["r2l"] = motion_maps_3[0]  # Motion from right (index 0) to left (last frame)
    
    return motion_gt