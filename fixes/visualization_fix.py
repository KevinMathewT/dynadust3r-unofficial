"""
Fix for visualization issues in utils/viz.py

Issues:
1. Point clouds may have points at origin due to invalid masking
2. Coloring is incorrect - using wrong coordinate frames or projections
3. GT point clouds look wrong
"""

import numpy as np
import torch

def check_for_origin_points(point_cloud, valid_mask, name=""):
    """
    Debug function to check for points being placed at origin.
    """
    # Check if there are any exact origin points
    origin_mask = np.all(point_cloud == 0, axis=-1)
    num_origin = np.sum(origin_mask & valid_mask)
    
    if num_origin > 0:
        print(f"WARNING: {name} has {num_origin} valid points at origin!")
        
    # Check for near-origin points
    distances = np.linalg.norm(point_cloud, axis=-1)
    near_origin_mask = (distances < 0.01) & valid_mask
    num_near_origin = np.sum(near_origin_mask)
    
    if num_near_origin > 0:
        print(f"WARNING: {name} has {num_near_origin} valid points within 0.01 units of origin!")
    
    return origin_mask, near_origin_mask


def fix_point_cloud_visualization(pc, valid_mask, name=""):
    """
    Fix point cloud by removing invalid points that might be at origin.
    """
    # Create a copy to avoid modifying original
    pc_fixed = pc.copy()
    
    # Set invalid points to NaN instead of 0
    pc_fixed[~valid_mask] = np.nan
    
    # Check for and report origin points
    origin_mask, near_origin_mask = check_for_origin_points(pc, valid_mask, name)
    
    # Additional check: remove points with all zero coordinates even if marked valid
    zero_points = np.all(pc == 0, axis=-1)
    if np.any(zero_points & valid_mask):
        print(f"Removing {np.sum(zero_points & valid_mask)} zero points from {name}")
        valid_mask = valid_mask & ~zero_points
    
    return pc_fixed, valid_mask


def debug_motion_visualization(motion_map, base_pc, name=""):
    """
    Debug motion map to find why motion lines are noisy.
    """
    valid_motion = motion_map[..., 3] > 0
    motion_vectors = motion_map[..., :3]
    
    if np.any(valid_motion):
        # Compute motion magnitudes
        motion_magnitudes = np.linalg.norm(motion_vectors, axis=-1)
        valid_magnitudes = motion_magnitudes[valid_motion]
        
        print(f"\n{name} Motion Statistics:")
        print(f"  Valid motions: {np.sum(valid_motion)}")
        print(f"  Min magnitude: {np.min(valid_magnitudes):.6f}")
        print(f"  Max magnitude: {np.max(valid_magnitudes):.6f}")
        print(f"  Mean magnitude: {np.mean(valid_magnitudes):.6f}")
        print(f"  Std magnitude: {np.std(valid_magnitudes):.6f}")
        
        # Check for outliers (motion > 10 units)
        large_motions = motion_magnitudes > 10.0
        if np.any(large_motions & valid_motion):
            print(f"  WARNING: {np.sum(large_motions & valid_motion)} motions > 10 units!")
            
        # Check if motion is mostly in one direction
        mean_motion = np.mean(motion_vectors[valid_motion], axis=0)
        print(f"  Mean motion vector: {mean_motion}")


def improved_visualize_sequence_from_pms(pms, motion_map, image_seq=None, name="seq_pm", save=False, path=None):
    """
    Improved visualization that properly handles point cloud coloring and validity.
    """
    import rerun as rr
    
    def to_np(x):
        if hasattr(x, 'detach'):  # likely a torch tensor
            return x.detach().cpu().numpy()
        return x
    
    def ensure_4d(x):
        """Ensure point cloud has 4 dimensions (x, y, z, validity)"""
        if x.shape[-1] == 3:
            # Add validity channel if missing
            if x.ndim == 3:  # (H, W, 3)
                validity = np.ones((x.shape[0], x.shape[1], 1), dtype=x.dtype)
                x = np.concatenate([x, validity], axis=-1)
                x = x.reshape(-1, 4)
            else:  # (N, 3)
                validity = np.ones((x.shape[0], 1), dtype=x.dtype)
                x = np.concatenate([x, validity], axis=-1)
            return x
        return x.reshape(-1, 4) if x.ndim == 3 else x

    rr.init(name)
    if save and path is not None:
        rr.save(path)
        print(f"Saved visualization to {path}.")
    rr.connect_tcp("127.0.0.1:9876")
    
    pms = [to_np(pm) for pm in pms]
    motion_map = to_np(motion_map)
    
    T = len(pms)
    assert motion_map.shape[0] == T - 1
    
    if image_seq is None:
        image_seq = [None] * T
    
    # Process first frame
    pc = ensure_4d(pms[0])
    valid_mask = pc[:, 3] > 0
    
    # Debug: check for origin points
    pc_coords = pc[:, :3]
    origin_points = np.all(np.abs(pc_coords) < 1e-6, axis=1)
    if np.any(origin_points & valid_mask):
        print(f"Frame 0: Found {np.sum(origin_points & valid_mask)} valid points at origin!")
    
    pc_valid = pc[valid_mask][:, :3]
    
    # Improved coloring logic
    if image_seq[0] is not None:
        # Reshape image for proper indexing
        img_flat = image_seq[0].reshape(-1, 3)
        
        # Ensure we have the right number of pixels
        if len(img_flat) == len(pc):
            colors = img_flat[valid_mask].astype(float) / 255.0
        else:
            print(f"Warning: Image size mismatch. Using default colors.")
            colors = np.array([[255, 95, 31]] * len(pc_valid), dtype=np.uint8) / 255.0
    else:
        colors = np.array([[255, 95, 31]] * len(pc_valid), dtype=np.uint8) / 255.0
    
    # Visualize each timestep
    for t in range(T):
        rr.set_time_sequence("time", 2 * t)
        rr.log("point_cloud", rr.Points3D(positions=pc_valid, colors=colors))
        
        if t < T - 1:
            # Process motion
            motion = ensure_4d(motion_map[t])
            motion_valid_mask = valid_mask & (motion[:, 3] > 0)
            
            if np.any(motion_valid_mask):
                src_pts = pc[motion_valid_mask][:, :3]
                motion_vecs = motion[motion_valid_mask][:, :3]
                
                # Debug large motions
                motion_mags = np.linalg.norm(motion_vecs, axis=1)
                if np.any(motion_mags > 10.0):
                    print(f"Frame {t}: Found {np.sum(motion_mags > 10.0)} large motions (>10 units)")
                
                dst_pts = src_pts + motion_vecs
                lines = [np.stack([src_pts[i], dst_pts[i]]) for i in range(len(src_pts))]
                
                rr.log("motion_vectors", rr.LineStrips3D(strips=lines, colors=[57, 255, 20], radii=0.001))
                rr.set_time_sequence("time", 2 * t + 1)
                rr.log("motion_vectors", rr.LineStrips3D(strips=lines, colors=[57, 255, 20], radii=0.001))
            
            # Process next frame
            pc_next = ensure_4d(pms[t + 1])
            valid_mask = pc_next[:, 3] > 0
            pc_valid = pc_next[valid_mask][:, :3]
            
            # Color next frame
            if image_seq[t + 1] is not None:
                img_flat = image_seq[t + 1].reshape(-1, 3)
                if len(img_flat) == len(pc_next):
                    colors = img_flat[valid_mask].astype(float) / 255.0
                else:
                    colors = np.array([[255, 95, 31]] * len(pc_valid), dtype=np.uint8) / 255.0
            else:
                colors = np.array([[255, 95, 31]] * len(pc_valid), dtype=np.uint8) / 255.0
            
            rr.log("point_cloud", rr.Points3D(positions=pc_valid, colors=colors))
            pc = pc_next