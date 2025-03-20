"""
Author: Kevin Mathew T
Date: 2025-03-10
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from loaders.utils import geometry as geo

def get_scale(self, world_pc, dm, cam):
    cam_pc = geo.world_pc_to_cam_pc(world_pc, cam)
    cam_dm_pc = geo.dm_to_cam_pc(dm, cam)
    cam_pm = geo.cam_pc_to_cam_pm(cam_pc, cam, dm.shape)
    cam_dm_pm = geo.cam_pc_to_cam_pm(cam_dm_pc, cam, dm.shape)

    scale = geo.compute_scale_difference(cam_pm, cam_dm_pm)
    return scale

def compute_point_cloud_scale_stats(pc1, pc2):
    """
    Compute the mean, median, and standard deviation of distances for two point clouds.

    pc1: (N, 3) first point cloud (e.g., from depth map)
    pc2: (M, 3) second point cloud (e.g., from 3D tracks)

    Returns:
        scale_ratio: Mean distance ratio (depth_pc / track_pc)
        stats: Dictionary containing mean, median, std deviation of distances
    """
    d1 = np.linalg.norm(pc1, axis=1)
    d2 = np.linalg.norm(pc2, axis=1)

    mean_d1, mean_d2 = np.mean(d1), np.mean(d2)
    median_d1, median_d2 = np.median(d1), np.median(d2)
    std_d1, std_d2 = np.std(d1), np.std(d2)

    scale_ratio = mean_d1 / mean_d2

    stats = {
        "depth_map_pc": {"mean": mean_d1, "median": median_d1, "std": std_d1},
        "track_pc": {"mean": mean_d2, "median": median_d2, "std": std_d2},
        "scale_ratio": scale_ratio
    }

    return scale_ratio, stats, d1, d2

def plot_distance_distributions(d1, d2, name1="Depth Map PC", name2="Track PC"):
    """
    Plots the histogram of distances for two point clouds to visualize scale differences.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(d1, bins=50, alpha=0.5, label=name1)
    plt.hist(d2, bins=50, alpha=0.5, label=name2)
    plt.xlabel("Distance from Camera")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Distance Distribution of Point Clouds")
    plt.show()


def compute_optimal_scale(depth_map_pc, track_pc, scale_range=(0.01, 500), tolerance=1e-3):
    """
    Finds the optimal scale factor to minimize MSE between depth_map_pc and track_pc using binary search.

    Args:
        depth_map_pc (np.ndarray): (N, 3) Point cloud from the depth map.
        track_pc (np.ndarray): (M, 3) Point cloud from the 3D tracks.
        scale_range (tuple): The range of scales to search within (min, max).
        tolerance (float): Stop when scale difference is below this threshold.

    Returns:
        float: Optimal scale factor.
    """
    tree = cKDTree(track_pc)

    def mse_for_scale(scale):
        """Compute MSE between scaled depth points and nearest track points."""
        scaled_pc = depth_map_pc * scale
        _, nn_indices = tree.query(scaled_pc, k=1)
        matched_track_pc = track_pc[nn_indices]
        mse = np.mean(np.linalg.norm(scaled_pc - matched_track_pc, axis=1) ** 2)
        return mse

    low, high = scale_range
    best_scale, best_mse = low, float('inf')

    while high - low > tolerance:
        mid = (low + high) / 2
        mse_mid = mse_for_scale(mid)

        if mse_mid < best_mse:
            best_mse = mse_mid
            best_scale = mid

        mse_low = mse_for_scale(low)
        mse_high = mse_for_scale(high)

        if mse_low < mse_high:
            high = mid
        else:
            low = mid

    return best_scale

def compute_optimal_scale_inverse(t, d, tol=1e-3, max_iter=50, smin=0.5, smax=1.5):
    from scipy.spatial import cKDTree
    import numpy as np
    kd=cKDTree(d)
    def f(s): return np.mean((kd.query(s*t)[0])**2)
    for _ in range(max_iter):
        mid=(smin+smax)/2; eps=tol
        if f(mid-eps)<f(mid+eps): smax=mid
        else: smin=mid
        if smax-smin<tol: break
    return (smin+smax)/2  # float



