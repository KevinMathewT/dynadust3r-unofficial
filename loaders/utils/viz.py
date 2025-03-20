"""
Author: Kevin Mathew T
Date: 2025-03-10
"""

import rerun as rr
import numpy as np
import loaders.utils.geometry as geo
from scipy.spatial.transform import Rotation

from PIL import Image

from loaders.utils.geometry import decompose_extrinsics

IDENTITY_EXTRINSIC = np.hstack((np.eye(3), np.zeros((3, 1))))


def show_grayscale_image(m):
    import matplotlib.pyplot as plt
    plt.imshow(m, cmap='gray')
    plt.axis('off')
    plt.show()


def visualize_cam_movement_in_world(dataset, seq_path, num_frames):
    """
    Visualizes the movement of a camera in the world coordinate system using rerun.

    Args:
        dataset: Object providing access to dataset frames.
                 Must implement `get_frame_info(seq_path, frame_idx)`, returning a dictionary 
                 with keys "cam" (intrinsics, extrinsics) and "image".
        seq_path (str): Path to the sequence of frames.
        num_frames (int): Number of frames to visualize.

    The function logs camera poses, intrinsic parameters, and images to rerun.
    It decomposes extrinsics into translation and quaternion representation, 
    and logs each frame's camera parameters under a "pinhole/{i}" namespace.
    """

    rr.init("Camera_Movement", spawn=True)
    rr.spawn()

    for i in range(num_frames):
        frame_info = dataset.get_frame_info(seq_path, i)
        intrinsics, extrinsics = frame_info["cam"]
        image = frame_info["image"]
        
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        img_height, img_width = image.shape[:2]
        translation, quaternion = decompose_extrinsics(extrinsics)
        rr.log(f"pinhole/{i}", rr.Transform3D(translation=translation, quaternion=quaternion))
        rr.log(f"pinhole/{i}", rr.Pinhole(
            focal_length=(fx, fy),
            principal_point=(cx, cy),
            resolution=(img_width, img_height)
        ))
        rr.log(f"pinhole/{i}", rr.Image(image))



def visualize_pc(pc_valid, image=None, cam=None, valid=True, name="point_cloud", pc_in_cam_coords=True):
    """
    Visualizes a 3D point cloud using rerun for logging and visualization.

    Args:
        pc_valid (numpy.ndarray): A (N, 3) or (N, 4) array representing 3D points.
            If the array has 4 columns, the last column is treated as a validity flag.
        image (numpy.ndarray, optional): Corresponding RGB image (H, W, 3) used for colorizing points. Defaults to None.
        cam (tuple, optional): A tuple containing (intrinsics, extrinsics) matrices.
            - intrinsics (numpy.ndarray): 3x3 camera intrinsic matrix.
            - extrinsics (numpy.ndarray): 4x4 transformation matrix (world-to-camera).
            Defaults to None.
        valid (bool, optional): If True, filters out invalid points using the validity column if present. Defaults to True.
        name (str, optional): The name for logging in rerun. Defaults to "point_cloud".
        pc_in_cam_coords (bool, optional): If True, assumes the point cloud is already in camera coordinates.
            If False, transforms world coordinates to camera coordinates using extrinsics. Defaults to True.

    Behavior:
        - If `pc_valid` contains a validity column (4th channel), filters points based on the `valid` flag.
        - If `image` and `cam` are provided, projects the point cloud into image space for colorization.
        - Logs the point cloud and camera information to rerun for visualization.
        - If `pc_in_cam_coords` is False, transforms the point cloud using extrinsics.

    Returns:
        None (logs visualization data to rerun).
    """
    
    rr.init(name)
    rr.spawn()

    if cam is not None:
        intrinsics, extrinsics = cam
    
    pc_valid = np.asarray(pc_valid)
    
    if pc_valid.shape[1] == 4:
        if valid:
            pc_valid = pc_valid[pc_valid[:, 3] > 0][:, :3]
        else:
            pc_valid = pc_valid[:, :3]
    
    if image is not None and cam is not None:
        intrinsics, extrinsics = cam
        
        if not pc_in_cam_coords:
            pc_h = np.hstack((pc_valid, np.ones((pc_valid.shape[0], 1))))
            cam_coords = (extrinsics @ pc_h.T).T[:, :3]
        else:
            cam_coords = pc_valid
        
        uv = (intrinsics @ cam_coords.T).T
        uv /= uv[:, 2:3]
        uv = uv[:, :2].astype(int)
        
        h, w, _ = image.shape
        mask = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        colors = np.zeros((pc_valid.shape[0], 3), dtype=np.uint8)
        colors[mask] = image[uv[mask, 1], uv[mask, 0]]
    else:
        colors = None

    rr.log(name, rr.Points3D(positions=pc_valid, colors=colors))

    if image is not None and intrinsics is not None:
        H, W, _ = image.shape
        rr.log("cam", rr.Pinhole(
            resolution=[W, H],
            focal_length=[intrinsics[0, 0], intrinsics[1, 1]],
            principal_point=[intrinsics[0, 2], intrinsics[1, 2]]
        ))
        
        if not pc_in_cam_coords:
            R = extrinsics[:3, :3]
            t = extrinsics[:3, 3]
            R_inv = R.T
            t_inv = -R_inv @ t
            inv_extrinsics = np.eye(4)
            inv_extrinsics[:3, :3] = R_inv
            inv_extrinsics[:3, 3] = t_inv
            translation, quaternion = decompose_extrinsics(inv_extrinsics)
            rr.log("cam_pose", rr.Transform3D(translation=translation, quaternion=quaternion))
        
        rr.log("cam", rr.Image(image))


def visualize_motion_map(motion_map, pm, cam=None, valid=True, name="motion_map"):
    """
    Visualizes motion vectors in 3D space.

    Args:
        motion_map (numpy.ndarray): (H, W, 4) - Motion vectors (dx, dy, dz, validity).
        pm (numpy.ndarray): (H, W, 4) - 3D points (X, Y, Z, validity).
        cam (tuple): (intrinsics, extrinsics) - Camera parameters (optional)
        valid (bool): Whether to apply the validity mask.
        name (str): Name for the visualization.
    """
    H, W, _ = motion_map.shape
    intrinsics, extrinsics = cam if cam is not None else (None, None)

    valid_mask = (motion_map[:, :, 3] > 0) & (pm[:, :, 3] > 0) if valid else np.ones((H, W), dtype=bool)
    y, x = np.where(valid_mask)
    motion_vectors = motion_map[y, x, :3]
    points_3d = pm[y, x, :3]

    if len(points_3d) == 0:
        return

    segments = np.stack([points_3d, points_3d + motion_vectors], axis=1)

    rr.init(name, spawn=True)
    rr.log(name, rr.SeriesLine(segments=segments))
    if extrinsics is not None:
        translation, quaternion = decompose_extrinsics(extrinsics)
        rr.log("cam", rr.Transform3D(translation=translation, quaternion=quaternion))
    if intrinsics is not None:
        rr.log("cam/im", rr.Pinhole(
            resolution=[W, H],
            focal_length=[intrinsics[0, 0], intrinsics[1, 1]],
            principal_point=[intrinsics[0, 2], intrinsics[1, 2]]
        ))

def visualize_dm(dm, cam=None, name="dm"):
    """
    Visualizes a depth map.

    Args:
        dm (numpy.ndarray): (H, W) - Depth values.
        cam (tuple): (intrinsics, extrinsics) - Camera parameters (optional)
        name (str): Name for the visualization.
    """
    rr.init(name, spawn=True)
    rr.log(name, rr.Image(dm))
    intrinsics, _ = cam if cam is not None else (None, None)
    if intrinsics is not None:
        H, W = dm.shape
        rr.log("cam/im", rr.Pinhole(
            resolution=[W, H],
            focal_length=[intrinsics[0, 0], intrinsics[1, 1]],
            principal_point=[intrinsics[0, 2], intrinsics[1, 2]]
        ))

def visualize_pm(pm, image=None, cam=None, valid=True, name="pm", pc_in_cam_coords=True):
    """
    Visualizes a point map using rerun for 3D logging and visualization.

    Args:
        pm (numpy.ndarray): A (H, W, C) point map where each pixel represents a 3D point.
        image (numpy.ndarray, optional): Corresponding RGB image (H, W, 3) used for colorizing points. Defaults to None.
        cam (tuple, optional): A tuple containing (intrinsics, extrinsics) matrices.
            - intrinsics (numpy.ndarray): 3x3 camera intrinsic matrix.
            - extrinsics (numpy.ndarray): 4x4 transformation matrix (world-to-camera).
            Defaults to None.
        valid (bool, optional): If True, filters out invalid points using the validity column if present. Defaults to True.
        name (str, optional): The name for logging in rerun. Defaults to "pm".
        pc_in_cam_coords (bool, optional): If True, assumes the point cloud is already in camera coordinates.
            If False, transforms world coordinates to camera coordinates using extrinsics. Defaults to True.

    Behavior:
        - If `pm` contains a validity column (4th channel), filters points based on the `valid` flag.
        - If `image` and `cam` are provided, projects the point cloud into image space for colorization.
        - Logs the point cloud and camera information to rerun for visualization.
        - If `pc_in_cam_coords` is False, transforms the point cloud using extrinsics.

    Returns:
        None (logs visualization data to rerun).
    """
    rr.init(name)
    rr.spawn()

    if cam is not None:
        intrinsics, extrinsics = cam
    
    H, W, C = pm.shape
    pc_valid = pm.reshape(-1, C)
    
    if pc_valid.shape[1] == 4:
        if valid:
            pc_valid = pc_valid[pc_valid[:, 3] > 0][:, :3]
        else:
            pc_valid = pc_valid[:, :3]
    
    if image is not None and cam is not None:
        intrinsics, extrinsics = cam
        
        if not pc_in_cam_coords:
            pc_h = np.hstack((pc_valid, np.ones((pc_valid.shape[0], 1))))
            cam_coords = (extrinsics @ pc_h.T).T[:, :3]
        else:
            cam_coords = pc_valid
        
        uv = (intrinsics @ cam_coords.T).T
        uv /= uv[:, 2:3]
        uv = uv[:, :2].astype(int)
        
        h, w, _ = image.shape
        mask = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        colors = np.zeros((pc_valid.shape[0], 3), dtype=np.uint8)
        colors[mask] = image[uv[mask, 1], uv[mask, 0]]
    else:
        colors = None
    
    rr.log(name, rr.Points3D(positions=pc_valid, colors=colors))

    if image is not None and intrinsics is not None:
        H, W, _ = image.shape
        rr.log("cam", rr.Pinhole(
            resolution=[W, H],
            focal_length=[intrinsics[0, 0], intrinsics[1, 1]],
            principal_point=[intrinsics[0, 2], intrinsics[1, 2]]
        ))
        
        if not pc_in_cam_coords:
            R = extrinsics[:3, :3]
            t = extrinsics[:3, 3]
            R_inv = R.T
            t_inv = -R_inv @ t
            inv_extrinsics = np.eye(4)
            inv_extrinsics[:3, :3] = R_inv
            inv_extrinsics[:3, 3] = t_inv
            translation, quaternion = decompose_extrinsics(inv_extrinsics)
            rr.log("cam_pose", rr.Transform3D(translation=translation, quaternion=quaternion))
        
        rr.log("cam", rr.Image(image))


def visualize_sequence_from_pms(pms, motion_map, image_seq=None, name="seq_pm"):
    """
    Visualizes a sequence of point maps with motion vectors, logging frames at separate timesteps in Rerun.

    Parameters:
    - pms: List or array of (H, W, 4) point maps (3D positions + validity mask).
    - motion_map: (T-1, H, W, 4) motion vectors between consecutive frames.
    - image_seq: Optional list of (H, W, 3) images for color visualization.
    - name: String identifier for the visualization session.

    Logging strategy:
    - `2t`: Source point cloud + motion vectors.
    - `2t+1`: Motion vectors + next frame's point cloud.
    """
    rr.init(name)
    rr.spawn()
    T = len(pms)
    assert motion_map.shape[0] == T - 1

    print(f"Frame {0}: pm shape = {pms[0].shape}")
    pm = pms[0].reshape(-1, 4)
    valid_mask = pm[:, 3] > 0
    pc_valid = pm[valid_mask][:, :3]

    img_flat = image_seq[0].reshape(-1, 3)
    print(f"Frame {0}: pm shape = {pm.shape}")
    print(f"Frame {0}: valid_mask shape = {valid_mask.shape}, count = {valid_mask.sum()}")
    print(f"Frame {0}: image_seq[0] shape = {image_seq[0].shape}, reshaped shape = {img_flat.shape}")

    colors = image_seq[0].reshape(-1, 3)[valid_mask] if image_seq is not None else None

    for t in range(T):
        rr.set_time_sequence("time", 2 * t)
        rr.log("point_cloud", rr.Points3D(positions=pc_valid, colors=colors))

        if t < T - 1:
            motion = motion_map[t].reshape(-1, 4)
            motion_valid_mask = valid_mask & (motion[:, 3] > 0)

            src_pts = pm[motion_valid_mask][:, :3]
            dst_pts = src_pts + motion[motion_valid_mask][:, :3]
            lines = [np.stack([src_pts[i], dst_pts[i]]) for i in range(len(src_pts))]

            rr.log("motion_vectors", rr.LineStrips3D(strips=lines))
            rr.set_time_sequence("time", 2 * t + 1)
            rr.log("motion_vectors", rr.LineStrips3D(strips=lines))

            pm = pms[t + 1].reshape(-1, 4)
            valid_mask = pm[:, 3] > 0
            pc_valid = pm[valid_mask][:, :3]
            colors = image_seq[t + 1].reshape(-1, 3)[valid_mask] if image_seq is not None else None

            rr.log("point_cloud", rr.Points3D(positions=pc_valid, colors=colors))

    print(f"Visualized sequence: {T} frames.")


def test_visualize_pc():
    test_points = np.array([
        [0, 0, 1],
        [1, 1, 2],
        [2, 0, 3],
        [1, -1, 4],
        [-1, 2, 5]
    ], dtype=np.float32)

    test_validity = np.ones((test_points.shape[0], 1), dtype=np.float32)

    test_pc_valid = np.hstack((test_points, test_validity))

    print(f"Calling visualize_pc with manually created data: {test_pc_valid.shape}")
    visualize_pc(test_pc_valid, valid=True, name="test_manual_pc")


def test_rerun():
    rr.init("test_rerun")
    rr.spawn()

    test_points = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [2, 0, 1],
        [1, -1, 2],
        [-1, 2, 0]
    ], dtype=np.float32)

    test_colors = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255]
    ], dtype=np.uint8)

    print("Logging test points to Rerun...")
    rr.log("test_points", rr.Points3D(positions=test_points, colors=test_colors))
    print("Logged successfully!")
    
    rr.disconnect()