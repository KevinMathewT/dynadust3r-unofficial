"""
Author: Kevin Mathew T
Date: 2025-03-10
"""

import rerun as rr
import numpy as np
from scipy.spatial.transform import Rotation

from PIL import Image

import utils.geometry as geo
from utils.geometry import decompose_extrinsics

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

    rr.init("Camera_Movement")
    rr.connect_tcp("127.0.0.1:9876")

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
    rr.connect_tcp("127.0.0.1:9876")

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

    rr.init(name)
    rr.connect_tcp("127.0.0.1:9876")
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


def visualize_dm(dm, cam=None, image=None, name="dm"):
    """
    Visualizes a depth map.

    Args:
        dm (numpy.ndarray): (H, W) - Depth values.
        cam (tuple): (intrinsics, extrinsics) - Camera parameters (optional)
        image (numpy.ndarray): (H, W, 3) - RGB image for colorization (optional).
        name (str): Name for the visualization.
    """
    rr.init(name, spawn=True)
    rr.connect_tcp("127.0.0.1:9876")

    if cam is not None:
        intr, _ = cam
        H, W = dm.shape
        fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
        rr.log(
            f"{name}/cam",
            rr.Pinhole(
                width=W,
                height=H,
                focal_length=(fx, fy),
                principal_point=(cx, cy),
            ),
        )
        rr.log(f"{name}/cam/depth", rr.DepthImage(dm, meter=1.0, colormap="turbo"))
        if image is not None:
            rr.log(f"{name}/cam/image", rr.Image(image))
    else:
        rr.log(f"{name}/depth", rr.DepthImage(dm, meter=1.0, colormap="turbo"))
        if image is not None:
            rr.log(f"{name}/image", rr.Image(image))


def visualize_image(image, cam=None, name="image"):
    """
    Visualizes an RGB image.
    Args:
        image (numpy.ndarray): (H, W, 3) - RGB image.
        cam (tuple): (intrinsics, extrinsics) - Camera parameters (optional)
        name (str): Name for the visualization.
    """

    rr.init(name)
    rr.connect_tcp("127.0.0.1:9876")
    rr.log(name, rr.Image(image))
    intrinsics, _ = cam if cam is not None else (None, None)
    if intrinsics is not None:
        H, W = image.shape
        rr.log("cam/im", rr.Pinhole(
            resolution=[W, H],
            focal_length=[intrinsics[0, 0], intrinsics[1, 1]],
            principal_point=[intrinsics[0, 2], intrinsics[1, 2]]
        ))


def visualize_pm(pm, image=None, cam=None, valid=True, name="pm", pc_in_cam_coords=True, colors=None, save=False, path=None):
    """
    Visualizes a point map using rerun for 3D logging and visualization.

    Args:
        pm (numpy.ndarray or list of numpy.ndarray): Either a single (H, W, C) point map or a sequence of T point maps
            with shape (T, H, W, C). If only one time step is provided, it will be unsqueezed to shape (1, H, W, C).
        image (numpy.ndarray or list of numpy.ndarray, optional): Corresponding RGB image(s) per timestep, either
            a single (H, W, 3) array or list of T arrays. Defaults to None.
        cam (tuple or list of tuples, optional): Either a single (intrinsics, extrinsics) tuple or a list of T such tuples.
            intrinsics: 3x3 camera intrinsic matrix.
            extrinsics: 4x4 transformation matrix (world-to-camera).
            Defaults to None.
        valid (bool, optional): If True, filters out invalid points using the validity column if present. Defaults to True.
        name (str, optional): The name for logging in rerun. Defaults to "pm".
        pc_in_cam_coords (bool, optional): If True, assumes the point cloud is already in camera coordinates.
            If False, transforms world coordinates to camera coordinates using extrinsics. Defaults to True.
        colors (numpy.ndarray or list of numpy.ndarray, optional): Either a single (N,3) color array or a list of T such arrays.
            Optional array to use directly as point colors. Defaults to None.
        save (bool, optional): If True, saves the rerun recording to `path`. Defaults to False.
        path (str, optional): File path to save the .rrd file if `save` is True. Defaults to None.

    Behavior:
        - If `pm` contains a validity column (4th channel), filters points based on the `valid` flag.
        - Supports multiple timesteps: logs each timestep with `rr.set_time_sequence("time", t)`.
        - If only one timestep is provided, it is unsqueezed.
        - If `colors` is provided, uses that for coloring.
        - Else if `image` and `cam` are provided, projects the point cloud into image space for colorization.
        - Logs the point cloud and camera information to rerun for visualization.
        - If `pc_in_cam_coords` is False, transforms the point cloud using extrinsics.
    """
    rr.init(name)
    if save and path is not None:
        rr.save(path)  # must be called before the first log
        print(f"Saved visualization to {path}.")
    rr.connect_tcp("127.0.0.1:9876")

    # normalize pm to array with time dim
    if isinstance(pm, list):
        pm = np.stack(pm, axis=0)  # (T, H, W, C)
    elif isinstance(pm, np.ndarray) and pm.ndim == 3:
        pm = pm[np.newaxis, ...]   # (1, H, W, C)

    T = pm.shape[0]

    # helper to wrap scalar or list into length-T list
    def wrap(var):
        if var is None:
            return [None] * T
        if isinstance(var, list):
            return var
        return [var] * T

    image_seq = wrap(image)
    cam_seq   = wrap(cam)
    color_seq = wrap(colors)

    for t in range(T):
        rr.set_time_sequence("time", t)

        pm_t      = pm[t]
        image_t   = image_seq[t]
        cam_t     = cam_seq[t]
        colors_t  = color_seq[t]

        if cam_t is not None:
            intrinsics, extrinsics = cam_t

        H, W, C = pm_t.shape
        pc_valid = pm_t.reshape(-1, C)

        if pc_valid.shape[1] == 4:
            if valid:
                pc_valid = pc_valid[pc_valid[:, 3] > 0][:, :3]
            else:
                pc_valid = pc_valid[:, :3]

        if colors_t is not None:
            colors = colors_t
        elif image_t is not None and cam_t is not None:
            intrinsics, extrinsics = cam_t

            if not pc_in_cam_coords:
                pc_h = np.hstack((pc_valid, np.ones((pc_valid.shape[0], 1))))
                cam_coords = (extrinsics @ pc_h.T).T[:, :3]
            else:
                cam_coords = pc_valid

            uv = (intrinsics @ cam_coords.T).T
            uv /= uv[:, 2:3]
            uv = uv[:, :2].astype(int)

            h, w, _ = image_t.shape
            mask = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            colors = np.zeros((pc_valid.shape[0], 3), dtype=np.uint8)
            colors[mask] = image_t[uv[mask, 1], uv[mask, 0]]
        else:
            colors = None

        rr.log(name, rr.Points3D(positions=pc_valid, colors=colors))

        if image_t is not None and cam_t is not None:
            H_img, W_img, _ = image_t.shape
            rr.log("cam", rr.Pinhole(
                resolution=[W_img, H_img],
                focal_length=[intrinsics[0, 0], intrinsics[1, 1]],
                principal_point=[intrinsics[0, 2], intrinsics[1, 2]]
            ))

            if not pc_in_cam_coords:
                R = extrinsics[:3, :3]
                t_vec = extrinsics[:3, 3]
                R_inv = R.T
                t_inv = -R_inv @ t_vec
                inv_extrinsics = np.eye(4)
                inv_extrinsics[:3, :3] = R_inv
                inv_extrinsics[:3, 3] = t_inv
                translation, quaternion = decompose_extrinsics(inv_extrinsics)
                rr.log("cam_pose", rr.Transform3D(translation=translation, quaternion=quaternion))

            rr.log("cam", rr.Image(image_t))
    


def visualize_sequence_from_pms(pms, motion_map, image_seq=None, name="seq_pm", save=False, path=None):
    """
    Visualizes a sequence of point maps with motion vectors, logging frames at separate timesteps in Rerun.

    Parameters:
    - pms: List or array of (H, W, 4) or (H, W, 3) point maps or (N, 4) or (N, 3) point clouds (3D positions [+ optional validity mask]).
    - motion_map: (T-1, H, W, 4) or (T-1, H, W, 3) motion maps or (T-1, N, 4) or (T-1, N, 3) motion vectors.
    - image_seq: Optional list of (H, W, 3) images for color visualization.
    - name: String identifier for the visualization session.
    - save: Boolean flag to save the recording to a .rrd file.
    - path: File path to save the .rrd file if save is True.
    """
    def to_np(x):
        if hasattr(x, 'detach'):  # likely a torch tensor
            return x.detach().cpu().numpy()
        return x
    
    def ensure_4d(x):
        if x.shape[-1] == 3:
            if x.ndim == 3:
                x = np.concatenate([x, np.ones((x.shape[0], x.shape[1], 1), dtype=x.dtype)], axis=-1)
                x = x.reshape(-1, 4)
            else:
                x = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=-1)
            return x
        return x.reshape(-1, 4) if x.ndim == 3 else x

    rr.init(name)
    if save and path is not None:
        rr.save(path)  # must be called before the first log
        print(f"Saved visualization to {path}.")
    rr.connect_tcp("127.0.0.1:9876")
    
    pms = [to_np(pm) for pm in pms]
    motion_map = to_np(motion_map)
    
    T = len(pms)
    assert motion_map.shape[0] == T - 1

    if image_seq is None:
        image_seq = [None] * T

    pc = ensure_4d(pms[0])
    valid_mask = pc[:, 3] > 0
    pc_valid = pc[valid_mask][:, :3]

    img_flat = image_seq[0].reshape(-1, 3) if image_seq[0] is not None else None
    # print(f"Frame {0}: pm shape = {pc.shape}")
    # print(f"Frame {0}: valid_mask shape = {valid_mask.shape}, count = {valid_mask.sum()}")
    # if img_flat is not None:
    #     print(f"Frame {0}: image_seq[0] shape = {image_seq[0].shape}, reshaped shape = {img_flat.shape}")

    colors = (img_flat[valid_mask].astype(float) / 255.0
        if image_seq[0] is not None
        else np.array([[255, 95, 31]] * len(pc_valid), dtype=np.uint8) / 255.0)

    for t in range(T):
        rr.set_time_sequence("time", 2 * t)
        rr.log("point_cloud", rr.Points3D(positions=pc_valid, colors=colors))

        if t < T - 1:
            motion = ensure_4d(motion_map[t])
            motion_valid_mask = valid_mask & (motion[:, 3] > 0)

            src_pts = pc[motion_valid_mask][:, :3]
            dst_pts = src_pts + motion[motion_valid_mask][:, :3]
            lines = [np.stack([src_pts[i], dst_pts[i]]) for i in range(len(src_pts))]

            rr.log("motion_vectors", rr.LineStrips3D(strips=lines, colors=[57, 255, 20], radii=0.001))
            rr.set_time_sequence("time", 2 * t + 1)
            rr.log("motion_vectors", rr.LineStrips3D(strips=lines, colors=[57, 255, 20], radii=0.001))

            pc_next = ensure_4d(pms[t + 1])
            valid_mask = pc_next[:, 3] > 0
            pc_valid = pc_next[valid_mask][:, :3]
            
            colors = (image_seq[t + 1].reshape(-1, 3)[valid_mask].astype(float) / 255.0
                if image_seq[t + 1] is not None
                else np.array([[255, 95, 31]] * len(lines), dtype=np.uint8) / 255.0)

            rr.log("point_cloud", rr.Points3D(positions=pc_valid, colors=colors))
            pc = pc_next


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

# port forwarding: ssh -J greene -L 9091:localhost:9091 -L 9877:localhost:9877 gr011
# rerun viewer: rerun --web-viewer
def test_rerun():
    rr.init("streaming_test")
    rr.connect_tcp("127.0.0.1:9876")

    rr.set_time_sequence("frame", 0)

    pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    rr.log("scene/points", rr.Points3D(positions=pts))

    rr.disconnect()


if __name__ == "__main__":
    test_rerun()
