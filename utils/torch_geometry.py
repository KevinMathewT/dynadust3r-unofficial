"""
Author: Kevin Mathew T
Date: 2025-08-17
LinkedIn: https://www.linkedin.com/in/kevinmathewt/

Torch-based (GPU-friendly) counterparts to utils.geometry.
All functions here mirror utils/geometry.py in name and order, but operate on
torch.Tensors and are written to run efficiently on CUDA when tensors are on GPU.

Notes
- Inputs/outputs are torch tensors unless specified.
- Intrinsics/extrinsics are expected as torch tensors.
- Image-shaped outputs follow (..., H, W, C) like numpy version.
- For functions that require heavy host-side libraries (e.g., PIL, cv2), we
  leave NotImplementedError placeholders — these are not needed in the GPU path.
"""

import math
import torch


# ---------------------------------------------------------
# Motion map helpers (torch versions)
# ---------------------------------------------------------

def compute_motion_map_direct(
    source_world_pc: torch.Tensor,
    target_world_pc: torch.Tensor,
    source_cam: tuple,
    target_cam: tuple,
    reference_cam: tuple,
    image_shape: tuple[int, int],
) -> torch.Tensor:
    """
    Torch version of utils.geometry.compute_motion_map_direct.
    source_world_pc/target_world_pc: (N,4) world xyz + validity (0/1)
    Returns: (H, W, 4) torch.float32 on same device as inputs.
    """
    device = source_world_pc.device
    H, W = image_shape
    motion_map = torch.zeros((H, W, 4), dtype=source_world_pc.dtype, device=device)

    valid_mask = (source_world_pc[:, 3] > 0) & (target_world_pc[:, 3] > 0)
    if not torch.any(valid_mask):
        return motion_map

    source_world = source_world_pc[valid_mask, :3]
    target_world = target_world_pc[valid_mask, :3]

    source_ref = world_pc_to_cam_pc(source_world, reference_cam)
    target_ref = world_pc_to_cam_pc(target_world, reference_cam)
    motion_3d = target_ref - source_ref  # (M,3)

    source_cam_coords = world_pc_to_cam_pc(source_world, source_cam)  # (M,3)
    intrinsics = source_cam[0]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x, y, z = source_cam_coords[:, 0], source_cam_coords[:, 1], source_cam_coords[:, 2]
    z_pos = z > 0
    if not torch.any(z_pos):
        return motion_map

    x = x[z_pos]
    y = y[z_pos]
    z = z[z_pos]
    motion_3d = motion_3d[z_pos]

    u = torch.round(fx * x / z + cx).to(torch.int64)
    v = torch.round(fy * y / z + cy).to(torch.int64)

    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not torch.any(in_bounds):
        return motion_map

    u = u[in_bounds]
    v = v[in_bounds]
    motion_valid = motion_3d[in_bounds]

    motion_map[v, u, :3] = motion_valid
    motion_map[v, u, 3] = 1
    return motion_map


def compute_all_motion_maps(
    left_world_pc: torch.Tensor,
    mid_world_pc: torch.Tensor,
    right_world_pc: torch.Tensor,
    left_cam: tuple,
    mid_cam: tuple,
    right_cam: tuple,
    image_shape: tuple[int, int],
) -> dict:
    """
    Torch version of utils.geometry.compute_all_motion_maps.
    Returns dict with torch tensors: keys "l2m", "r2m", "l2r", "r2l".
    """
    reference_cam = left_cam
    return {
        "l2m": compute_motion_map_direct(left_world_pc, mid_world_pc, left_cam, mid_cam, reference_cam, image_shape),
        "r2m": compute_motion_map_direct(right_world_pc, mid_world_pc, right_cam, mid_cam, reference_cam, image_shape),
        "l2r": compute_motion_map_direct(left_world_pc, right_world_pc, left_cam, right_cam, reference_cam, image_shape),
        "r2l": compute_motion_map_direct(right_world_pc, left_world_pc, right_cam, left_cam, reference_cam, image_shape),
    }


# ---------------------------------------------------------
# Pose / transforms (torch versions)
# ---------------------------------------------------------

def inv(mat: torch.Tensor) -> torch.Tensor:
    """
    Torch version of utils.geometry.inv supporting (...,4,4) and (...,3,4).
    Always returns a 4x4 matrix when input is 3x4.
    """
    h, w = mat.shape[-2:]
    if h == 4 and w == 4:
        return torch.linalg.inv(mat)
    if h == 3 and w == 4:
        R, t = mat[:, :3], mat[:, 3]
        Ri = R.transpose(-2, -1)
        ti = -Ri @ t.unsqueeze(-1)
        E = torch.eye(4, dtype=mat.dtype, device=mat.device)
        E[:3, :3], E[:3, 3] = Ri, ti.squeeze(-1)
        return E
    raise ValueError(f"Can't invert matrix of shape {mat.shape}")


def recolor(pm: torch.Tensor, cam1: tuple, cam2: tuple, img2: torch.Tensor) -> torch.Tensor:
    """
    Torch version of utils.geometry.recolor.
    pm: (H, W, 4) in cam1 coords, img2: (H2, W2, 3) torch.uint8/float.
    Returns (H, W, 3) on same device as img2/pm.
    """
    intrinsics1, extrinsics1 = cam1
    intrinsics2, extrinsics2 = cam2
    device = pm.device
    H, W = pm.shape[:2]

    mask = pm[..., 3] > 0
    if not torch.any(mask):
        return torch.zeros((H, W, 3), dtype=img2.dtype, device=device)

    pts_cam1 = pm[mask, :3]
    R1, t1 = extrinsics1[:3, :3], extrinsics1[:3, 3]
    pts_world = (R1.transpose(-2, -1) @ (pts_cam1 - t1).T).T

    R2, t2 = extrinsics2[:3, :3], extrinsics2[:3, 3]
    pts_cam2 = (R2 @ pts_world.T).T + t2

    uvw = (intrinsics2 @ pts_cam2.T).T
    u = torch.round(uvw[:, 0] / uvw[:, 2]).to(torch.int64)
    v = torch.round(uvw[:, 1] / uvw[:, 2]).to(torch.int64)

    H2, W2 = img2.shape[:2]
    valid = (uvw[:, 2] > 0) & (u >= 0) & (u < W2) & (v >= 0) & (v < H2)

    out = torch.zeros((H * W, 3), dtype=img2.dtype, device=device)
    idx_flat = torch.nonzero(mask.reshape(-1), as_tuple=True)[0]
    choose_idx = idx_flat[valid]
    if choose_idx.numel() > 0:
        out[choose_idx] = img2[v[valid], u[valid]]
    return out.view(H, W, 3)


def decompose_extrinsics(extrinsics: torch.Tensor):
    """
    Torch version of utils.geometry.decompose_extrinsics.
    Returns (translation (3,), quaternion (4,) in (x,y,z,w)).
    """
    t = extrinsics[:3, 3]
    R = extrinsics[:3, :3]
    q = _rotation_matrix_to_quaternion(R)
    return t, q


def _rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 rotation matrix to quaternion (x,y,z,w)."""
    # Based on standard algorithms
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    trace = m00 + m11 + m22
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = torch.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = torch.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = torch.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return torch.stack([x, y, z, w])


# ---------------------------------------------------------
# Point cloud conversions (torch versions)
# ---------------------------------------------------------

def dm_to_cam_pc(dm: torch.Tensor, cam: tuple) -> torch.Tensor:
    """Torch version of utils.geometry.dm_to_cam_pc. Returns (N,3)."""
    intrinsics, _ = cam
    H, W = dm.shape
    device = dm.device
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    v = torch.arange(H, device=device)
    u = torch.arange(W, device=device)
    uu, vv = torch.meshgrid(u, v, indexing='xy')

    z = dm
    mask = z > 0
    if not torch.any(mask):
        return torch.zeros((0, 3), dtype=dm.dtype, device=device)

    x = (uu[mask] - cx) / fx
    y = (vv[mask] - cy) / fy
    z_vals = z[mask]
    cam_pc = torch.stack([x * z_vals, y * z_vals, z_vals], dim=-1)
    return cam_pc


def cam_pc_to_world_pc(cam_pc: torch.Tensor, cam: tuple) -> torch.Tensor:
    """Torch version of utils.geometry.cam_pc_to_world_pc. (N,3)."""
    _, extrinsics = cam
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    return (R.transpose(-2, -1) @ (cam_pc - t).T).T


def world_pc_to_cam_pc(world_pc: torch.Tensor, cam: tuple) -> torch.Tensor:
    """Torch version of utils.geometry.world_pc_to_cam_pc. (N,3)."""
    _, extrinsics = cam
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    return (R @ world_pc.T).T + t


def dm_to_world_pc(dm: torch.Tensor, cam: tuple) -> torch.Tensor:
    """Torch version of utils.geometry.dm_to_world_pc."""
    cam_pc = dm_to_cam_pc(dm, cam)
    world_pc = cam_pc_to_world_pc(cam_pc, cam)
    return world_pc


def dm_to_cam_pm(dm: torch.Tensor, cam: tuple) -> torch.Tensor:
    """Torch version of utils.geometry.dm_to_cam_pm. Returns (H,W,4)."""
    intrinsics, _ = cam
    H, W = dm.shape
    device = dm.device
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    v = torch.arange(H, device=device)
    u = torch.arange(W, device=device)
    uu, vv = torch.meshgrid(u, v, indexing='xy')
    z = dm

    x = (uu - cx) / fx
    y = (vv - cy) / fy

    cam_pm = torch.zeros((H, W, 4), dtype=dm.dtype, device=device)
    cam_pm[..., 0] = x * z
    cam_pm[..., 1] = y * z
    cam_pm[..., 2] = z
    cam_pm[..., 3] = (z > 0).to(dm.dtype)
    return cam_pm


def cam_pc_to_cam_pm_with_torch(cam_pc: torch.Tensor, cam: tuple, image_shape: tuple, valid: bool = False) -> torch.Tensor:
    """Torch version mirroring utils.geometry.cam_pc_to_cam_pm_with_torch."""
    intrinsics, _ = cam
    height, width = image_shape
    device = cam_pc.device

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x, y, z = cam_pc[:, 0], cam_pc[:, 1], cam_pc[:, 2]
    u = (x * fx / z + cx).round().to(torch.int64)
    v = (y * fy / z + cy).round().to(torch.int64)

    cam_pm = torch.zeros((height, width, 4), dtype=cam_pc.dtype, device=device)

    validity_mask = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if valid and cam_pc.shape[1] == 4:
        validity_mask &= cam_pc[:, 3] > 0

    if torch.any(validity_mask):
        cam_pm[v[validity_mask], u[validity_mask], :3] = cam_pc[validity_mask, :3]
        cam_pm[v[validity_mask], u[validity_mask], 3] = 1
    
    return cam_pm


def cam_pc_to_cam_pm(cam_pc: torch.Tensor, cam: tuple, image_shape: tuple, valid: bool = False) -> torch.Tensor:
    """Torch version of utils.geometry.cam_pc_to_cam_pm."""
    return cam_pc_to_cam_pm_with_torch(cam_pc, cam, image_shape, valid=valid)


def create_pm_in_ref_frame(world_pc: torch.Tensor, validity: torch.Tensor, cam_source: tuple, cam_reference: tuple,
                            image_shape: tuple, pm_source: str = "3d_tracks") -> torch.Tensor:
    """
    Torch version of utils.geometry.create_pm_in_ref_frame.
    world_pc: (N,3), validity: (N,1) 0/1
    Returns (H,W,4) torch tensor.
    """
    device = world_pc.device
    h, w = image_shape

    cam_pc_in_ref = world_pc_to_cam_pc(world_pc, cam_reference)  # (N,3)
    cam_pc_in_source = world_pc_to_cam_pc(world_pc, cam_source)  # (N,3)

    pm = torch.zeros((h, w, 4), dtype=world_pc.dtype, device=device)

    if pm_source == "3d_tracks":
        intrinsics = cam_source[0]
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        validity_flat = validity.flatten() > 0
        valid_cam_pc_source = cam_pc_in_source[validity_flat]
        valid_cam_pc_ref = cam_pc_in_ref[validity_flat]

        if valid_cam_pc_source.numel() == 0:
            return pm

        x, y, z = valid_cam_pc_source[:, 0], valid_cam_pc_source[:, 1], valid_cam_pc_source[:, 2]
        depth_valid = z > 0
        if not torch.any(depth_valid):
            return pm

        x, y, z = x[depth_valid], y[depth_valid], z[depth_valid]
        valid_cam_pc_ref = valid_cam_pc_ref[depth_valid]

        u = torch.round(x * fx / z + cx).to(torch.int64)
        v = torch.round(y * fy / z + cy).to(torch.int64)

        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not torch.any(in_bounds):
            return pm

        u = u[in_bounds]
        v = v[in_bounds]
        z = z[in_bounds]
        valid_cam_pc_ref = valid_cam_pc_ref[in_bounds]

        if u.numel() == 0:
            return pm

        pixel_keys = v * w + u
        # sort by (pixel, depth) using composite key to avoid version-specific APIs
        comp = pixel_keys.to(torch.float64) * 1e9 + z.to(torch.float64)
        order = torch.argsort(comp)
        pixel_keys_sorted = pixel_keys[order]
        u_sorted, v_sorted = u[order], v[order]
        ref_points_sorted = valid_cam_pc_ref[order]

        # keep first occurrence per pixel (nearest depth after sort)
        first_mask = torch.ones_like(pixel_keys_sorted, dtype=torch.bool)
        if pixel_keys_sorted.numel() > 1:
            first_mask[1:] = pixel_keys_sorted[1:] != pixel_keys_sorted[:-1]
        u_unique = u_sorted[first_mask]
        v_unique = v_sorted[first_mask]
        pts_unique = ref_points_sorted[first_mask]

        pm[v_unique, u_unique, :3] = pts_unique
        pm[v_unique, u_unique, 3] = 1
        return pm

    elif pm_source == "dm":
        cam_pc_in_source_valid = torch.cat([cam_pc_in_source, validity], dim=1)
        pm_source_map = cam_pc_to_cam_pm(cam_pc_in_source_valid, (cam_source[0], None), image_shape, valid=True)
        valid_mask = pm_source_map[..., 3] > 0
        if torch.any(valid_mask):
            valid_points_source = pm_source_map[valid_mask, :3]
            valid_points_world = cam_pc_to_world_pc(valid_points_source, cam_source)
            valid_points_ref = world_pc_to_cam_pc(valid_points_world, cam_reference)
            pm[valid_mask, :3] = valid_points_ref
            pm[valid_mask, 3] = 1
        return pm

    else:
        raise NotImplementedError("point map source not implemented")


def create_pm_from_dm_in_ref_frame(depthmap: torch.Tensor, cam_source: tuple, cam_reference: tuple) -> torch.Tensor:
    """Torch version of utils.geometry.create_pm_from_dm_in_ref_frame."""
    pm_in_source = dm_to_cam_pm(depthmap, cam_source)
    h, w = depthmap.shape
    device = depthmap.device
    pm_in_reference = torch.zeros((h, w, 4), dtype=depthmap.dtype, device=device)

    valid_mask = pm_in_source[..., 3] > 0
    if torch.any(valid_mask):
        # Use flat indexing to avoid boolean mask broadcasting quirks across dims
        flat_pm = pm_in_source.view(-1, 4)
        flat_mask = valid_mask.view(-1)
        valid_points_source = flat_pm[flat_mask, :3]
        valid_points_world = cam_pc_to_world_pc(valid_points_source, cam_source)
        valid_points_ref = world_pc_to_cam_pc(valid_points_world, cam_reference)

        # Scatter back to reference point map at valid pixel locations
        idxs = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)
        pm_in_reference_flat = pm_in_reference.view(-1, 4)
        pm_in_reference_flat[idxs, :3] = valid_points_ref
        pm_in_reference_flat[idxs, 3] = 1
    return pm_in_reference


def compute_scale_difference(approx_pm: torch.Tensor, actual_pm: torch.Tensor):
    """Torch version of utils.geometry.compute_scale_difference."""
    valid_approx = approx_pm[..., 3] > 0
    valid_actual = actual_pm[..., 3] > 0
    valid_mask = valid_approx & valid_actual

    approx_points = approx_pm[valid_mask, :3]
    actual_points = actual_pm[valid_mask, :3]

    if approx_points.numel() == 0:
        return None

    scale_ratios = actual_points.norm(dim=1) / (approx_points.norm(dim=1).clamp(min=1e-8))
    return torch.median(scale_ratios)


def get_motion_map_from_world_pc(
    world_pc_valid_list: list,
    cam_list: list,
    image_dimensions: tuple[int, int],
) -> torch.Tensor:
    """Torch version of utils.geometry.get_motion_map_from_world_pc."""
    device = world_pc_valid_list[0].device
    H, W = image_dimensions
    T = len(world_pc_valid_list)

    ref_cam = cam_list[0]
    world_pc_last = world_pc_valid_list[-1]
    last_valid_mask = world_pc_last[:, 3] > 0

    motion_map = torch.zeros((T - 1, H, W, 4), dtype=world_pc_valid_list[0].dtype, device=device)
    for k in range(T - 1):
        intr_k, _ = cam_list[k]
        world_pc_k = world_pc_valid_list[k]
        valid_mask = (world_pc_k[:, 3] > 0) & last_valid_mask
        if not torch.any(valid_mask):
            continue

        world_k = world_pc_k[valid_mask, :3]
        world_last = world_pc_last[valid_mask, :3]

        cam_ref_k = world_pc_to_cam_pc(world_k, ref_cam)
        cam_ref_last = world_pc_to_cam_pc(world_last, ref_cam)
        motion_3d = cam_ref_last - cam_ref_k

        cam_k = world_pc_to_cam_pc(world_k, cam_list[k])
        x, y, z = cam_k[:, 0], cam_k[:, 1], cam_k[:, 2]
        fx, fy = intr_k[0, 0], intr_k[1, 1]
        cx, cy = intr_k[0, 2], intr_k[1, 2]

        z_pos = z > 0
        if not torch.any(z_pos):
            continue
        x, y, z = x[z_pos], y[z_pos], z[z_pos]
        motion_3d = motion_3d[z_pos]

        u = torch.round(fx * x / z + cx).to(torch.int64)
        v = torch.round(fy * y / z + cy).to(torch.int64)

        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not torch.any(in_bounds):
            continue

        u, v = u[in_bounds], v[in_bounds]
        motion_valid = motion_3d[in_bounds]

        motion_map[k, v, u, :3] = motion_valid
        motion_map[k, v, u, 3] = 1
    return motion_map


def get_motion_map_from_cam_pc(
    cam_pc_valid_list: list,
    ref_intrinsics: torch.Tensor,
    image_dimensions: tuple[int, int],
) -> torch.Tensor:
    """Torch version of utils.geometry.get_motion_map_from_cam_pc."""
    device = cam_pc_valid_list[0].device
    H, W = image_dimensions
    num_frames = len(cam_pc_valid_list)

    motion_map = torch.zeros((num_frames - 1, H, W, 4), dtype=cam_pc_valid_list[0].dtype, device=device)
    fx, fy = ref_intrinsics[0, 0], ref_intrinsics[1, 1]
    cx, cy = ref_intrinsics[0, 2], ref_intrinsics[1, 2]

    for i in range(num_frames - 1):
        cam_pc_ref = cam_pc_valid_list[i]
        cam_pc_target = cam_pc_valid_list[i + 1]

        valid_mask = (cam_pc_ref[:, 3] > 0) & (cam_pc_target[:, 3] > 0)
        cam_pc_ref_valid = cam_pc_ref[valid_mask, :3]
        cam_pc_target_valid = cam_pc_target[valid_mask, :3]

        motion_3d = cam_pc_target_valid - cam_pc_ref_valid

        x, y, z = cam_pc_ref_valid[:, 0], cam_pc_ref_valid[:, 1], cam_pc_ref_valid[:, 2]
        z_positive_mask = z > 0
        x, y, z = x[z_positive_mask], y[z_positive_mask], z[z_positive_mask]
        motion_3d = motion_3d[z_positive_mask]

        u = torch.round(fx * x / z + cx).to(torch.int64)
        v = torch.round(fy * y / z + cy).to(torch.int64)

        in_bounds_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u_valid, v_valid = u[in_bounds_mask], v[in_bounds_mask]
        motion_valid = motion_3d[in_bounds_mask]

        motion_map[i, v_valid, u_valid, :3] = motion_valid
        motion_map[i, v_valid, u_valid, 3] = 1

    return motion_map


# ---------------------------------------------------------
# Cropping and resizing (not required for GPU pipeline here)
# ---------------------------------------------------------

def crop_image_depthmap(*args, **kwargs):
    raise NotImplementedError("torch crop_image_depthmap not implemented; use CPU path if needed.")


def rescale_image_depthmap(*args, **kwargs):
    raise NotImplementedError("torch rescale_image_depthmap not implemented; use CPU path if needed.")


def compute_camera_matrix_for_crop(*args, **kwargs):
    raise NotImplementedError("torch compute_camera_matrix_for_crop not implemented; use CPU path if needed.")


def crop_resize_if_necessary(*args, **kwargs):
    raise NotImplementedError("torch crop_resize_if_necessary not implemented; use CPU path if needed.")


def bbox_from_intrinsics_in_out(*args, **kwargs):
    raise NotImplementedError("torch bbox_from_intrinsics_in_out not implemented; use CPU path if needed.")


def normalize_pointcloud(pts1: torch.Tensor, pts2: torch.Tensor | None, norm_mode: str = 'avg_dis',
                         valid1: torch.Tensor | None = None, valid2: torch.Tensor | None = None, ret_factor: bool = False):
    """
    Torch version of utils.geometry.normalize_pointcloud (limited features).
    Implements the 'avg' family and a subset of modes used in GPU pipelines.
    """
    assert pts1.ndim >= 3 and pts1.shape[-1] == 3
    if pts2 is not None:
        assert pts2.ndim >= 3 and pts2.shape[-1] == 3

    nm, dis_mode = norm_mode.split('_')

    def _invalid_to_zeros(pts: torch.Tensor, valid: torch.Tensor | None):
        if valid is None:
            return pts, pts.numel() // 3
        mask = valid > 0
        out = torch.zeros_like(pts)
        out[mask.expand_as(pts)] = pts[mask.expand_as(pts)]
        return out, mask.sum().item()

    if nm == 'avg':
        nan_pts1, nnz1 = _invalid_to_zeros(pts1, valid1)
        if pts2 is not None:
            nan_pts2, nnz2 = _invalid_to_zeros(pts2, valid2)
            all_pts = torch.cat((nan_pts1, nan_pts2), dim=1)
        else:
            nan_pts2, nnz2 = None, 0
            all_pts = nan_pts1

        all_dis = torch.linalg.norm(all_pts, dim=-1)
        if dis_mode == 'dis':
            pass
        elif dis_mode == 'log1p':
            all_dis = torch.log1p(all_dis)
        else:
            raise ValueError(f'bad {dis_mode=}')

        norm_factor = all_dis.sum(dim=1) / (nnz1 + nnz2 + 1e-8)
    else:
        raise ValueError(f'bad {nm=}')

    norm_factor = norm_factor.clamp(min=1e-8)
    while norm_factor.ndim < pts1.ndim:
        norm_factor = norm_factor.unsqueeze(-1)

    res = pts1 / norm_factor
    if pts2 is not None:
        res = (res, pts2 / norm_factor)
    if ret_factor:
        if isinstance(res, tuple):
            res = res + (norm_factor,)
        else:
            res = (res, norm_factor)
    return res
