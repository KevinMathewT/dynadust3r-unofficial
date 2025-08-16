import torch
import numpy as np
import matplotlib.cm as cm
import cv2
import wandb

from utils.geometry import cam_pc_to_world_pc, world_pc_to_cam_pc
import utils.rerun_viz as rerun_viz  # rerun-based visualizations (moved from loaders/utils/viz.py)

def _normalize_pc_pair_np(pc1, pc2, valid1, valid2):
    """
    joint scale = mean‖p‖ all *valid* points in the two clouds.
    returns a scalar.
    """
    pc1_valid = pc1[valid1]          # (N1, 3)
    pc2_valid = pc2[valid2]          # (N2, 3)

    if pc1_valid.size + pc2_valid.size == 0:
        return 1.0

    all_dis = np.linalg.norm(np.concatenate((pc1_valid, pc2_valid), axis=0), axis=-1)
    return np.maximum(np.mean(all_dis), 1e-8)                      # scalar


# helper -----------------------------------------------------------
def _make_depth_and_disp_imgs_np(pc: np.ndarray,
                                 valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    pc    : (h, w, 3) xyz in left/right camera frame
    valid : (h, w) boolean mask from gt indicating valid points

    returns
        depth_img : (h, w, 3) uint8, prism colormap
        disp_img  : (h, w, 3) uint8, turbo colormap
    """
    # print(f"DEBUG: _make_depth_and_disp_imgs_np - pc shape: {pc.shape}, valid shape: {valid.shape}")
    depth = pc[..., 2]            # (h, w)  z-component depth
    disp  = 1.0 / np.clip(depth, a_min=1e-6, a_max=None)       # (h, w)  disparity

    depth[~valid] = np.nan
    disp[ ~valid] = np.nan

    # normalise 0‒1 on valid region
    def _norm(x: np.ndarray) -> np.ndarray:      # (h, w)
        vmin = np.nanmin(x)
        vmax = np.nanmax(x)
        return (x - vmin) / (vmax - vmin + 1e-6)

    depth_n   = _norm(depth)           # (h, w)
    disp_n    = _norm(disp )           # (h, w)

    depth_rgb = (cm.prism(depth_n)[..., :3] * 255).astype(np.uint8)  # (h, w, 3)
    disp_rgb  = (cm.turbo(disp_n )[..., :3] * 255).astype(np.uint8)  # (h, w, 3)
    return depth_rgb, disp_rgb


def _conf_to_gray_img_np(conf: np.ndarray,
                         valid: np.ndarray | None = None) -> np.ndarray:
    """
    conf  : (H, W)  confidence ∈ [1, ∞)
    valid : (H, W)  optional mask; invalid → nan
    maps 1 → black,   large → white,   returns (H, W, 3) uint8
    """
    # print(f"DEBUG: _conf_to_gray_img_np - conf shape: {conf.shape if conf is not None else 'None'}, valid shape: {valid.shape if valid is not None else 'None'}")
    if conf is None:
        return None
    conf = conf.astype(np.float32).copy()
    if valid is not None:
        conf[~valid] = np.nan

    # normalise so that 1 → 0,  max → 1  (logically black→white)
    c_min, c_max = 1.0, np.nanmax(conf)
    norm = (conf - c_min) / (c_max - c_min + 1e-6)     # (H, W)
    norm = np.clip(norm, 0, 1)

    gray_rgb = (cm.gray(norm)[..., :3] * 255).astype(np.uint8)  # (H, W, 3)
    return gray_rgb


def project_3d_to_2d_vec(points: np.ndarray, source_cam: tuple, proj_cam: tuple) -> np.ndarray:
    """
    Vectorized projection of 3D points to 2D UV.

    Args:
        points (np.ndarray): (N, 3) 3D points in source frame.
        source_cam (tuple): (intrinsics, extrinsics) for source.
        proj_cam (tuple): (intrinsics, extrinsics) for projection.

    Returns:
        np.ndarray: (N, 2) UV coordinates, nan where invalid.
    """
    intrinsics_proj, _ = proj_cam

    is_same_cam = np.allclose(source_cam[0], proj_cam[0]) and np.allclose(source_cam[1], proj_cam[1])
    # print(f"DEBUG: project_3d_to_2d_vec - points shape: {points.shape}, source_cam same as proj_cam: {is_same_cam}")

    if is_same_cam:
        uvw = (intrinsics_proj @ points.T).T
    else:
        pts_world = cam_pc_to_world_pc(points, source_cam)
        pts_proj = world_pc_to_cam_pc(pts_world, proj_cam)
        uvw = (intrinsics_proj @ pts_proj.T).T
    # print(f"DEBUG: project_3d_to_2d_vec - uvw shape: {uvw.shape}, min z: {np.min(uvw[:, 2]) if uvw.size > 0 else 'N/A'}, max z: {np.max(uvw[:, 2]) if uvw.size > 0 else 'N/A'}")

    mask = uvw[:, 2] > 1e-6
    # print(f"DEBUG: project_3d_to_2d_vec - mask shape: {mask.shape}, num valid: {np.sum(mask)}")
    uv = np.full((len(points), 2), np.nan)
    if np.any(mask):
        # print(f"DEBUG: project_3d_to_2d_vec - uvw[mask, :2] shape: {uvw[mask, :2].shape}, uvw[mask, 2, None] shape: {uvw[mask, 2, None].shape}")
        uv[mask] = uvw[mask, :2] / uvw[mask, 2, None]
    return uv


def draw_motion_on_image(image: np.ndarray, pc: np.ndarray, npc: np.ndarray, valid: np.ndarray,
                         source_cam: tuple, proj_cam: tuple) -> np.ndarray:
    """
    Draws fluorescent green lines on the image representing motions from pc to npc.

    Args:
        image (np.ndarray): (H, W, 3) uint8 base image.
        pc (np.ndarray): (H, W, 3) base points.
        npc (np.ndarray): (H, W, 3) next points.
        valid (np.ndarray): (H, W) boolean validity mask.
        source_cam (tuple): Camera for point coordinates.
        proj_cam (tuple): Camera for projection (image's camera).

    Returns:
        np.ndarray: Image with drawn lines.
    """
    # print(f"DEBUG: draw_motion_on_image - image shape: {image.shape}, pc shape: {pc.shape}, npc shape: {npc.shape}, valid shape: {valid.shape}")
    img_draw = image.copy()
    h, w = img_draw.shape[:2]

    v_idx, u_idx = np.where(valid)
    # print(f"DEBUG: draw_motion_on_image - num valid points: {len(v_idx)}")
    if len(v_idx) == 0:
        return img_draw

    p = pc[v_idx, u_idx]  # (N, 3)
    n = npc[v_idx, u_idx]  # (N, 3)
    # print(f"DEBUG: draw_motion_on_image - p shape: {p.shape}, n shape: {n.shape}")
    
    # Debug: Check motion magnitudes
    motion = n - p
    motion_magnitudes = np.linalg.norm(motion, axis=-1)
    if len(motion_magnitudes) > 0:
        large_motions = motion_magnitudes > 10.0  # Flag motions larger than 10 units
        if large_motions.any():
            print(f"WARNING: Large motions detected! Count: {large_motions.sum()}/{len(motion_magnitudes)}")
            print(f"Motion stats - Min: {motion_magnitudes.min():.3f}, Max: {motion_magnitudes.max():.3f}, Mean: {motion_magnitudes.mean():.3f}")

    uv_p = project_3d_to_2d_vec(p, source_cam, proj_cam)  # (N, 2)
    uv_n = project_3d_to_2d_vec(n, source_cam, proj_cam)  # (N, 2)
    # print(f"DEBUG: draw_motion_on_image - uv_p shape: {uv_p.shape}, uv_n shape: {uv_n.shape}")

    mask_valid = np.isfinite(uv_p).all(axis=1) & np.isfinite(uv_n).all(axis=1)
    # print(f"DEBUG: draw_motion_on_image - num valid projections: {np.sum(mask_valid)}")
    if not np.any(mask_valid):
        return img_draw

    uv_p = uv_p[mask_valid]
    uv_n = uv_n[mask_valid]

    mask_bounds_p = (uv_p[:, 0] >= 0) & (uv_p[:, 0] < w) & (uv_p[:, 1] >= 0) & (uv_p[:, 1] < h)
    mask_bounds_n = (uv_n[:, 0] >= 0) & (uv_n[:, 0] < w) & (uv_n[:, 1] >= 0) & (uv_n[:, 1] < h)
    mask_bounds = mask_bounds_p & mask_bounds_n
    # print(f"DEBUG: draw_motion_on_image - num in bounds: {np.sum(mask_bounds)}")
    if not np.any(mask_bounds):
        return img_draw

    uv_p_valid = uv_p[mask_bounds].round().astype(int)
    uv_n_valid = uv_n[mask_bounds].round().astype(int)

    for i in range(len(uv_p_valid)):
        pt1 = (uv_p_valid[i, 0], uv_p_valid[i, 1])
        pt2 = (uv_n_valid[i, 0], uv_n_valid[i, 1])
        cv2.line(img_draw, pt1, pt2, (0, 255, 0), thickness=1)

    return img_draw


def save_visualizations(batch, outputs, base_name, i=0, *args, **kwargs):
    # Extract all tensors to numpy early where possible

    # Images to numpy
    left_img = batch["left_image"][i].permute(1, 2, 0).detach().cpu().numpy()
    mid_img = batch["mid_image"][i].permute(1, 2, 0).detach().cpu().numpy()
    right_img = batch["right_image"][i].permute(1, 2, 0).detach().cpu().numpy()
    # print(f"DEBUG: save_visualizations - left_img shape: {left_img.shape}, mid_img shape: {mid_img.shape}, right_img shape: {right_img.shape}")

    # Cameras to numpy
    intrinsic_left_np = batch["cam"][0][i].detach().cpu().numpy()
    extrinsic_left_np = batch["cam"][1][i].detach().cpu().numpy()
    intrinsic_mid_np = batch["cam_mid"][0][i].detach().cpu().numpy()
    extrinsic_mid_np = batch["cam_mid"][1][i].detach().cpu().numpy()
    intrinsic_right_np = batch["cam_right"][0][i].detach().cpu().numpy()
    extrinsic_right_np = batch["cam_right"][1][i].detach().cpu().numpy()

    left_cam = (intrinsic_left_np, extrinsic_left_np)
    mid_cam = (intrinsic_mid_np, extrinsic_mid_np)
    right_cam = (intrinsic_right_np, extrinsic_right_np)
    # print(f"DEBUG: save_visualizations - left_cam intrinsics shape: {intrinsic_left_np.shape}, extrinsics shape: {extrinsic_left_np.shape}")

    # Extract base point clouds and validity masks to numpy
    gt_left_pc_np = batch["left_pm"][i, ..., :3].detach().cpu().numpy()
    gt_right_pc_np = batch["right_pm"][i, ..., :3].detach().cpu().numpy()
    valid_left_np = (batch["left_pm"][i, ..., 3] > 0).detach().cpu().numpy()
    valid_right_np = (batch["right_pm"][i, ..., 3] > 0).detach().cpu().numpy()
    # print(f"DEBUG: save_visualizations - gt_left_pc_np shape: {gt_left_pc_np.shape}, valid_left_np shape: {valid_left_np.shape}")
    
    # Debug: Check for points at origin
    left_at_origin = np.all(gt_left_pc_np == 0, axis=-1) & valid_left_np
    right_at_origin = np.all(gt_right_pc_np == 0, axis=-1) & valid_right_np
    if left_at_origin.any() or right_at_origin.any():
        print(f"WARNING: Found valid GT points at origin - Left: {left_at_origin.sum()}, Right: {right_at_origin.sum()}")
        print(f"Total valid points - Left: {valid_left_np.sum()}, Right: {valid_right_np.sum()}")
    
    pred_left_pc_np = outputs["left_map_pred"][i].detach().cpu().numpy()
    pred_right_pc_np = outputs["right_map_pred_in_left_frame"][i].detach().cpu().numpy()
    # print(f"DEBUG: save_visualizations - pred_left_pc_np shape: {pred_left_pc_np.shape}")

    # Create next point clouds by adding motion and convert to numpy
    # GT next PCs
    gt_l2m_pc_np = (batch["left_pm"][i, ..., :3] + batch["motion_gt"]["l2m"][i, ..., :3]).detach().cpu().numpy()
    gt_r2m_pc_np = (batch["right_pm"][i, ..., :3] + batch["motion_gt"]["r2m"][i, ..., :3]).detach().cpu().numpy()
    gt_l2r_pc_np = (batch["left_pm"][i, ..., :3] + batch["motion_gt"]["l2r"][i, ..., :3]).detach().cpu().numpy()
    gt_r2l_pc_np = (batch["right_pm"][i, ..., :3] + batch["motion_gt"]["r2l"][i, ..., :3]).detach().cpu().numpy()
    # print(f"DEBUG: save_visualizations - gt_l2m_pc_np shape: {gt_l2m_pc_np.shape}")
    
    # Motion validity masks to numpy
    valid_l2m_np = (valid_left_np & (batch["motion_gt"]["l2m"][i, ..., 3] > 0).detach().cpu().numpy())
    valid_r2m_np = (valid_right_np & (batch["motion_gt"]["r2m"][i, ..., 3] > 0).detach().cpu().numpy())
    valid_l2r_np = (valid_left_np & (batch["motion_gt"]["l2r"][i, ..., 3] > 0).detach().cpu().numpy())
    valid_r2l_np = (valid_right_np & (batch["motion_gt"]["r2l"][i, ..., 3] > 0).detach().cpu().numpy())
    # print(f"DEBUG: save_visualizations - valid_l2m_np shape: {valid_l2m_np.shape}, num valid: {np.sum(valid_l2m_np)}")
    
    # Pred next PCs to numpy
    tq_mid = batch["query_times"][i, 0].item()
    pred_l2m_pc_np = (outputs["left_map_pred"][i] + outputs["motion_pred"][f"l_to_{tq_mid:.3g}"][i, ..., :3]).detach().cpu().numpy()
    pred_r2m_pc_np = (outputs["right_map_pred_in_left_frame"][i] + outputs["motion_pred"][f"r_to_{tq_mid:.3g}"][i, ..., :3]).detach().cpu().numpy()
    pred_l2r_pc_np = (outputs["left_map_pred"][i] + outputs["motion_pred"]["l_to_1"][i, ..., :3]).detach().cpu().numpy()
    pred_r2l_pc_np = (outputs["right_map_pred_in_left_frame"][i] + outputs["motion_pred"]["r_to_0"][i, ..., :3]).detach().cpu().numpy()
    # print(f"DEBUG: save_visualizations - pred_l2m_pc_np shape: {pred_l2m_pc_np.shape}")
    
    # ---------- confidence maps -----------------------------------------------
    k_l2m, k_r2m, k_l2r, k_r2l = (
        f"l_to_{tq_mid:.3g}", f"r_to_{tq_mid:.3g}", "l_to_1", "r_to_0"
    )

    # per‑pixel confidence for maps to numpy
    conf_left_pred_np  = outputs.get("left_map_pred_conf",  None)
    conf_right_pred_np = outputs.get("right_map_pred_conf", None)
    conf_left_pred_np  = conf_left_pred_np[i].detach().cpu().numpy()  if conf_left_pred_np  is not None else None
    conf_right_pred_np = conf_right_pred_np[i].detach().cpu().numpy() if conf_right_pred_np is not None else None
    # print(f"DEBUG: save_visualizations - conf_left_pred_np shape: {conf_left_pred_np.shape if conf_left_pred_np is not None else 'None'}")

    # per‑pixel confidence for motions to numpy
    motion_pred = outputs["motion_pred"]
    conf_motion_l2m_pred_np, conf_motion_r2m_pred_np, \
    conf_motion_l2r_pred_np,  conf_motion_r2l_pred_np = [
        (motion_pred.get(f"{k}_conf")[i].detach().cpu().numpy() if motion_pred.get(f"{k}_conf") is not None else None)
        for k in (k_l2m, k_r2m, k_l2r, k_r2l)
    ]
    # print(f"DEBUG: save_visualizations - conf_motion_l2m_pred_np shape: {conf_motion_l2m_pred_np.shape if conf_motion_l2m_pred_np is not None else 'None'}")
    # ---------------------------------------------------------------------------

    # Joint normalization of base point clouds
    gt_base_scale = _normalize_pc_pair_np(gt_left_pc_np, gt_right_pc_np, valid_left_np, valid_right_np)
    pred_base_scale = _normalize_pc_pair_np(pred_left_pc_np, pred_right_pc_np, valid_left_np, valid_right_np)
    # print(f"DEBUG: save_visualizations - gt_base_scale: {gt_base_scale}, pred_base_scale: {pred_base_scale}")
    
    # Compute multiplier to bring pred to GT scale
    multiplier = gt_base_scale / pred_base_scale
    # print(f"DEBUG: save_visualizations - multiplier: {multiplier}")
    
    # Apply scaling: GT untouched, pred scaled by multiplier
    gt_left_norm_np = gt_left_pc_np
    gt_right_norm_np = gt_right_pc_np
    gt_l2m_norm_np = gt_l2m_pc_np
    gt_r2m_norm_np = gt_r2m_pc_np
    gt_l2r_norm_np = gt_l2r_pc_np
    gt_r2l_norm_np = gt_r2l_pc_np
    
    pred_left_norm_np = pred_left_pc_np * multiplier
    pred_right_norm_np = pred_right_pc_np * multiplier
    pred_l2m_norm_np = pred_l2m_pc_np * multiplier
    pred_r2m_norm_np = pred_r2m_pc_np * multiplier
    pred_l2r_norm_np = pred_l2r_pc_np * multiplier
    pred_r2l_norm_np = pred_r2l_pc_np * multiplier

     # 4 × (depth, disparity) visualisations ----------------------------
    gt_left_depth_img,  gt_left_disp_img  = _make_depth_and_disp_imgs_np(gt_left_norm_np,  valid_left_np)
    gt_right_depth_img, gt_right_disp_img = _make_depth_and_disp_imgs_np(gt_right_norm_np, valid_right_np)
    pr_left_depth_img,  pr_left_disp_img  = _make_depth_and_disp_imgs_np(pred_left_norm_np, valid_left_np)
    pr_right_depth_img, pr_right_disp_img = _make_depth_and_disp_imgs_np(pred_right_norm_np, valid_right_np)
    # print(f"DEBUG: save_visualizations - gt_left_depth_img shape: {gt_left_depth_img.shape}")

    # ---------- confidence visualisations (greyscale) ---------------------------
    conf_imgs = {}

    if conf_left_pred_np is not None:
        conf_imgs["conf_left"]  = _conf_to_gray_img_np(conf_left_pred_np,  valid_left_np)
    if conf_right_pred_np is not None:
        conf_imgs["conf_right"] = _conf_to_gray_img_np(conf_right_pred_np, valid_right_np)

    motion_masks_np = (valid_l2m_np, valid_r2m_np, valid_l2r_np, valid_r2l_np)
    for name, conf_map, vmask in zip(
        ("conf_l2m", "conf_r2m", "conf_l2r", "conf_r2l"),
        (conf_motion_l2m_pred_np, conf_motion_r2m_pred_np,
         conf_motion_l2r_pred_np,  conf_motion_r2l_pred_np),
        motion_masks_np,
    ):
        if conf_map is not None:
            conf_imgs[name] = _conf_to_gray_img_np(conf_map, vmask)
    # print(f"DEBUG: save_visualizations - num conf_imgs: {len(conf_imgs)}")
    # ---------------------------------------------------------------------------

    # Unnormalize images for visualization
    left_img_unnorm = ((left_img + 1) / 2 * 255).astype(np.uint8)
    mid_img_unnorm = ((mid_img + 1) / 2 * 255).astype(np.uint8)
    right_img_unnorm = ((right_img + 1) / 2 * 255).astype(np.uint8)
    # print(f"DEBUG: save_visualizations - left_img_unnorm shape: {left_img_unnorm.shape}")

    # Motion line visualizations
    motion_line_imgs = {}

    # GT
    motion_line_imgs["gt_l2m"] = draw_motion_on_image(left_img_unnorm, gt_left_norm_np, gt_l2m_norm_np, valid_l2m_np, left_cam, left_cam)
    motion_line_imgs["gt_l2r"] = draw_motion_on_image(left_img_unnorm, gt_left_norm_np, gt_l2r_norm_np, valid_l2r_np, left_cam, left_cam)
    motion_line_imgs["gt_r2m"] = draw_motion_on_image(right_img_unnorm, gt_right_norm_np, gt_r2m_norm_np, valid_r2m_np, left_cam, right_cam)
    motion_line_imgs["gt_r2l"] = draw_motion_on_image(right_img_unnorm, gt_right_norm_np, gt_r2l_norm_np, valid_r2l_np, left_cam, right_cam)

    # Pred
    motion_line_imgs["pred_l2m"] = draw_motion_on_image(left_img_unnorm, pred_left_norm_np, pred_l2m_norm_np, valid_l2m_np, left_cam, left_cam)
    motion_line_imgs["pred_l2r"] = draw_motion_on_image(left_img_unnorm, pred_left_norm_np, pred_l2r_norm_np, valid_l2r_np, left_cam, left_cam)
    motion_line_imgs["pred_r2m"] = draw_motion_on_image(right_img_unnorm, pred_right_norm_np, pred_r2m_norm_np, valid_r2m_np, left_cam, right_cam)
    motion_line_imgs["pred_r2l"] = draw_motion_on_image(right_img_unnorm, pred_right_norm_np, pred_r2l_norm_np, valid_r2l_np, left_cam, right_cam)
    # print(f"DEBUG: save_visualizations - example motion_line_img shape: {motion_line_imgs['gt_l2m'].shape if 'gt_l2m' in motion_line_imgs else 'N/A'}")

    base = f"{base_name}_i{i}"
    # global k; base += f"_k{k}"; k+=1; print(base);

    log_dict = {
        # input images
        f"{base}/left_img/input"   : wandb.Image(left_img_unnorm,  caption="left input"),
        f"{base}/mid_img/input"    : wandb.Image(mid_img_unnorm,   caption="mid input"),
        f"{base}/right_img/input"  : wandb.Image(right_img_unnorm, caption="right input"),

        # depth
        f"{base}/left_depth/gt"  : wandb.Image(gt_left_depth_img,  caption="GT Left Depth"),
        f"{base}/right_depth/gt" : wandb.Image(gt_right_depth_img, caption="GT Right Depth"),
        f"{base}/left_depth/pred"  : wandb.Image(pr_left_depth_img,  caption="Pred Left Depth"),
        f"{base}/right_depth/pred" : wandb.Image(pr_right_depth_img, caption="Pred Right Depth"),

        # disparity
        f"{base}/left_disp/gt"  : wandb.Image(gt_left_disp_img,  caption="GT Left Disp"),
        f"{base}/right_disp/gt" : wandb.Image(gt_right_disp_img, caption="GT Right Disp"),
        f"{base}/left_disp/pred"  : wandb.Image(pr_left_disp_img,  caption="Pred Left Disp"),
        f"{base}/right_disp/pred" : wandb.Image(pr_right_disp_img, caption="Pred Right Disp"),

        # motion lines
        f"{base}/l2m_motion/gt"  : wandb.Image(motion_line_imgs["gt_l2m"],  caption="GT L2M Motion"),
        f"{base}/l2r_motion/gt"  : wandb.Image(motion_line_imgs["gt_l2r"],  caption="GT L2R Motion"),
        f"{base}/r2m_motion/gt"  : wandb.Image(motion_line_imgs["gt_r2m"],  caption="GT R2M Motion"),
        f"{base}/r2l_motion/gt"  : wandb.Image(motion_line_imgs["gt_r2l"],  caption="GT R2L Motion"),
        f"{base}/l2m_motion/pred": wandb.Image(motion_line_imgs["pred_l2m"], caption="Pred L2M Motion"),
        f"{base}/l2r_motion/pred": wandb.Image(motion_line_imgs["pred_l2r"], caption="Pred L2R Motion"),
        f"{base}/r2m_motion/pred": wandb.Image(motion_line_imgs["pred_r2m"], caption="Pred R2M Motion"),
        f"{base}/r2l_motion/pred": wandb.Image(motion_line_imgs["pred_r2l"], caption="Pred R2L Motion"),
    }


    # add confidence maps if available
    for key, img in conf_imgs.items():
        log_dict[f"{base}/conf/{key}"] = wandb.Image(img, caption=f"Conf {key.upper()}")

    wandb.log(log_dict, commit=True)

    # Extract GT motions to numpy
    gt_l2m_motion_np = batch["motion_gt"]["l2m"][i].detach().cpu().numpy()
    gt_r2m_motion_np = batch["motion_gt"]["r2m"][i].detach().cpu().numpy()
    gt_l2r_motion_np = batch["motion_gt"]["l2r"][i].detach().cpu().numpy()
    gt_r2l_motion_np = batch["motion_gt"]["r2l"][i].detach().cpu().numpy()

    # Zero motion template (H, W, 4) with validity=0
    zero_motion = np.zeros_like(gt_l2m_motion_np)
    
    # GT visualization sequence
    pms_gt = [
        gt_left_norm_np, gt_l2m_norm_np,
        gt_right_norm_np, gt_r2m_norm_np,
        gt_left_norm_np, gt_l2r_norm_np,
        gt_right_norm_np, gt_r2l_norm_np
    ]
    motion_map_gt = np.stack([
        gt_l2m_motion_np,
        zero_motion,
        gt_r2m_motion_np,
        zero_motion,
        gt_l2r_motion_np,
        zero_motion,
        gt_r2l_motion_np
    ], axis=0)
    image_seq_gt = [
        left_img_unnorm, mid_img_unnorm,
        right_img_unnorm, mid_img_unnorm,
        left_img_unnorm, right_img_unnorm,
        right_img_unnorm, left_img_unnorm
    ]
    # If you want rerun-based sequence viz, use utils.rerun_viz:
    # rerun_viz.visualize_sequence_from_pms(pms_gt, motion_map_gt, image_seq_gt, name=f"{base}/gt_motions", save=False)
    
    # Pred visualization sequence (add GT validity to pred motions since pred motions are :3)
    pred_l2m_motion_np = np.concatenate((outputs["motion_pred"][k_l2m][i].detach().cpu().numpy(), gt_l2m_motion_np[..., 3:]), axis=-1)
    pred_r2m_motion_np = np.concatenate((outputs["motion_pred"][k_r2m][i].detach().cpu().numpy(), gt_r2m_motion_np[..., 3:]), axis=-1)
    pred_l2r_motion_np = np.concatenate((outputs["motion_pred"]["l_to_1"][i].detach().cpu().numpy(), gt_l2r_motion_np[..., 3:]), axis=-1)
    pred_r2l_motion_np = np.concatenate((outputs["motion_pred"]["r_to_0"][i].detach().cpu().numpy(), gt_r2l_motion_np[..., 3:]), axis=-1)
    
    pms_pred = [
        pred_left_norm_np, pred_l2m_norm_np,
        pred_right_norm_np, pred_r2m_norm_np,
        pred_left_norm_np, pred_l2r_norm_np,
        pred_right_norm_np, pred_r2l_norm_np
    ]
    motion_map_pred = np.stack([
        pred_l2m_motion_np,
        zero_motion,
        pred_r2m_motion_np,
        zero_motion,
        pred_l2r_motion_np,
        zero_motion,
        pred_r2l_motion_np
    ], axis=0)
    image_seq_pred = image_seq_gt  # Same images for coloring
    # rerun_viz.visualize_sequence_from_pms(pms_pred, motion_map_pred, image_seq_pred, name=f"{base}/pred_motions", save=False)
