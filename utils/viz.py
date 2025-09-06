import numpy as np
import torch
import cv2
import matplotlib.cm as cm
import wandb

from utils.geometry import cam_pc_to_world_pc, world_pc_to_cam_pc
from models.dust3r.utils.heads.postprocess import reg_dense_depth

# =====================
# Utility helpers
# =====================

def _to_np(x):
    """Torch tensor -> numpy (no grad, on CPU)."""
    return x.detach().cpu().numpy()


def _norm01(x: np.ndarray) -> np.ndarray:
    """Normalize array to [0,1] over finite entries; return zeros if none finite."""
    m = np.isfinite(x)
    if not np.any(m):
        return np.zeros_like(x, dtype=np.float32)
    vmin = np.nanmin(x)
    vmax = np.nanmax(x)
    return (x - vmin) / (vmax - vmin + 1e-6)


def make_depth_and_disp_imgs(pc_left: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Visualize depth & disparity from LEFT-CAMERA 3D coordinates.
    pc_left: (H, W, 3) 3D points in LEFT camera frame
    valid  : (H, W) boolean mask
    Returns 8-bit RGB images.
    """
    depth = pc_left[..., 2].astype(np.float32)
    disp  = 1.0 / np.clip(depth, 1e-6, None)

    depth = depth.copy(); depth[~valid] = np.nan
    disp  = disp.copy();  disp[~valid]  = np.nan

    d01 = _norm01(depth)
    q01 = _norm01(disp)

    depth_rgb = (cm.prism(d01)[..., :3] * 255).astype(np.uint8)
    disp_rgb  = (cm.turbo(q01)[..., :3] * 255).astype(np.uint8)
    return depth_rgb, disp_rgb


def conf_to_gray_img(conf: np.ndarray | None, valid: np.ndarray | None = None) -> np.ndarray | None:
    """Map confidence (≥1) to grayscale: 1→black, large→white. Returns RGB uint8 or None."""
    if conf is None:
        return None
    c = conf.astype(np.float32).copy()
    if valid is not None:
        c[~valid] = np.nan
    c_min, c_max = 1.0, np.nanmax(c) if np.isfinite(c).any() else 1.0
    norm = np.clip((c - c_min) / (c_max - c_min + 1e-6), 0, 1)
    return (cm.gray(norm)[..., :3] * 255).astype(np.uint8)


def project_left3d_to_uv(points_left: np.ndarray,
                         left_cam: tuple,
                         proj_cam: tuple) -> np.ndarray:
    """
    Project 3D points expressed in the LEFT camera frame into a projection camera's image plane.
    Returns (N,2) pixel coords with NaNs where depth<=0.
    """
    K_proj, E_proj = proj_cam
    K_left, E_left = left_cam

    # Fast path if projecting to the LEFT camera itself
    same_cam = np.allclose(K_proj, K_left) and np.allclose(E_proj, E_left)
    if same_cam:
        cam_pts = points_left  # already in left cam frame
    else:
        world = cam_pc_to_world_pc(points_left, left_cam)
        cam_pts = world_pc_to_cam_pc(world, proj_cam)

    uvw = (K_proj @ cam_pts.T).T  # (N,3)
    z = uvw[:, 2]
    mask = z > 1e-6
    uv = np.full((len(points_left), 2), np.nan, dtype=np.float32)
    if np.any(mask):
        uv[mask] = uvw[mask, :2] / z[mask, None]
    return uv


def draw_motion(image: np.ndarray,
                base_left: np.ndarray,
                next_left: np.ndarray,
                valid: np.ndarray,
                left_cam: tuple,
                proj_cam: tuple,
                thickness: int = 1) -> np.ndarray:
    """
    Draw motion vectors on `image`, projecting BOTH endpoints into the SAME `proj_cam` plane.
    - base_left / next_left: (H,W,3) in LEFT camera frame
    - valid: (H,W) boolean mask for pixels to draw
    - image: (H,W,3) uint8 canvas corresponding to `proj_cam`'s image
    """
    out = image.copy()
    H, W = out.shape[:2]

    v_idx, u_idx = np.where(valid)
    if len(v_idx) == 0:
        return out

    p = base_left[v_idx, u_idx]  # (N,3) in LEFT frame
    n = next_left[v_idx, u_idx]  # (N,3) in LEFT frame

    uv_p = project_left3d_to_uv(p, left_cam, proj_cam)
    uv_n = project_left3d_to_uv(n, left_cam, proj_cam)

    good = np.isfinite(uv_p).all(1) & np.isfinite(uv_n).all(1)
    if not np.any(good):
        return out

    uv_p = uv_p[good]
    uv_n = uv_n[good]

    inb_p = (uv_p[:, 0] >= 0) & (uv_p[:, 0] < W) & (uv_p[:, 1] >= 0) & (uv_p[:, 1] < H)
    inb_n = (uv_n[:, 0] >= 0) & (uv_n[:, 0] < W) & (uv_n[:, 1] >= 0) & (uv_n[:, 1] < H)
    inb = inb_p & inb_n
    if not np.any(inb):
        return out

    uv_p = uv_p[inb].round().astype(int)
    uv_n = uv_n[inb].round().astype(int)

    for (x1, y1), (x2, y2) in zip(uv_p, uv_n):
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), thickness=thickness)
    return out


# =====================
# Main visualization entry
# =====================

def save_visualizations(batch, outputs, base_name: str, i: int = 0,
                        depth_post_mode=None, motion_depth_post_mode=None):
    """
    Assumptions (enforced by the caller):
    - All point/motion map VALUES are in the LEFT camera frame.
    - Grid locations for left/right maps are pixels from the left/right images respectively.
    - Validity is indicated by channel 3 (>0) for *_pm and motion_gt maps.
    """
    # ---- Images (de-normalize from [-1,1] to uint8) ----
    def _unnorm_img(t):
        return ((_to_np(t).transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)

    left_img  = _unnorm_img(batch["left_image"][i])
    mid_img   = _unnorm_img(batch["mid_image"][i])
    right_img = _unnorm_img(batch["right_image"][i])

    # ---- Cameras ----
    left_cam  = (_to_np(batch["cam"][0][i]),       _to_np(batch["cam"][1][i]))
    mid_cam   = (_to_np(batch["cam_mid"][0][i]),   _to_np(batch["cam_mid"][1][i]))
    right_cam = (_to_np(batch["cam_right"][0][i]), _to_np(batch["cam_right"][1][i]))

    # ---- Base point maps (LEFT-frame values) & valid masks ----
    gt_left_pc   = _to_np(batch["left_pm"][i, ..., :3])
    gt_right_pc  = _to_np(batch["right_pm"][i, ..., :3])  # values already in LEFT frame
    valid_left   = _to_np(batch["left_pm"][i, ..., 3] > 0)
    valid_right  = _to_np(batch["right_pm"][i, ..., 3] > 0)

    pred_left_pc  = _to_np(outputs["left_map_pred"][i])
    pred_right_pc = _to_np(outputs["right_map_pred_in_left_frame"][i])

    # Apply post-processing if provided
    if depth_post_mode is not None:
        pred_left_pc = _to_np(reg_dense_depth(torch.tensor(pred_left_pc), depth_post_mode))
        pred_right_pc = _to_np(reg_dense_depth(torch.tensor(pred_right_pc), depth_post_mode))

    # ---- Motions (LEFT-frame displacements) & next point maps ----
    def _gt_motion(name):
        mot = _to_np(batch["motion_gt"][name][i])  # (H,W,4)
        return mot[..., :3], mot[..., 3] > 0

    gt_l2m, m_valid_l2m = _gt_motion("l2m")
    gt_r2m, m_valid_r2m = _gt_motion("r2m")
    gt_l2r, m_valid_l2r = _gt_motion("l2r")
    gt_r2l, m_valid_r2l = _gt_motion("r2l")

    # next = base + motion (all in LEFT frame)
    gt_l2m_pc = gt_left_pc  + gt_l2m
    gt_r2m_pc = gt_right_pc + gt_r2m
    gt_l2r_pc = gt_left_pc  + gt_l2r
    gt_r2l_pc = gt_right_pc + gt_r2l

    # motion valid = base valid & motion valid
    v_l2m = valid_left  & m_valid_l2m
    v_r2m = valid_right & m_valid_r2m
    v_l2r = valid_left  & m_valid_l2r
    v_r2l = valid_right & m_valid_r2l

    # Pred motions (LEFT-frame) and next points
    mp = outputs["motion_pred"]
    pred_l2m_pc = _to_np(outputs["left_map_pred"][i] + mp["l_to_t0"][i, ..., :3])
    pred_r2m_pc = _to_np(outputs["right_map_pred_in_left_frame"][i] + mp["r_to_t0"][i, ..., :3])
    pred_l2r_pc = _to_np(outputs["left_map_pred"][i] + mp["l_to_r"][i, ..., :3])
    pred_r2l_pc = _to_np(outputs["right_map_pred_in_left_frame"][i] + mp["r_to_l"][i, ..., :3])

    # Apply motion post-processing if provided
    if motion_depth_post_mode is not None:
        pred_l2m_pc = _to_np(reg_dense_depth(torch.tensor(pred_l2m_pc), motion_depth_post_mode))
        pred_r2m_pc = _to_np(reg_dense_depth(torch.tensor(pred_r2m_pc), motion_depth_post_mode))
        pred_l2r_pc = _to_np(reg_dense_depth(torch.tensor(pred_l2r_pc), motion_depth_post_mode))
        pred_r2l_pc = _to_np(reg_dense_depth(torch.tensor(pred_r2l_pc), motion_depth_post_mode))

    # ---- Depth/Disparity images (LEFT-frame Z) ----
    gt_left_depth,  gt_left_disp  = make_depth_and_disp_imgs(gt_left_pc,  valid_left)
    gt_right_depth, gt_right_disp = make_depth_and_disp_imgs(gt_right_pc, valid_right)
    pr_left_depth,  pr_left_disp  = make_depth_and_disp_imgs(pred_left_pc,  valid_left)
    pr_right_depth, pr_right_disp = make_depth_and_disp_imgs(pred_right_pc, valid_right)

    gt_l2m_depth, gt_l2m_disp = make_depth_and_disp_imgs(gt_l2m_pc, v_l2m)
    gt_r2m_depth, gt_r2m_disp = make_depth_and_disp_imgs(gt_r2m_pc, v_r2m)
    gt_l2r_depth, gt_l2r_disp = make_depth_and_disp_imgs(gt_l2r_pc, v_l2r)
    gt_r2l_depth, gt_r2l_disp = make_depth_and_disp_imgs(gt_r2l_pc, v_r2l)

    pr_l2m_depth, pr_l2m_disp = make_depth_and_disp_imgs(pred_l2m_pc, v_l2m)
    pr_r2m_depth, pr_r2m_disp = make_depth_and_disp_imgs(pred_r2m_pc, v_r2m)
    pr_l2r_depth, pr_l2r_disp = make_depth_and_disp_imgs(pred_l2r_pc, v_l2r)
    pr_r2l_depth, pr_r2l_disp = make_depth_and_disp_imgs(pred_r2l_pc, v_r2l)

    # ---- Motion line overlays (draw BOTH endpoints in SAME plane) ----
    # Left canvas (project to LEFT camera)
    gt_l2m_lines_left  = draw_motion(left_img, gt_left_pc,  gt_l2m_pc,  v_l2m, left_cam, left_cam)
    gt_l2r_lines_left  = draw_motion(left_img, gt_left_pc,  gt_l2r_pc,  v_l2r, left_cam, left_cam)
    pr_l2m_lines_left  = draw_motion(left_img, pred_left_pc, pred_l2m_pc, v_l2m, left_cam, left_cam)
    pr_l2r_lines_left  = draw_motion(left_img, pred_left_pc, pred_l2r_pc, v_l2r, left_cam, left_cam)

    # Right canvas (project to RIGHT camera)
    gt_r2m_lines_right = draw_motion(right_img, gt_right_pc, gt_r2m_pc, v_r2m, left_cam, right_cam)
    gt_r2l_lines_right = draw_motion(right_img, gt_right_pc, gt_r2l_pc, v_r2l, left_cam, right_cam)
    pr_r2m_lines_right = draw_motion(right_img, pred_right_pc, pred_r2m_pc, v_r2m, left_cam, right_cam)
    pr_r2l_lines_right = draw_motion(right_img, pred_right_pc, pred_r2l_pc, v_r2l, left_cam, right_cam)

    # ---- Confidences (optional) ----
    def _maybe_squeeze(hw_or_hwc):
        if hw_or_hwc is None:
            return None
        a = hw_or_hwc
        return a[..., 0] if (a.ndim == 3 and a.shape[-1] == 1) else a

    conf_left  = _maybe_squeeze(outputs.get("left_map_pred_conf", None))
    conf_right = _maybe_squeeze(outputs.get("right_map_pred_conf", None))
    if conf_left is not None:
        conf_left  = _to_np(conf_left[i])
    if conf_right is not None:
        conf_right = _to_np(conf_right[i])

    def _pred_conf(key):
        arr = outputs["motion_pred"].get(f"{key}_conf")
        if arr is None:
            return None
        return _to_np(_maybe_squeeze(arr[i]))

    conf_l2m = _pred_conf("l_to_t0")
    conf_r2m = _pred_conf("r_to_t0")
    conf_l2r = _pred_conf("l_to_r")
    conf_r2l = _pred_conf("r_to_l")

    conf_imgs = {}
    if conf_left is not None:
        conf_imgs["conf_left"]  = conf_to_gray_img(conf_left,  valid_left)
    if conf_right is not None:
        conf_imgs["conf_right"] = conf_to_gray_img(conf_right, valid_right)
    if conf_l2m is not None:
        conf_imgs["conf_l2m"] = conf_to_gray_img(conf_l2m, v_l2m)
    if conf_r2m is not None:
        conf_imgs["conf_r2m"] = conf_to_gray_img(conf_r2m, v_r2m)
    if conf_l2r is not None:
        conf_imgs["conf_l2r"] = conf_to_gray_img(conf_l2r, v_l2r)
    if conf_r2l is not None:
        conf_imgs["conf_r2l"] = conf_to_gray_img(conf_r2l, v_r2l)

    # ---- Log to Weights & Biases ----
    base = f"{base_name}_i{i}"
    log = {
        f"{base}/left_img/input":  wandb.Image(left_img,  caption="left input"),
        f"{base}/mid_img/input":   wandb.Image(mid_img,   caption="mid input"),
        f"{base}/right_img/input": wandb.Image(right_img, caption="right input"),

        # Base depth/disp (LEFT-frame Z)
        f"{base}/left_depth/gt":   wandb.Image(gt_left_depth,  caption="GT Left Depth (left-frame)"),
        f"{base}/right_depth/gt":  wandb.Image(gt_right_depth, caption="GT Right Depth (left-frame)"),
        f"{base}/left_depth/pred": wandb.Image(pr_left_depth,  caption="Pred Left Depth (left-frame)"),
        f"{base}/right_depth/pred":wandb.Image(pr_right_depth, caption="Pred Right Depth (left-frame)"),

        f"{base}/left_disp/gt":    wandb.Image(gt_left_disp,  caption="GT Left Disp (1/z left-frame)"),
        f"{base}/right_disp/gt":   wandb.Image(gt_right_disp, caption="GT Right Disp (1/z left-frame)"),
        f"{base}/left_disp/pred":  wandb.Image(pr_left_disp,  caption="Pred Left Disp (1/z left-frame)"),
        f"{base}/right_disp/pred": wandb.Image(pr_right_disp, caption="Pred Right Disp (1/z left-frame)"),

        # Next depths/disps (LEFT-frame Z)
        f"{base}/l2m_depth/gt":    wandb.Image(gt_l2m_depth, caption="GT L2M Depth (left-frame)"),
        f"{base}/r2m_depth/gt":    wandb.Image(gt_r2m_depth, caption="GT R2M Depth (left-frame)"),
        f"{base}/l2r_depth/gt":    wandb.Image(gt_l2r_depth, caption="GT L2R Depth (left-frame)"),
        f"{base}/r2l_depth/gt":    wandb.Image(gt_r2l_depth, caption="GT R2L Depth (left-frame)"),

        f"{base}/l2m_depth/pred":  wandb.Image(pr_l2m_depth, caption="Pred L2M Depth (left-frame)"),
        f"{base}/r2m_depth/pred":  wandb.Image(pr_r2m_depth, caption="Pred R2M Depth (left-frame)"),
        f"{base}/l2r_depth/pred":  wandb.Image(pr_l2r_depth, caption="Pred L2R Depth (left-frame)"),
        f"{base}/r2l_depth/pred":  wandb.Image(pr_r2l_depth, caption="Pred R2L Depth (left-frame)"),

        f"{base}/l2m_disp/gt":     wandb.Image(gt_l2m_disp, caption="GT L2M Disp (1/z left-frame)"),
        f"{base}/r2m_disp/gt":     wandb.Image(gt_r2m_disp, caption="GT R2M Disp (1/z left-frame)"),
        f"{base}/l2r_disp/gt":     wandb.Image(gt_l2r_disp, caption="GT L2R Disp (1/z left-frame)"),
        f"{base}/r2l_disp/gt":     wandb.Image(gt_r2l_disp, caption="GT R2L Disp (1/z left-frame)"),

        f"{base}/l2m_disp/pred":   wandb.Image(pr_l2m_disp, caption="Pred L2M Disp (1/z left-frame)"),
        f"{base}/r2m_disp/pred":   wandb.Image(pr_r2m_disp, caption="Pred R2M Disp (1/z left-frame)"),
        f"{base}/l2r_disp/pred":   wandb.Image(pr_l2r_disp, caption="Pred L2R Disp (1/z left-frame)"),
        f"{base}/r2l_disp/pred":   wandb.Image(pr_r2l_disp, caption="Pred R2L Disp (1/z left-frame)"),

        # Motion overlays (same-plane projections)
        f"{base}/l2m_motion/gt_left_canvas":  wandb.Image(gt_l2m_lines_left,  caption="GT L2M (drawn on LEFT plane)"),
        f"{base}/l2r_motion/gt_left_canvas":  wandb.Image(gt_l2r_lines_left,  caption="GT L2R (drawn on LEFT plane)"),
        f"{base}/l2m_motion/pred_left_canvas": wandb.Image(pr_l2m_lines_left, caption="Pred L2M (drawn on LEFT plane)"),
        f"{base}/l2r_motion/pred_left_canvas": wandb.Image(pr_l2r_lines_left, caption="Pred L2R (drawn on LEFT plane)"),

        f"{base}/r2m_motion/gt_right_canvas": wandb.Image(gt_r2m_lines_right, caption="GT R2M (drawn on RIGHT plane)"),
        f"{base}/r2l_motion/gt_right_canvas": wandb.Image(gt_r2l_lines_right, caption="GT R2L (drawn on RIGHT plane)"),
        f"{base}/r2m_motion/pred_right_canvas":wandb.Image(pr_r2m_lines_right, caption="Pred R2M (drawn on RIGHT plane)"),
        f"{base}/r2l_motion/pred_right_canvas":wandb.Image(pr_r2l_lines_right, caption="Pred R2L (drawn on RIGHT plane)"),
    }

    for k, img in conf_imgs.items():
        log[f"{base}/conf/{k}"] = wandb.Image(img, caption=k)

    wandb.log(log, commit=True)
