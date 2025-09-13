#!/usr/bin/env python
# coding: utf-8
"""
Infer on paper GIF examples and save visualizations per GIF under a results folder.

Usage:
    poetry run python -m infer.infer_on_paper_examples
"""

import os
import glob
from pathlib import Path
from typing import Optional

import torch
import imageio.v3 as iio
import numpy as np

import hydra
from omegaconf import DictConfig

from models import get_model

# ---- import your viz function
from utils.viz import save_visualizations


# ------------------ helpers ------------------

def _intrinsic_K(width: int, height: int, hfov_deg: float) -> np.ndarray:
    fx = width * 0.5 / np.tan(np.deg2rad(hfov_deg) * 0.5)
    K = np.array([[fx, 0, width / 2],
                  [0, fx, height / 2],
                  [0,  0,          1]], dtype=np.float32)
    return K


def _ensure_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Ensure (H, W, 3) uint8 RGB. Handles RGBA, grayscale, and float inputs.
    """
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        pass
    else:
        raise ValueError(f"Unexpected image shape {arr.shape}; need HxWx{{1,3,4}}")

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0 + 0.5).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def _to_chw_norm_uint8_to_float(img_hw3: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(img_hw3).to(device=device, dtype=torch.float32)  # (H,W,3)
    t = t.permute(2, 0, 1) / 255.0                                       # (3,H,W)
    t = (t - 0.5) / 0.5
    return t


def _fmt_t(t: float) -> str:
    s = f"{t:.6f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def _robust_load_state_dict(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    new_sd = {}
    for k, v in sd.items():
        new_sd[k[7:]] = v if k.startswith("module.") else v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (showing up to 10): {missing[:10]}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (showing up to 10): {unexpected[:10]}")


def _get_hfov_deg(cfg: DictConfig, default_deg: float = 90.0) -> float:
    hfov: Optional[float] = None
    try:
        if "dataset" in cfg and "stereo4d" in cfg.dataset and "hfov" in cfg.dataset.stereo4d:
            hfov = float(cfg.dataset.stereo4d.hfov)
    except Exception as e:
        print(f"[warn] could not parse dataset.stereo4d.hfov from config: {e}")
    if hfov is None:
        print(f"[info] using default HFOV={default_deg}° (no dataset.stereo4d.hfov in config)")
        hfov = default_deg
    return hfov


def _images_same_size_or_resize(frames: np.ndarray) -> np.ndarray:
    """
    Ensures all frames share identical HxW by resizing to the first frame's size if needed.
    Nearest-neighbor to avoid heavy deps.
    """
    T = frames.shape[0]
    h0, w0 = frames[0].shape[:2]
    out = []
    for t in range(T):
        img = frames[t]
        if img.shape[:2] != (h0, w0):
            y = np.linspace(0, img.shape[0]-1, h0).astype(np.int32)
            x = np.linspace(0, img.shape[1]-1, w0).astype(np.int32)
            img = img[y][:, x]
        out.append(img)
    return np.stack(out, axis=0)


# ------------------ main ------------------

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    if save_visualizations is None:
        raise ImportError("save_visualizations(...) could not be imported.")
    
    # Clear output directory if it exists
    output_dir = Path(cfg.infer.output_dir)
    if output_dir.exists():
        import shutil
        print(f"Clearing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
        
    # Resolve device
    dev_cfg = str(cfg.infer.device).lower()
    device = torch.device("cuda" if (dev_cfg == "cuda" and torch.cuda.is_available()) else "cpu")

    # Build model and load checkpoint
    model = get_model(cfg, device)
    model.to(device).eval()
    _robust_load_state_dict(model, cfg.infer.checkpoint_path, device)

    input_dir = Path(cfg.infer.input_dir)
    output_dir = Path(cfg.infer.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover GIFs
    pattern = str(input_dir / cfg.infer.gif_glob)
    gif_paths = sorted(glob.glob(pattern))
    if not gif_paths:
        print(f"No GIFs matched: {pattern}")
        return

    # HFOV
    hfov = _get_hfov_deg(cfg, default_deg=90.0)

    # Big disclaimer: extrinsics are identity
    print(
        "[note] Using IDENTITY extrinsics for left/mid/right. "
        "Cross-view overlays (e.g., drawing on the right/mid images) are illustrative only."
    )

    # Read filtering knobs
    conf_pct = float(getattr(cfg.infer, "motion_conf_top_percentile", 0.0) or 0.0)
    conf_pct = max(0.0, min(1.0, conf_pct))          # 0..1 (e.g., 0.25 => top 25%)
    pct = float(getattr(cfg.infer, "mask_motion_top_percentile", 0.0) or 0.0)
    pct = max(0.0, min(1.0, pct))                    # 0..1

    for gif_path in gif_paths:
        gif_path = Path(gif_path)
        gif_stem = gif_path.stem
        per_gif_dir = output_dir / gif_stem
        per_gif_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {gif_path} -> {per_gif_dir}")

        try:
            # (T,H,W,C?) possibly variable modes
            raw = iio.imread(gif_path)
            # normalize to (T,H,W,3) uint8 RGB
            if raw.ndim == 3:
                frames = _ensure_rgb_uint8(raw)[None]
            elif raw.ndim == 4:
                frames = np.stack([_ensure_rgb_uint8(f) for f in raw], axis=0)
            else:
                raise ValueError(f"Unexpected GIF array shape: {raw.shape}")
            frames = _images_same_size_or_resize(frames)
            T, H, W = frames.shape[0], frames.shape[1], frames.shape[2]

            # Indices
            if T == 1:
                left_idx = right_idx = mid_idx = 0
            else:
                left_idx = 0
                right_idx = T - 1
                mid_idx = (left_idx + right_idx) // 2

            # To tensors
            left_img  = _to_chw_norm_uint8_to_float(frames[left_idx], device)
            mid_img   = _to_chw_norm_uint8_to_float(frames[mid_idx],  device)
            right_img = _to_chw_norm_uint8_to_float(frames[right_idx], device)

            left_b  = left_img.unsqueeze(0)
            mid_b   = mid_img.unsqueeze(0)    # only for viz panels
            right_b = right_img.unsqueeze(0)

            # Query times
            denom = max(right_idx - left_idx, 1)
            t_mid_raw = float((mid_idx - left_idx) / denom)
            t_mid = t_mid_raw if 0.0 < t_mid_raw < 1.0 else 0.5
            query_times = torch.tensor([[t_mid, 1.0, 0.0]], dtype=torch.float32, device=device)

            # Cameras (shared K; identity extrinsics)
            K_np = _intrinsic_K(W, H, hfov)
            K_t  = torch.from_numpy(K_np).to(device)
            EX_t = torch.eye(4, dtype=torch.float32, device=device)
            cam       = (K_t.unsqueeze(0), EX_t.unsqueeze(0))
            cam_mid   = (K_t.unsqueeze(0), EX_t.unsqueeze(0))
            cam_right = (K_t.unsqueeze(0), EX_t.unsqueeze(0))

            # Inference
            batch_infer = {
                "left_image": left_b,
                "right_image": right_b,
                "query_times": query_times,
                "left_instance":  [f"{gif_stem}_left"],
                "right_instance": [f"{gif_stem}_right"],
            }
            with torch.no_grad():
                outputs = model(batch_infer)

            # --- Build viz batch -------------------------------------------------
            print("[note] No GT available. GT depth/disparity panels will show the model’s own point maps.")

            # Map predictions using model's configured postprocess modes
            from models.dust3r.utils.heads.postprocess import reg_dense_depth
            mref = model.module if hasattr(model, "module") else model
            depth_post_mode = getattr(mref, "depth_post_mode", ("exp", -float("inf"), float("inf")))
            motion_depth_post_mode = getattr(mref, "motion_depth_post_mode", ("linear", -float("inf"), float("inf")))

            left_pred  = reg_dense_depth(outputs["left_map_pred"], depth_post_mode).detach()
            right_pred = reg_dense_depth(outputs["right_map_pred_in_left_frame"], depth_post_mode).detach()
            B, h, w, _ = left_pred.shape
            ones = torch.ones((B, h, w, 1), dtype=left_pred.dtype, device=left_pred.device)

            # Compute an estimated mid-time point map from base + predicted motion at mid index
            mp = outputs.get("motion_pred", {})
            tq_mid_idx = 0
            l2m_key = f"l_to_t{tq_mid_idx}"
            r2m_key = f"r_to_t{tq_mid_idx}"
            mid_pred = None
            if l2m_key in mp:
                mid_pred = reg_dense_depth(outputs["left_map_pred"] + mp[l2m_key][..., :3], motion_depth_post_mode).detach()
            elif r2m_key in mp:
                mid_pred = reg_dense_depth(outputs["right_map_pred_in_left_frame"] + mp[r2m_key][..., :3], motion_depth_post_mode).detach()
            else:
                mid_pred = torch.zeros_like(left_pred)

            left_pm  = torch.cat([left_pred,  ones], dim=-1)  # (B,H,W,4)
            right_pm = torch.cat([right_pred, ones], dim=-1)  # (B,H,W,4)
            mid_pm   = torch.cat([mid_pred,  ones], dim=-1)   # (B,H,W,4)

            def _make_mask_for_key(base_key: str) -> torch.Tensor:
                """
                1) confidence gate: keep top 'conf_pct' by confidence (if conf exists).
                   If conf missing or conf_pct <= 0 -> all pass.
                2) among the passed pixels, keep top 'pct' by motion magnitude.
                   If pct <= 0 -> return the confidence gate only.
                Returns (B,H,W,1) float mask in {0,1}.
                """
                # 1) confidence gate (top conf_pct)
                conf = mp.get(f"{base_key}_conf")
                if conf is None or conf_pct <= 0.0:
                    conf_mask = torch.ones((B, h, w), dtype=torch.bool, device=left_pred.device)
                else:
                    c = conf[..., 0]  # (B,H,W)
                    conf_mask = torch.zeros_like(c, dtype=torch.bool)
                    q_conf = 1.0 - conf_pct  # top conf_pct -> quantile at 1 - conf_pct
                    for b in range(B):
                        flat = c[b].view(-1)
                        thr = torch.quantile(flat, q=q_conf)
                        conf_mask[b] = c[b] >= thr

                # 2) magnitude gate within conf_mask
                arr = mp.get(base_key)
                if arr is None:
                    return torch.zeros((B, h, w, 1), dtype=left_pred.dtype, device=left_pred.device)

                if pct <= 0.0:
                    return conf_mask.unsqueeze(-1).to(dtype=left_pred.dtype)

                mag = torch.norm(arr[..., :3], dim=-1)  # (B,H,W)
                final_mask = torch.zeros_like(conf_mask)
                q_mag = 1.0 - pct

                for b in range(B):
                    sel = mag[b][conf_mask[b]]
                    if sel.numel() == 0:
                        continue
                    thr = torch.quantile(sel, q=q_mag)
                    keep = (mag[b] >= thr) & conf_mask[b]
                    final_mask[b] = keep

                return final_mask.unsqueeze(-1).to(dtype=left_pred.dtype)

            # Use DynaDUSt3R's motion key scheme
            k_l2m = l2m_key           # "l_to_t0"
            k_r2m = r2m_key           # "r_to_t0"
            k_l2r = "l_to_r"
            k_r2l = "r_to_l"

            v_l2m = _make_mask_for_key(k_l2m)
            v_r2m = _make_mask_for_key(k_r2m)
            v_l2r = _make_mask_for_key(k_l2r)
            v_r2l = _make_mask_for_key(k_r2l)

            zero4 = torch.zeros((B, h, w, 4), dtype=left_pred.dtype, device=left_pred.device)
            def _zero_with_mask(vmask: torch.Tensor) -> torch.Tensor:
                z = zero4.clone()
                z[..., 3:4] = vmask
                return z

            motion_gt = {
                "l2m": _zero_with_mask(v_l2m),
                "r2m": _zero_with_mask(v_r2m),
                "l2r": _zero_with_mask(v_l2r),
                "r2l": _zero_with_mask(v_r2l),
            }

            viz_batch = {
                "left_pm":   left_pm.cpu(),
                "mid_pm":    mid_pm.cpu(),
                "right_pm":  right_pm.cpu(),
                "motion_gt": {k: v.cpu() for k, v in motion_gt.items()},
                "left_image":  left_b.cpu(),
                "mid_image":   mid_b.cpu(),
                "right_image": right_b.cpu(),
                "query_times": query_times.cpu(),
                "left_instance":  [f"{gif_stem}_left"],
                "right_instance": [f"{gif_stem}_right"],
                "cam":       (K_t.unsqueeze(0).cpu(), EX_t.unsqueeze(0).cpu()),
                "cam_mid":   (K_t.unsqueeze(0).cpu(), EX_t.unsqueeze(0).cpu()),
                "cam_right": (K_t.unsqueeze(0).cpu(), EX_t.unsqueeze(0).cpu()),
            }

            base_name = gif_stem
            save_visualizations(
                viz_batch,
                outputs,
                base_name,
                save_dir=str(per_gif_dir),
                depth_post_mode=depth_post_mode,
                motion_depth_post_mode=motion_depth_post_mode,
            )
            print(f"Saved visualizations to: {per_gif_dir}")

        except Exception as e:
            print(f"[error] Failed on {gif_path}: {e}")


if __name__ == "__main__":
    main()
