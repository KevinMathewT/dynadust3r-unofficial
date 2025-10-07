#!/usr/bin/env python
# coding: utf-8
"""
Author: Kevin Mathew T
Date: 2025-09-29
LinkedIn: https://www.linkedin.com/in/kevinmathewt/

Purpose:
- Run DynaDUSt3R on a folder of two-frame GIF "paper examples" (frame 0=left, frame 1=right)
- Save basic visualization outputs (depth/confidence) per GIF
- Stream a short time sequence to Rerun using utils.rerun_viz.visualize_sequence_from_pms

Usage:
    poetry run python -m infer.infer_on_paper_examples infer.device=auto \
        infer.input_dir=/path/to/gifs infer.output_dir=/path/to/out infer.checkpoint_path=/path/to.ckpt
"""

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from models import get_model
from models.dust3r.utils.heads.postprocess import reg_dense_depth
from models.dust3r.dyna_dust3r import DynaDUSt3R
from utils.rerun_viz import visualize_sequence_from_pms
from utils.viz import make_depth_and_disp_imgs, conf_to_gray_img


def _select_device(cfg_device: str) -> torch.device:
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg_device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _load_gif_first_two_frames(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load first two frames from a GIF as RGB uint8 arrays. If only one, duplicate it."""
    with Image.open(path) as im:
        im.seek(0)
        frame0 = im.convert("RGB")
        left = np.array(frame0)
        right = None
        try:
            im.seek(1)
            frame1 = im.convert("RGB")
            right = np.array(frame1)
        except EOFError:
            right = left.copy()
    return left, right


def _resize_to_square(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return img
    pil = Image.fromarray(img)
    pil = pil.resize((size, size), resample=Image.BILINEAR)
    return np.array(pil)


def _to_model_tensor(img_uint8_hw3: np.ndarray, *, normalize_m1p1: bool) -> torch.Tensor:
    """Convert HWC uint8 [0,255] -> CHW float32 in either [0,1] or [-1,1]."""
    t = torch.from_numpy(img_uint8_hw3).to(dtype=torch.float32)
    if t.ndim == 3 and t.shape[-1] == 3:
        t = t.permute(2, 0, 1)
    t = t / 255.0
    if normalize_m1p1:
        t = (t - 0.5) / 0.5
    return t


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _save_basic_outputs(out_dir: Path,
                        base_name: str,
                        pred_left_hw3: np.ndarray,
                        pred_right_hw3: np.ndarray,
                        conf_left_hw1: np.ndarray | None,
                        conf_right_hw1: np.ndarray | None):
    """Save depth and confidence images for left/right predictions."""
    # Depth/disp visualizations from predicted PCs (LEFT-frame Z)
    valid_all_left = np.ones(pred_left_hw3.shape[:2], dtype=bool)
    valid_all_right = np.ones(pred_right_hw3.shape[:2], dtype=bool)
    left_depth_rgb, _ = make_depth_and_disp_imgs(pred_left_hw3, valid_all_left)
    right_depth_rgb, _ = make_depth_and_disp_imgs(pred_right_hw3, valid_all_right)

    _ensure_dir(out_dir)
    # Save depth
    Image.fromarray(left_depth_rgb).save(out_dir / f"{base_name}_left_depth_pred.png")
    Image.fromarray(right_depth_rgb).save(out_dir / f"{base_name}_right_depth_pred.png")

    # Save confidence if available
    if conf_left_hw1 is not None:
        conf_left_img = conf_to_gray_img(conf_left_hw1, valid_all_left)
        if conf_left_img is not None:
            Image.fromarray(conf_left_img).save(out_dir / f"{base_name}_left_conf.png")
    if conf_right_hw1 is not None:
        conf_right_img = conf_to_gray_img(conf_right_hw1, valid_all_right)
        if conf_right_img is not None:
            Image.fromarray(conf_right_img).save(out_dir / f"{base_name}_right_conf.png")


def _load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    if ckpt_path is None or (not ckpt_path.exists()):
        print("[warn] checkpoint path not provided or not found; running with model as-is")
        return
    print(f"loading checkpoint from {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = None
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        else:
            # might already be a state dict
            state_keys = list(ckpt.keys())
            if state_keys and isinstance(ckpt[state_keys[0]], (torch.Tensor, np.ndarray)):
                state_dict = ckpt
    else:
        state_dict = ckpt

    if state_dict is None:
        print("[warn] could not resolve state_dict from checkpoint; skipping load")
        return model

    # Strip potential DistributedDataParallel prefixes
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    # If the checkpoint stored a model config, prefer to rebuild the model with that config
    normalize_m1p1 = True  # default prior (older pipelines used [-1,1])
    try:
        ck_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
        if isinstance(ck_cfg, dict) and "model" in ck_cfg and isinstance(ck_cfg["model"], dict):
            model_cfg = dict(ck_cfg["model"])  # shallow copy
            # remove keys not accepted by constructor
            for bad in ("name", "use_pretrained", "pretrained_link"):
                model_cfg.pop(bad, None)
            # Rebuild model to match checkpoint architecture exactly
            rebuilt = DynaDUSt3R(**model_cfg).to(device)
            rebuilt.eval()
            model = rebuilt
            # Also apply post-process modes from ckpt-config for consistency
            for k in ["depth_post_mode", "motion_depth_post_mode", "depth_mode", "motion_depth_mode", "conf_mode", "motion_conf_mode"]:
                if k in model_cfg and hasattr(model, k):
                    setattr(model, k, tuple(model_cfg[k]) if isinstance(model_cfg[k], list) else model_cfg[k])
                    print(f"[ckpt-config] set model.{k} = {getattr(model, k)}")
        # Decide normalization based on data loader used during training
        if isinstance(ck_cfg, dict) and "data" in ck_cfg:
            # Stereo4D streamer feeds images in [0,1] (no mean/std) according to _to_chw_float
            loader_name = str(ck_cfg.get("data", {}).get("loader", "")).lower()
            if loader_name == "stereo4d":
                normalize_m1p1 = False
                print("[infer] using [0,1] normalization to match training loader")
    except Exception as e:
        print(f"[ckpt-config] warning: failed to rebuild/apply checkpoint config: {e}")

    # Prefer strict load for exact arch match; fallback to non-strict
    try:
        model.load_state_dict(state_dict, strict=True)
        print("[ckpt] loaded with strict=True")
    except Exception as e:
        print(f"[ckpt] strict load failed: {e}; retrying strict=False")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")

    # Attach a flag so caller can normalize inputs consistently
    setattr(model, "_normalize_m1p1", normalize_m1p1)
    return model


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig):
    # Resolve device
    device = _select_device(str(config.infer.device))
    print(f"Using device: {device}")

    # Build model and load checkpoint
    model = get_model(config, device)
    model.to(device)
    model.eval()

    ckpt_path = None
    if "checkpoint_path" in config.infer:
        path_str = str(config.infer.checkpoint_path)
        if path_str and Path(path_str).is_file():
            ckpt_path = Path(path_str)
    if ckpt_path is not None:
        model = _load_checkpoint_into_model(model, ckpt_path, device)
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print("No valid checkpoint file provided; using current model weights")

    # IO setup
    input_dir = Path(str(config.infer.input_dir))
    output_dir = Path(str(config.infer.output_dir))
    _ensure_dir(output_dir)

    gif_glob = str(config.infer.gif_glob)
    gif_paths = sorted(input_dir.glob(gif_glob))
    if not gif_paths:
        print(f"no GIFs found in {input_dir} matching {gif_glob}")
        return

    if bool(getattr(config, "debug", False)):
        gif_paths = gif_paths[:1]
        print("[debug] limiting to first GIF")

    # Query times: include 0 and 1 and a mid time (0.5)
    # Important: place 0.5 at index 0 so outputs use key 'l_to_t0' for mid
    query_times_for_model = torch.tensor([0.5, 1.0, 0.0], dtype=torch.float32, device=device)

    # Target input size
    target_size = int(config.data.size)

    for gif_path in gif_paths:
        try:
            left, right = _load_gif_first_two_frames(gif_path)
        except Exception as e:
            print(f"error reading {gif_path}: {e}")
            continue

        # Resize to model size
        left_r = _resize_to_square(left, target_size)
        right_r = _resize_to_square(right, target_size)

        # Build model batch (normalize according to training loader)
        normalize_m1p1 = bool(getattr(model, "_normalize_m1p1", True))
        left_tensor = _to_model_tensor(left_r, normalize_m1p1=normalize_m1p1).unsqueeze(0).to(device)   # (1,3,H,W)
        right_tensor = _to_model_tensor(right_r, normalize_m1p1=normalize_m1p1).unsqueeze(0).to(device) # (1,3,H,W)

        batch = {
            "left_image": left_tensor,
            "right_image": right_tensor,
            "query_times": query_times_for_model,  # (T,)
            # provide instances so is_symmetrized() doesn't crash on None
            "left_instance": [f"{gif_path.stem}_L"],
            "right_instance": [f"{gif_path.stem}_R"],
        }

        # Forward
        with torch.inference_mode():
            outputs = model(batch)

        # Extract predictions (CPU numpy for saving)
        left_pred = outputs["left_map_pred"][0].detach().cpu().numpy()              # (H,W,3)
        right_pred = outputs["right_map_pred_in_left_frame"][0].detach().cpu().numpy()  # (H,W,3)

        conf_left = outputs.get("left_map_pred_conf", None)
        conf_right = outputs.get("right_map_pred_conf", None)
        if conf_left is not None:
            # ensure (H,W)
            c = conf_left[0].detach().cpu().numpy()
            conf_left_np = c[..., 0] if c.ndim == 3 and c.shape[-1] == 1 else c
        else:
            conf_left_np = None
        if conf_right is not None:
            c = conf_right[0].detach().cpu().numpy()
            conf_right_np = c[..., 0] if c.ndim == 3 and c.shape[-1] == 1 else c
        else:
            conf_right_np = None

        # Optional: apply post depth mapping (matches training visualization)
        # Here depth_post_mode is often linear so identity, but keep for consistency
        left_pred_pp = reg_dense_depth(outputs["left_map_pred"][0], model.depth_post_mode).detach().cpu().numpy()
        right_pred_pp = reg_dense_depth(outputs["right_map_pred_in_left_frame"][0], model.depth_post_mode).detach().cpu().numpy()

        # Save basic outputs
        base = gif_path.stem
        out_dir = output_dir / base
        _save_basic_outputs(out_dir, base, left_pred_pp, right_pred_pp, conf_left_np, conf_right_np)

        # Build a short sequence for Rerun using left-frame PCs and motion
        # Chronological times: 0.0, 0.5, 1.0
        # We requested query_times in order [0.5, 1.0, 0.0], so:
        #   left->mid displacement:  outputs['motion_pred']['l_to_t0'][0]
        #   left->right displacement: outputs['motion_pred']['l_to_r'][0]
        l_to_mid = outputs["motion_pred"].get("l_to_t0", None)
        l_to_r = outputs["motion_pred"].get("l_to_r", None)

        if l_to_mid is not None and l_to_r is not None:
            l_to_mid_np = l_to_mid[0].detach().cpu().numpy()[..., :3]
            l_to_r_np = l_to_r[0].detach().cpu().numpy()[..., :3]

            # Sequence of point maps in LEFT frame
            pm_t0 = left_pred_pp  # (H,W,3)
            pm_t05 = left_pred_pp + l_to_mid_np
            pm_t1 = left_pred_pp + l_to_r_np
            pms_seq = [pm_t0, pm_t05, pm_t1]

            # Motion between successive pms: [t0->t05, t05->t1]
            mm_0 = l_to_mid_np
            mm_1 = l_to_r_np - l_to_mid_np
            motion_seq = np.stack([mm_0, mm_1], axis=0)  # (2,H,W,3)

            # Image sequence for coloring: use left for t0, None for mid, right for t1
            image_seq = [left_r, left_r, left_r]

            # Stream to rerun
            try:
                visualize_sequence_from_pms(
                    pms_seq, motion_seq, image_seq=image_seq, name=f"paper_seq_{base}")
            except Exception as e:
                print(f"[rerun] failed for {base}: {e}")
        else:
            print(f"[warn] motion predictions missing for {gif_path.name}; skipping rerun sequence")

        print(f"Processed {gif_path}")


if __name__ == "__main__":
    main()


