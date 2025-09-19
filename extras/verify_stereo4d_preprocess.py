#!/usr/bin/env python3
"""
Verify Stereo4D webdataset shards match raw files for a set of sequence IDs.

For each sequence id, this script:
- Locates which split (train/test) contains the id (prefers WDS index membership, otherwise raw presence)
- Loads the sample from the webdataset (video.mp4, ann.npz, intr.npy)
- Loads the raw sources from disk (mp4, npz), and recomputes intrinsics K from raw width and configured hfov
- Compares for exact equality:
  - video bytes (WDS vs raw)
  - ann bytes (WDS vs raw)
  - intrinsics array (WDS vs recomputed from raw width)

If IDS_TO_CHECK is empty, the script randomly samples up to NUM_AUTO_IDS IDs
from the raw dataset across both train and test (requires both mp4 and npz present).
"""

from __future__ import annotations

import os
import io
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import numpy as np
import cv2
from tqdm import tqdm
import hydra
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Capitalized constant of IDs to verify. Fill with specific ids if desired.
# Example IDs look like: "ytid_timestamp" (e.g., "QYb1m94fAX8_1709").
IDS_TO_CHECK: List[str] = []

# Auto-sample count if IDS_TO_CHECK is empty (can override via env: VERIFY_S4D_N)
NUM_AUTO_IDS: int = int(os.environ.get("VERIFY_S4D_N", "100"))

# Splits to search for IDs
SPLITS_TO_CHECK: List[str] = ["train", "test"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_bool_env(val: Optional[str], default: bool) -> bool:
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _get_decord_ctx():
    try:
        import torch
        if torch.cuda.is_available():
            from decord import gpu as _gpu
            return _gpu(torch.cuda.current_device())
    except Exception:
        pass
    from decord import cpu as _cpu
    return _cpu(0)


def _probe_dims_cv(path: str | os.PathLike) -> Tuple[int, int] | None:
    try:
        cap = cv2.VideoCapture(str(path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            return w, h
    except Exception:
        return None
    return None


def _probe_dims_decord(path: str | os.PathLike) -> Tuple[int, int] | None:
    try:
        from decord import VideoReader
        vr = VideoReader(str(path), ctx=_get_decord_ctx())
        f0 = vr[0]
        return int(f0.shape[1]), int(f0.shape[0])
    except Exception:
        return None


def _probe_dims_av(path: str | os.PathLike) -> Tuple[int, int] | None:
    try:
        import av  # type: ignore
        with av.open(str(path)) as c:
            for st in c.streams:
                if st.type == 'video':
                    w = int(st.codec_context.width)
                    h = int(st.codec_context.height)
                    if w > 0 and h > 0:
                        return w, h
    except Exception:
        return None
    return None


def probe_width_height(path: str | os.PathLike) -> Tuple[int, int]:
    dims = _probe_dims_cv(path) or _probe_dims_decord(path) or _probe_dims_av(path)
    if dims is None:
        raise RuntimeError(f"Failed to probe video dimensions: {path}")
    return dims


def intrinsic_K(width: int, hfov_deg: float) -> np.ndarray:
    import math
    fx = width * 0.5 / math.tan(math.radians(hfov_deg) * 0.5)
    return np.array([[fx, 0, width / 2], [0, fx, width / 2], [0, 0, 1]], np.float32)


def _bytes_from_sample_field(sample: dict, key: str) -> Optional[bytes]:
    val = sample.get(key)
    if val is None:
        return None
    return val.read() if hasattr(val, "read") else val


def _load_wids_split(wds_split_dir: Path) -> Tuple[Optional[object], Dict[str, int]]:
    idx_json = wds_split_dir / "stereo4d-idx.json"
    keymap_json = wds_split_dir / "key_to_idx.json"
    if not idx_json.exists() or not keymap_json.exists():
        return None, {}
    import wids
    idx_ds = wids.ShardListDataset(str(idx_json), transformations=[])
    with open(keymap_json) as f:
        key_to_idx = json.load(f)
    return idx_ds, key_to_idx


def _find_split_for_id(seq_id: str, split_to_maps: Dict[str, Dict[str, int]], raw_roots: Dict[str, Tuple[Path, Path]]) -> Optional[str]:
    for sp, mp in split_to_maps.items():
        if seq_id in mp:
            return sp
    for sp, (lefteye_dir, ann_dir) in raw_roots.items():
        mp4 = lefteye_dir / f"{seq_id}-left_rectified.mp4"
        npz = ann_dir / f"{seq_id}.npz"
        if mp4.exists() and npz.exists():
            return sp
    return None


def _ensure_wids_cache(cfg: DictConfig):
    os.environ.setdefault("TMPDIR", "/scratch/km6748/tmp")
    os.environ.setdefault("TMP", "/scratch/km6748/tmp")
    os.environ.setdefault("TEMP", "/scratch/km6748/tmp")
    cache_dir = str(getattr(cfg.dataset.stereo4d, "cache", os.path.join(os.environ.get("TMP", "/tmp"), "_wids_cache")))
    os.environ.setdefault("WIDS_CACHE", cache_dir)
    Path(os.environ["WIDS_CACHE"]).mkdir(parents=True, exist_ok=True)


def _collect_ids_from_raw(raw_video_root: Path, raw_ann_root: Path, splits: List[str], max_n: int) -> List[str]:
    """Scan raw train/test for ids that have both mp4 and npz, then sample.

    Returns up to max_n unique ids randomly sampled across the provided splits.
    """
    ids: List[str] = []
    for sp in splits:
        vdir = raw_video_root / sp
        adir = raw_ann_root / sp
        if not vdir.exists() or not adir.exists():
            continue
        # Gather union of ids with both files present
        # mp4 pattern: <id>-left_rectified.mp4 ; npz pattern: <id>.npz
        mp4_ids = []
        try:
            for p in vdir.glob("*-left_rectified.mp4"):
                mp4_ids.append(p.name[:-len("-left_rectified.mp4")])
        except Exception:
            pass
        mp4_id_set = set(mp4_ids)
        try:
            for p in adir.glob("*.npz"):
                seq = p.stem
                if seq in mp4_id_set:
                    ids.append(seq)
        except Exception:
            pass
    # Deduplicate and sample
    ids = sorted(set(ids))
    if len(ids) <= max_n:
        random.seed(0)
        random.shuffle(ids)
        return ids
    random.seed(0)
    return random.sample(ids, max_n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    _ensure_wids_cache(cfg)

    wds_root = Path(cfg.dataset.stereo4d.wds_dir)
    raw_video_root = Path(cfg.dataset.stereo4d.lefteye_dir)
    raw_ann_root = Path(cfg.dataset.stereo4d.path)
    hfov = float(cfg.dataset.stereo4d.hfov)

    split_to_idx = {}
    split_to_map: Dict[str, Dict[str, int]] = {}
    for sp in SPLITS_TO_CHECK:
        idx_ds, keymap = _load_wids_split(wds_root / sp)
        split_to_idx[sp] = idx_ds
        split_to_map[sp] = keymap

    raw_roots: Dict[str, Tuple[Path, Path]] = {
        sp: (raw_video_root / sp, raw_ann_root / sp) for sp in SPLITS_TO_CHECK
    }

    ids: List[str] = list(IDS_TO_CHECK)
    if not ids:
        ids = _collect_ids_from_raw(raw_video_root, raw_ann_root, SPLITS_TO_CHECK, NUM_AUTO_IDS)
        if not ids:
            raise SystemExit("No IDs available to verify from raw dataset (provide IDS_TO_CHECK or ensure raw train/test exist).")

    results: List[Tuple[str, str]] = []
    ok_count = 0
    fail_count = 0

    for seq_id in tqdm(ids, desc="verify", dynamic_ncols=True):
        split = _find_split_for_id(seq_id, split_to_map, raw_roots)
        if split is None:
            results.append((seq_id, "missing: not found in WDS indices or raw files"))
            fail_count += 1
            continue

        idx_ds = split_to_idx.get(split)
        keymap = split_to_map.get(split) or {}
        raw_v_dir, raw_a_dir = raw_roots[split]
        mp4_path = raw_v_dir / f"{seq_id}-left_rectified.mp4"
        npz_path = raw_a_dir / f"{seq_id}.npz"

        if seq_id not in keymap or idx_ds is None:
            results.append((seq_id, f"missing in WDS index for split={split}"))
            fail_count += 1
            continue

        try:
            sample = idx_ds[keymap[seq_id]]  # type: ignore[index]
        except Exception as e:
            results.append((seq_id, f"wds read error: {type(e).__name__}: {e}"))
            fail_count += 1
            continue

        vbytes_wds = _bytes_from_sample_field(sample, ".video.mp4") or _bytes_from_sample_field(sample, "video.mp4")
        abytes_wds = _bytes_from_sample_field(sample, ".ann.npz") or _bytes_from_sample_field(sample, "ann.npz")
        intr_wds = _bytes_from_sample_field(sample, ".intr.npy") or _bytes_from_sample_field(sample, "intr.npy")

        if not mp4_path.exists() or not npz_path.exists():
            results.append((seq_id, f"missing raw file(s) mp4={mp4_path.exists()} npz={npz_path.exists()}"))
            fail_count += 1
            continue

        try:
            with open(mp4_path, "rb") as f:
                vbytes_raw = f.read()
            with open(npz_path, "rb") as f:
                abytes_raw = f.read()
        except Exception as e:
            results.append((seq_id, f"raw read error: {type(e).__name__}: {e}"))
            fail_count += 1
            continue

        if vbytes_wds is None or abytes_wds is None or intr_wds is None:
            results.append((seq_id, "wds sample missing one or more blobs (video/ann/intr)"))
            fail_count += 1
            continue

        mismatches: List[str] = []
        if vbytes_wds != vbytes_raw:
            mismatches.append("video.mp4")
        if abytes_wds != abytes_raw:
            mismatches.append("ann.npz")

        try:
            intr_arr_wds = np.load(io.BytesIO(intr_wds), allow_pickle=False)
        except Exception as e:
            results.append((seq_id, f"intr read error: {type(e).__name__}: {e}"))
            fail_count += 1
            continue

        try:
            w, _h = probe_width_height(str(mp4_path))
            intr_arr_ref = intrinsic_K(int(w), hfov).astype(np.float32)
        except Exception as e:
            results.append((seq_id, f"intr compute error: {type(e).__name__}: {e}"))
            fail_count += 1
            continue

        if not np.array_equal(intr_arr_wds, intr_arr_ref):
            mismatches.append("intr.npy")

        if mismatches:
            results.append((seq_id, f"mismatch: {', '.join(mismatches)}"))
            fail_count += 1
        else:
            results.append((seq_id, "ok"))
            ok_count += 1

    print(f"\nVerified: {ok_count} ok  |  {fail_count} failed  |  total={len(ids)}")
    if fail_count > 0:
        print("\nFailures:")
        for seq_id, msg in results:
            if msg != "ok":
                print(f"  {seq_id}: {msg}")


if __name__ == "__main__":
    main()


