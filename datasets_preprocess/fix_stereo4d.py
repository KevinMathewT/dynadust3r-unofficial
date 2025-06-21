#!/usr/bin/env python3
"""
repair_stereo4d_bad_mp4s.py
---------------------------
Fix any missing / corrupt WebDataset samples for the current split.

Adds detailed diagnostics:
  • why each sequence is considered bad
  • why a bad sequence could NOT be regenerated
"""

from __future__ import annotations
import io, json, os
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from decord import VideoReader, cpu
from joblib import Parallel, delayed
from tqdm import tqdm

import webdataset as wds
import wids
import hydra
from omegaconf import DictConfig


# ───────────────────────────── helper functions ──────────────────────────────
def intrinsic_k(w: int, hfov: float) -> np.ndarray:
    fx = w * 0.5 / np.tan(np.radians(hfov) * 0.5)
    return np.array([[fx, 0, w / 2], [0, fx, w / 2], [0, 0, 1]], np.float32)


def vr_ok(blob) -> bool:
    try:
        f = blob if hasattr(blob, "read") else io.BytesIO(blob)
        VideoReader(f, ctx=cpu(0))
        return True
    except Exception:
        return False


# ────────── phase 1: CSV → split-aware candidate list (parallel) ─────────────
def _row_to_seq(row, meta_df, lefteye_dir: Path) -> Optional[str]:
    vid, cid = row["ytid"], row["clipid"]
    ts = meta_df.loc[(meta_df.vid == vid) & (meta_df.clipid == cid), "timestamp"]
    if ts.empty:
        return None
    seq = f"{vid}_{int(ts.values[0])}"
    if (lefteye_dir / f"{seq}-left_rectified.mp4").exists():
        return seq
    return None


# ────────── phase 2: diagnose & build fixed sample (parallel) ────────────────
def _probe_and_fix(
    seq: str,
    idx_json: str,
    map_json: str,
    ann_dir: str,
    lefteye_dir: str,
    hfov: float,
) -> dict:
    """
    Return dict:
        { 'seq': str,
          'sample': dict|None,      # fresh sample if we could repair
          'reason': str|None }      # explanation why seq is bad / skipped
    """
    idx_ds = _probe_and_fix.idx_ds
    key_to_idx = _probe_and_fix.key_to_idx
    if idx_ds is None:  # first call (per worker)
        _probe_and_fix.idx_ds = wids.ShardListDataset(idx_json, transformations=[])
        with open(map_json) as f:
            _probe_and_fix.key_to_idx = json.load(f)
        idx_ds, key_to_idx = _probe_and_fix.idx_ds, _probe_and_fix.key_to_idx

    ann_dir = Path(ann_dir)
    lefteye_dir = Path(lefteye_dir)

    # ------------------------------------------------ existing WDS diagnostics
    def why_bad() -> Optional[str]:
        if seq not in key_to_idx:
            return "missing from index"
        try:
            sample = idx_ds[key_to_idx[seq]]
        except Exception as e:
            return f"tar read error: {e}"
        vid_blob = sample.get(".video.mp4")
        ann_blob = sample.get(".ann.npz")
        if vid_blob is None:
            return "missing .video.mp4 key"
        if ann_blob is None:
            return "missing .ann.npz key"
        if isinstance(vid_blob, (bytes, bytearray, memoryview)) and len(vid_blob) == 0:
            return "video blob is zero bytes"
        if not vr_ok(vid_blob):
            return "video decode failure"
        return None  # looks fine

    reason = why_bad()
    if reason is None:
        return {"seq": seq, "sample": None, "reason": None}  # good

    # ------------------------------------------------ regeneration attempt
    mp4 = lefteye_dir / f"{seq}-left_rectified.mp4"
    npz = ann_dir / f"{seq}.npz"
    if not (mp4.exists() and npz.exists()):
        return {"seq": seq, "sample": None, "reason": f"{reason}; assets missing on disk"}

    try:
        vr = VideoReader(str(mp4), ctx=cpu(0))
        K = intrinsic_k(vr[0].shape[1], hfov)
        buf = io.BytesIO(); np.save(buf, K.astype(np.float32))
        sample = {
            "__key__": seq,
            "video.mp4": mp4.read_bytes(),
            "ann.npz": npz.read_bytes(),
            "intr.npy": buf.getvalue(),
        }
        return {"seq": seq, "sample": sample, "reason": reason}
    except Exception as e:
        return {"seq": seq, "sample": None, "reason": f"{reason}; regen failed ({e})"}


_probe_and_fix.idx_ds = None
_probe_and_fix.key_to_idx = {}


# ────────── phase 3: merge patch shard into index/key map ────────────────────
def merge_index(patch_idx: Path, idx_json: Path, map_json: Path):
    with open(idx_json) as fm, open(patch_idx) as fp:
        main, patch = json.load(fm), json.load(fp)
    main["shards"] += patch["shards"]
    main["samples"] += patch["samples"]
    with open(idx_json, "w") as f:
        json.dump(main, f)
    with open(map_json) as f:
        k2i = json.load(f)
    start = max(k2i.values(), default=-1) + 1
    for i, e in enumerate(patch["samples"]):
        k2i[e["key"]] = start + i
    with open(map_json, "w") as f:
        json.dump(k2i, f)


# ─────────────────────────────────── main ────────────────────────────────────
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    split = cfg.dataset.stereo4d.train_split
    hfov = cfg.dataset.stereo4d.hfov

    # paths
    lefteye_dir = Path(cfg.dataset.stereo4d.lefteye_dir) / split
    ann_dir = Path(cfg.dataset.stereo4d.path) / split
    wds_dir = Path(cfg.dataset.stereo4d.path) / "wds" / split
    idx_json, map_json = wds_dir / "stereo4d-idx.json", wds_dir / "key_to_idx.json"

    # ── phase 1: candidate list
    meta_csv = Path(cfg.dataset.stereo4d.meta_dir) / "stereo4d_id_to_time_and_fov_metadata.csv"
    stats_csv = Path(cfg.dataset.stereo4d.meta_dir) / "stats.csv"
    meta = pd.read_csv(
        meta_csv,
        header=0,
        names=["vid", "clipid", "timestamp", "start_yaw", "end_yaw", "start_tilt", "end_tilt"],
    )
    stats = pd.read_csv(stats_csv, skipinitialspace=True).query(
        "displacement_percentage_50 > 0.10 and d_frame > 5*16"
    )
    seqs = Parallel(n_jobs=os.cpu_count())(
        delayed(_row_to_seq)(row, meta, lefteye_dir) for _, row in stats.iterrows()
    )
    sequences = [s for s in seqs if s is not None]
    tqdm.write(f"split-aware candidates: {len(sequences)}")

    if not sequences:
        tqdm.write("nothing to do")
        return

    # ── phase 2: diagnostics & regeneration (parallel)
    results = Parallel(n_jobs=os.cpu_count(), backend="threading")(
        delayed(_probe_and_fix)(
            seq,
            str(idx_json),
            str(map_json),
            str(ann_dir),
            str(lefteye_dir),
            hfov,
        )
        for seq in tqdm(sequences, desc="check", unit="seq")
    )

    repair_samples, skipped = [], []
    for r in results:
        if r["reason"] is not None:
            tqdm.write(f"[needs-fix] {r['seq']} : {r['reason']}")
        if r["sample"] is not None:
            repair_samples.append(r["sample"])
        elif r["reason"] is not None:
            skipped.append(r["seq"])

    tqdm.write(f"regenerating {len(repair_samples)}  |  unrecoverable {len(skipped)}")

    if not repair_samples:
        tqdm.write("index already clean ✔︎")
        return

    # ── phase 3: sequential write
    patch_shard = wds_dir / "stereo4d-patch-%06d.tar"
    with wds.ShardWriter(str(patch_shard), maxcount=8000) as shard, tqdm(
        repair_samples, desc="write", unit="sample"
    ) as bar:
        for s in bar:
            shard.write(s)

    # ── phase 4: merge index
    tmp_idx = idx_json.with_name("patch-idx.json")
    os.system(f"widsindex create {patch_shard} -o {tmp_idx}")
    merge_index(tmp_idx, idx_json, map_json)
    tmp_idx.unlink(missing_ok=True)
    tqdm.write("index & key_to_idx patched ✔︎")


if __name__ == "__main__":
    main()
