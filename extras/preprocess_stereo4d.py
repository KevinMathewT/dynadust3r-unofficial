#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-sample and pack Stereo4D into WebDataset shards.

Changes in this version:
- "Producers write directly": a small pool of writer workers open their own
  WebDataset ShardWriter (unique pattern per worker) and write samples directly
  to disk (no central JoinableQueue / single writer bottleneck).
- Shard naming stays compatible with your globbing: if the base pattern is
  'stereo4d-%06d.tar', workers produce 'stereo4d-w00-%06d.tar', 'stereo4d-w01-%06d.tar', etc.,
  so 'stereo4d-*.tar' still finds everything for verify / wds2idx.

All original functionality is preserved:
- Filesystem discovery, minimal batched scanning for counts, presampling with
  frame-window constraint, JPEG/NPY encoding in workers,
  optional parallel verification, and optional DALI index generation.

Environment
- Uses cfg.dataset.stereo4d.cache as WIDS_CACHE.

Debug run:
poetry run python -m extras.preprocess_stereo4d \
  +preproc.split=train \
  dataset.stereo4d.wds_dir=/scratch/projects/fouheylab/km6748/stereo4d-data/wds/train \
  +preproc.debug=false +preproc.num_triplets=10000 \
  +preproc.verify=true +preproc.verify_max_samples=10000 \
  +preproc.num_workers=32 \
  +preproc.image_format=npy \
  +preproc.shard_size_gb=8.0 +preproc.samples_per_shard=5000 \
  +preproc.clean_output=true

poetry run python -m extras.preprocess_stereo4d \
  +preproc.split=test \
  dataset.stereo4d.wds_dir=/scratch/projects/fouheylab/km6748/stereo4d-data/wds/test \
  +preproc.debug=false +preproc.num_triplets=1000 \
  +preproc.verify=true +preproc.verify_max_samples=1000 \
  +preproc.num_workers=16 \
  +preproc.image_format=npy \
  +preproc.shard_size_gb=8.0 +preproc.samples_per_shard=5000 \
  +preproc.clean_output=true

Full save:
poetry run python -m extras.preprocess_stereo4d \
  +preproc.split=train \
  dataset.stereo4d.wds_dir=/scratch/projects/fouheylab/km6748/stereo4d-data/wds-full/train \
  +preproc.debug=false +preproc.num_triplets=1568000 \
  +preproc.verify=false \
  +preproc.num_workers=64 \
  +preproc.image_format=npy \
  +preproc.shard_size_gb=8.0 +preproc.samples_per_shard=20000
  +preproc.clean_output=true


poetry run python -m extras.preprocess_stereo4d \
  +preproc.split=test \
  dataset.stereo4d.wds_dir=/scratch/projects/fouheylab/km6748/stereo4d-data/wds-full/test \
  +preproc.debug=false +preproc.num_triplets=16000 \
  +preproc.verify=false \
  +preproc.num_workers=64 \
  +preproc.image_format=npy \
  +preproc.shard_size_gb=8.0 +preproc.samples_per_shard=20000
  +preproc.clean_output=true
"""

from __future__ import annotations

import os, sys, io, json, math, time, random, shutil, tarfile, logging, tempfile, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Any

from joblib import Parallel, delayed
import multiprocessing as mp
from multiprocessing import Process, cpu_count, RLock
from concurrent.futures import ProcessPoolExecutor

from bisect import bisect_left
import numpy as np
import pandas as pd  # (not strictly required; kept for future extensions)
from PIL import Image
from tqdm import tqdm

import webdataset as wds
from decord import VideoReader, cpu as decord_cpu

import hydra
from omegaconf import DictConfig, OmegaConf

# ensure tqdm output from multiple processes is synchronized
try:
    tqdm.set_lock(RLock())
except Exception:
    pass

# -----------------------------------------------------------------------------#
# Utility: geometry helpers (no external deps)
# -----------------------------------------------------------------------------#

def _intrinsic_K(width: int, hfov_deg: float) -> np.ndarray:
    fx = width * 0.5 / np.tan(np.deg2rad(hfov_deg) * 0.5)
    return np.array([[fx, 0, width / 2],
                     [0, fx, width / 2],
                     [0,  0,          1]], dtype=np.float32)

def _safe_inv_pose(mat: np.ndarray) -> np.ndarray:
    """Invert 3x4 or 4x4 camera-to-world matrix to get extrinsics (world-to-camera)."""
    if mat.shape == (4, 4):
        return np.linalg.inv(mat).astype(np.float32)
    if mat.shape == (3, 4):
        pad = np.eye(4, dtype=np.float32)
        pad[:3, :4] = mat.astype(np.float32)
        return np.linalg.inv(pad).astype(np.float32)
    raise ValueError(f"Unexpected pose shape {mat.shape}")

def _optimized_pcs(lengths: np.ndarray,
                   track_indices: np.ndarray,
                   track_coordinates: np.ndarray,
                   idxs: np.ndarray) -> np.ndarray:
    """scatter tracks -> (T,F,4) with validity channel (copied from your loader)."""
    frame_idxs = np.asarray(idxs)
    col_idx_full = track_indices
    keep = np.isin(col_idx_full, frame_idxs)

    if keep.any():
        obs_idx = np.flatnonzero(keep)
        row_s = np.searchsorted(lengths.cumsum(), obs_idx, side='right')
        frames_kept = col_idx_full[keep]
        max_track_frame = int(col_idx_full.max()) if col_idx_full.size > 0 else -1
        max_req_frame = int(frame_idxs.max()) if len(frame_idxs) > 0 else -1
        max_frame = max(max_track_frame, max_req_frame)
        frame2col = np.full(max_frame + 1, -1, dtype=np.int32)
        frame2col[frame_idxs] = np.arange(len(frame_idxs), dtype=np.int32)
        col_s = frame2col[frames_kept]
        coord_s = track_coordinates[keep]
    else:
        row_s = col_s = np.empty((0,), dtype=np.int32)
        coord_s = np.empty((0, 3), dtype=np.float32)

    tracks = np.full((len(lengths), len(frame_idxs), 3), np.nan, np.float32)
    if len(row_s):
        tracks[row_s, col_s] = coord_s
    valid = (~np.isnan(tracks[..., 0]))[..., None].astype(np.float32)
    return np.concatenate([tracks, valid], axis=-1)  # (T,F,4)

# -----------------------------------------------------------------------------#
# Config dataclass (overlay on top of your Hydra config tree)
# -----------------------------------------------------------------------------#

@dataclass
class PreprocArgs:
    split: str = "train"            # which split to write (train/test)
    seed: int = 1337
    num_workers: int = 16           # used both for scanning and for writer workers
    # presampling
    num_triplets: int = 1_568_000   # default: 98k * 16
    max_frame_window: Optional[int] = None  # if None, use cfg.dataset.stereo4d.max_frame_window
    # writing
    shard_size_gb: float = 4.0      # rotate at ~4GB per shard
    samples_per_shard: int = 20000  # safety limit
    image_format: str = "jpg"       # jpg|npy
    jpeg_quality: int = 95
    # debug / verify
    debug: bool = False
    debug_num_triplets: int = 256
    verify: bool = False
    verify_max_samples: int = 32
    verify_workers: int = 16          # <--- parallel verify pool size
    # indexing
    make_dali_index: bool = True
    wds_pattern: Optional[str] = None  # base pattern; workers insert their token before %06d
    # scanning strategy
    count_batch_size: int = 2000         # how many seqs to scan per batch
    count_all_for_csv: bool = False      # if True, scan all for CSV at the very end
    # housekeeping
    clean_output: bool = False           # if True, remove old shards/idx before writing

# -----------------------------------------------------------------------------#
# FS discovery
# -----------------------------------------------------------------------------#

def list_sequences(lefteye_split_dir: Path, data_root_split_dir: Path) -> List[Tuple[str, Path, Path]]:
    """Return list of (seq_id, mp4_path, npz_path)."""
    mp4s = sorted(lefteye_split_dir.glob("*-left_rectified.mp4"))
    out = []
    for mp4 in mp4s:
        seq = mp4.name.replace("-left_rectified.mp4", "")
        npz = data_root_split_dir / f"{seq}.npz"
        if npz.is_file():
            out.append((seq, mp4, npz))
    return out

# -----------------------------------------------------------------------------#
# Frame counting: min(len(mp4), len(tracks)), batched scanning
# -----------------------------------------------------------------------------#

def _count_frames_one_both(arg: Tuple[str, str, str]) -> Tuple[str, int, int, int, int]:
    """
    Returns: (seq, n_min, width, n_mp4, n_npz)
      - n_min = min(n_mp4, n_npz) if both >0; else max(valid) or -1 if none valid
      - width is from MP4 (or -1 on failure)
    """
    seq, mp4_path, npz_path = arg
    n_mp4, n_npz, width = -1, -1, -1

    # MP4 length & width via Decord (cheap; uses container index)
    try:
        vr = VideoReader(mp4_path, ctx=decord_cpu(0))
        n_mp4 = len(vr)
        width = int(vr[0].shape[1])
        del vr
    except Exception:
        pass

    # Tracks length from NPZ: number of frames in camera2world
    try:
        with np.load(npz_path, allow_pickle=True) as ann:
            c2w = ann["camera2world"]
            n_npz = int(c2w.shape[0])
    except Exception:
        pass

    if n_mp4 > 0 and n_npz > 0:
        n_min = min(n_mp4, n_npz)
    elif n_mp4 > 0:
        n_min = n_mp4
    elif n_npz > 0:
        n_min = n_npz
    else:
        n_min = -1

    return seq, n_min, width, n_mp4, n_npz

def _count_frames_batch(pairs: List[Tuple[str, Path, Path]], max_workers: int
                        ) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Count a batch of sequences in parallel.
    Returns mapping: seq -> (n_min, width, n_mp4, n_npz)
    """
    tasks = [(seq, str(mp4), str(npz)) for (seq, mp4, npz) in pairs]
    out: Dict[str, Tuple[int, int, int, int]] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for seq, n_min, width, n_mp4, n_npz in ex.map(_count_frames_one_both, tasks, chunksize=64):
            out[seq] = (n_min, width, n_mp4, n_npz)
    return out

def _enough_triplets(seq_to_counts: Dict[str, Tuple[int, int, int, int]],
                     max_frame_window: int,
                     want_triplets: int,
                     rng: random.Random) -> Tuple[bool, Dict[str, np.ndarray], List[Tuple[str,int,int,int]]]:
    """
    Uniformly sample triplets (seq, l, m, r) over the entire population of valid triplets
    across the provided sequences, with constraint r - l <= max_frame_window.
    """
    # Build per-sequence pools of gaps with weights w_d = (n - d) * (d - 1)
    pools: Dict[str, Dict[str, Any]] = {}
    seq_names: List[str] = []
    seq_totals: List[int] = []

    for seq, (n_min, _w, _nm, _nn) in seq_to_counts.items():
        n = int(n_min)
        if n < 3:
            continue
        D = min(max_frame_window, n - 1)
        if D < 2:
            continue
        ds = np.arange(2, D + 1, dtype=np.int64)
        w = (n - ds) * (ds - 1)  # number of (l,m,r) with gap d
        total_d = int(w.sum())
        if total_d <= 0:
            continue
        cum_d = np.cumsum(w, dtype=np.int64).tolist()  # list for bisect
        pools[seq] = {"n": n, "ds": ds.tolist(), "cum_d": cum_d, "total_d": total_d}
        seq_names.append(seq)
        seq_totals.append(total_d)

    if not seq_names:
        return False, {}, []

    # Global sequence-level weights -> uniform over ALL valid triplets overall
    seq_cum = np.cumsum(np.asarray(seq_totals, dtype=np.int64)).tolist()
    global_total = int(seq_cum[-1])

    # If the request exceeds the population, cap to avoid an endless loop
    if want_triplets > global_total:
        print(f"[presample] requested {want_triplets} triplets but only {global_total} possible; reducing.")
        want_triplets = global_total

    seen: set[Tuple[str,int,int,int]] = set()
    triplets: List[Tuple[str,int,int,int]] = []

    while len(triplets) < want_triplets:
        # 1) choose a sequence proportional to its number of valid triplets
        t_seq = rng.randrange(global_total)   # 0..global_total-1
        si = bisect_left(seq_cum, t_seq + 1)
        seq = seq_names[si]
        pool = pools[seq]
        n = pool["n"]

        # 2) choose a gap d with probability proportional to its weight
        t_d = rng.randrange(pool["total_d"])
        di = bisect_left(pool["cum_d"], t_d + 1)
        d = pool["ds"][di]

        # 3) choose l, then m uniformly
        #    l in [0, n - d - 1]  (n - d choices), r = l + d, m in (l, r)
        l = rng.randrange(0, n - d)
        r = l + d
        m = rng.randrange(l + 1, r)

        key = (seq, l, m, r)
        if key in seen:
            continue
        seen.add(key)
        triplets.append(key)

    return True, pools, triplets




def presample_with_minimal_scans(seqs: List[Tuple[str, Path, Path]],
                                 pre: PreprocArgs,
                                 ds_max_window: int,
                                 rng: random.Random) -> Tuple[
                                     Dict[str, Tuple[int,int,int,int]],
                                     Dict[str, np.ndarray],
                                     List[Tuple[str,int,int,int]]
                                 ]:
    """
    New behavior:
      1) Randomly pick at most `want` sequence IDs (no replacement).
      2) Parallel-count ONLY those selected seqs.
      3) Uniformly sample triplets (same rules as the newer _compute_triplets).
    """
    max_window = pre.max_frame_window or int(ds_max_window)
    want = pre.debug_num_triplets if pre.debug else pre.num_triplets

    # --- CHANGED: select exactly up to `want` seq IDs first
    shuffled = seqs[:]
    rng.shuffle(shuffled)
    selected = shuffled[:min(len(shuffled), want)]

    # parallel-count only the selected seqs
    seq_to_counts: Dict[str, Tuple[int,int,int,int]] = {}
    progress = tqdm(total=len(selected), desc="scan selected (batches)", leave=False)
    for i in range(0, len(selected), pre.count_batch_size):
        batch = selected[i:i+pre.count_batch_size]
        batch_counts = _count_frames_batch(batch, pre.num_workers)
        seq_to_counts.update(batch_counts)
        progress.update(len(batch))
    progress.close()

    # --- CHANGED: uniform sampling; pools no longer used
    ok, pools, triplets = _enough_triplets(seq_to_counts, max_window, want, rng)
    if not ok:
        print(f"[presample] WARNING: could not form {want} triplets from {len(selected)} selected sequences; got {len(triplets)}.")

    return seq_to_counts, pools, triplets


# -----------------------------------------------------------------------------#
# Presampling helpers (triplet sampling)
# -----------------------------------------------------------------------------#

# -----------------------------------------------------------------------------#
# Encoding helpers
# -----------------------------------------------------------------------------#

def _encode_image(arr: np.ndarray, fmt: str, quality: int) -> bytes:
    if fmt == "npy":
        buf = io.BytesIO()
        np.save(buf, arr)
        return buf.getvalue()
    # default: jpg
    img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def _npy_bytes(array: np.ndarray) -> bytes:
    """Serialize a numpy array to true .npy bytes (with header)."""
    buf = io.BytesIO()
    np.save(buf, np.asarray(array))
    return buf.getvalue()

def _prepare_sample_bytes(seq: str,
                          l: int, m: int, r: int,
                          vr: VideoReader,
                          ann: Dict[str, Any],
                          K: np.ndarray,
                          image_fmt: str, jpeg_quality: int) -> Dict[str, bytes]:
    # frames
    frames = [vr[i].asnumpy() for i in (l, m, r)]
    # world_pc_valid for (l,m,r)
    pcs_all = _optimized_pcs(ann["track_lengths"], ann["track_indices"], ann["track_coordinates"],
                             np.asarray([l, m, r], dtype=np.int32))  # (T,3,4)
    l_pv = pcs_all[:, 0, :].astype(np.float32)
    m_pv = pcs_all[:, 1, :].astype(np.float32)
    r_pv = pcs_all[:, 2, :].astype(np.float32)
    # extrinsics
    c2w = ann["camera2world"]
    l_cam = _safe_inv_pose(c2w[l])
    m_cam = _safe_inv_pose(c2w[m])
    r_cam = _safe_inv_pose(c2w[r])

    # bytes
    sample = {
        "__key__": f"{seq}_{l:05d}_{m:05d}_{r:05d}",
        "l." + ("jpg" if image_fmt == "jpg" else "npy"): _encode_image(frames[0], image_fmt, jpeg_quality),
        "m." + ("jpg" if image_fmt == "jpg" else "npy"): _encode_image(frames[1], image_fmt, jpeg_quality),
        "r." + ("jpg" if image_fmt == "jpg" else "npy"): _encode_image(frames[2], image_fmt, jpeg_quality),
        "l.pv.npy": _npy_bytes(l_pv.astype(np.float32, copy=False)),
        "m.pv.npy": _npy_bytes(m_pv.astype(np.float32, copy=False)),
        "r.pv.npy": _npy_bytes(r_pv.astype(np.float32, copy=False)),
        "l.cam.npy": _npy_bytes(np.asarray(l_cam, np.float32)),
        "m.cam.npy": _npy_bytes(np.asarray(m_cam, np.float32)),
        "r.cam.npy": _npy_bytes(np.asarray(r_cam, np.float32)),
        "k.npy": _npy_bytes(np.asarray(K, np.float32)),
        "seq.txt": (seq + "\n").encode("utf-8"),
    }
    return sample

# -----------------------------------------------------------------------------#
# Direct-writer worker (no central queue)
# -----------------------------------------------------------------------------#

def _pattern_with_token(base_pattern: str, token: str) -> str:
    """
    Insert a unique token before %06d to avoid filename collisions across workers.
    If '%06d' is missing, append '-{token}-%06d' before the file extension.
    """
    if "%06d" in base_pattern:
        return base_pattern.replace("%06d", f"{token}-%06d")
    # robust fallback
    p = Path(base_pattern)
    stem = p.stem
    return str(p.with_name(f"{stem}-{token}-%06d{p.suffix}"))

def _writer_worker(
    worker_id: int,
    assignments: List[Tuple[str, str, str, List[Tuple[int,int,int]]]],  # (seq, mp4_path, npz_path, triplets)
    out_pattern: str,
    shard_size_bytes: int,
    samples_per_shard: int,
    hfov: float,
    image_fmt: str,
    jpeg_quality: int,
) -> None:
    """
    Each worker opens its own ShardWriter with a unique token (w{worker_id:02d})
    and iterates through its assigned sequences, writing samples directly.
    """
    token = f"w{worker_id:02d}"
    my_pattern = _pattern_with_token(out_pattern, token)
    out_dir = Path(str(my_pattern)).parent
    os.makedirs(out_dir, exist_ok=True)

    total_to_write = sum(len(t) for (_seq, _mp4, _npz, t) in assignments)
    pbar = tqdm(
        total=total_to_write,
        desc=f"writer[{token}]",
        position=worker_id + 1,   # <-- unique line per process
        leave=True,
        dynamic_ncols=True,
    )


    # one ShardWriter per worker to accumulate across sequences -> large shards
    with wds.ShardWriter(my_pattern, maxsize=shard_size_bytes, maxcount=samples_per_shard) as sink:
        for seq, mp4_path, npz_path, trips in assignments:
            try:
                vr = VideoReader(mp4_path, ctx=decord_cpu(0))
                w = vr[0].shape[1]
                ann = np.load(npz_path, allow_pickle=True)
                K = _intrinsic_K(int(w), hfov)

                for (l, m, r) in trips:
                    sample = _prepare_sample_bytes(seq, l, m, r, vr, ann, K,
                                                   image_fmt=image_fmt, jpeg_quality=jpeg_quality)
                    key = sample.pop("__key__")
                    sink.write({"__key__": key, **sample})
                    pbar.update(1)
            except Exception as e:
                logging.exception(f"[writer {token} | seq={seq}] {e}")
            finally:
                # ensure NPZ file handle is closed (np.load returns NpzFile if not using context)
                try:
                    if 'ann' in locals() and hasattr(ann, 'close'):
                        ann.close()
                except Exception:
                    pass
    pbar.close()

# -----------------------------------------------------------------------------#
# Verification helpers (debug)
# -----------------------------------------------------------------------------#

def _decode_image(bytes_or_arr: bytes, fmt: str) -> np.ndarray:
    if fmt == "npy":
        if isinstance(bytes_or_arr, np.ndarray):
            return bytes_or_arr
        try:
            return np.load(io.BytesIO(bytes_or_arr), allow_pickle=False)
        except Exception:
            arr = np.frombuffer(bytes_or_arr, dtype=np.uint8)
            return arr
    return np.array(Image.open(io.BytesIO(bytes_or_arr)).convert("RGB")).astype(np.uint8)

def _decode_image_auto(sample: Dict[str, Any], base: str) -> np.ndarray:
    if f"{base}.npy" in sample:
        return _decode_image(sample[f"{base}.npy"], "npy")
    if f"{base}.jpg" in sample:
        return _decode_image(sample[f"{base}.jpg"], "jpg")
    for k in sample.keys():
        if isinstance(k, str) and k.startswith(base + "."):
            ext = k.rsplit(".", 1)[-1].lower()
            if ext in ("npy", "jpg", "jpeg"):
                return _decode_image(sample[k], "npy" if ext == "npy" else "jpg")
    raise KeyError(f"{base}.(npy|jpg)")

def _get_sample_field(sample: Dict[str, Any], key: str) -> Any:
    if key in sample:
        return sample[key]
    if "." in key:
        base = key.split(".", 1)[0]
        if base in sample:
            return sample[base]
        for k in sample.keys():
            if isinstance(k, str) and k.startswith(base + "."):
                return sample[k]
    raise KeyError(key)

def _as_array(val: Any, dtype: np.dtype, shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    if isinstance(val, np.ndarray):
        arr = val
    else:
        buf = bytes(val)
        if len(buf) >= 6 and buf[:6] == b"\x93NUMPY":
            arr = np.load(io.BytesIO(buf), allow_pickle=False)
        else:
            arr = np.frombuffer(buf, dtype=dtype)
    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def _verify_worker_ordered(
    worker_id: int,
    assignments: List[Tuple[str, str, str, List[Tuple[int,int,int]]]],  # (seq, mp4_path, npz_path, triplets)
    out_pattern: str,
    split_dirs: Tuple[Path, Path],
    hfov: float,
    image_fmt: str,
    per_worker_limit: Optional[int] = None,
    mse_tol: float = 2.0,
    atol: float = 1e-5,
) -> Tuple[int, int, int, int, int, int]:
    """
    Verify that this worker's shards contain exactly the samples implied by `assignments`,
    in the SAME ORDER they would have been written (seq-by-seq, triplet-by-triplet).

    Returns: (total_expected, total_read, passed_content, order_mismatch, wrong_shard, exceptions)
    """
    import webdataset as wds

    token = f"w{worker_id:02d}"
    my_pattern = _pattern_with_token(out_pattern, token)
    shard_glob = Path(str(my_pattern)).name.replace("%06d", "*")
    shard_paths = sorted(Path(str(my_pattern)).parent.glob(shard_glob))

    # Build expected stream for this worker
    expected: List[Tuple[str,int,int,int]] = []
    seq_to_paths: Dict[str, Tuple[str, str]] = {}
    for seq, mp4_path, npz_path, trips in assignments:
        seq_to_paths[seq] = (mp4_path, npz_path)
        expected.extend([(seq, l, m, r) for (l, m, r) in trips])

    total_expected = len(expected)
    if per_worker_limit is not None:
        total_expected = min(total_expected, per_worker_limit)
        expected = expected[:per_worker_limit]

    # No shards for this worker token -> everything is missing
    if not shard_paths:
        print(f"[verify {token}] no shards found matching {shard_glob}; expected {len(expected)} samples.")
        return (len(expected), 0, 0, len(expected), len(expected), 0)

    # Stream this worker's shards in order (no shuffling)
    urls = [str(p) for p in shard_paths]
    ds = wds.WebDataset(urls, shardshuffle=False)

    # stats & progress
    passed_content = 0
    order_mismatch = 0
    wrong_shard = 0
    exceptions = 0
    total_read = 0

    pbar = tqdm(
        total=total_expected,
        desc=f"verify[{token}]",
        position=worker_id + 1,
        leave=True,
        dynamic_ncols=True,
    )

    it = iter(ds)
    extra = 0
    try:
        for i, (seq_exp, l, m, r) in enumerate(expected):
            try:
                sample = next(it)
            except StopIteration:
                # ran out of samples before fulfilling expected stream
                order_mismatch += (len(expected) - i)
                break

            total_read += 1
            key = sample["__key__"]
            exp_key = f"{seq_exp}_{l:05d}_{m:05d}_{r:05d}"

            # 1) Order check (strict)
            if key != exp_key:
                order_mismatch += 1
                # also check if this sample even belongs to this worker's assignments
                try:
                    seq_chk, ls, ms, rs = key.rsplit("_", 3)
                    l_use, m_use, r_use = int(ls), int(ms), int(rs)
                except Exception:
                    # couldn't parse the key at all
                    exceptions += 1
                    pbar.update(1)
                    continue
                if seq_chk not in seq_to_paths:
                    wrong_shard += 1
                seq_for_content = seq_chk
            else:
                # ordered; verify the expected (seq_exp,l,m,r)
                seq_for_content = seq_exp
                l_use, m_use, r_use = l, m, r

            # 2) Content check against ground truth (no caching; open per-sample)
            try:
                mp4_path, npz_path = seq_to_paths[seq_for_content]

                # decode images from wds sample
                a_l = _decode_image_auto(sample, "l")
                a_m = _decode_image_auto(sample, "m")
                a_r = _decode_image_auto(sample, "r")

                # open ground-truth video and grab frames
                vr = VideoReader(mp4_path, ctx=decord_cpu(0))
                try:
                    f_l = vr[l_use].asnumpy()
                    f_m = vr[m_use].asnumpy()
                    f_r = vr[r_use].asnumpy()
                finally:
                    # ensure decoder resources are freed promptly
                    del vr

                # compute K from the decoded frame width
                width = int(f_l.shape[1])
                K = _intrinsic_K(width, hfov)

                # open annotations just for this sample
                with np.load(npz_path, allow_pickle=True) as ann:
                    pcs_all = _optimized_pcs(
                        ann["track_lengths"], ann["track_indices"], ann["track_coordinates"],
                        np.asarray([l_use, m_use, r_use], dtype=np.int32)
                    )
                    c2w = ann["camera2world"]
                    cams = [_safe_inv_pose(c2w[idx]) for idx in (l_use, m_use, r_use)]

                # from sample
                v_l = _as_array(_get_sample_field(sample, "l.pv.npy"), np.float32, (-1, 4))
                v_m = _as_array(_get_sample_field(sample, "m.pv.npy"), np.float32, (-1, 4))
                v_r = _as_array(_get_sample_field(sample, "r.pv.npy"), np.float32, (-1, 4))
                k_key = "K.npy" if ("K.npy" in sample) else ("k.npy" if ("k.npy" in sample) else "K.npy")
                k_w = _as_array(_get_sample_field(sample, k_key), np.float32, (3, 3))
                c_l = _as_array(_get_sample_field(sample, "l.cam.npy"), np.float32, (4, 4))

                # basic presence checks
                img_ext = "npy" if image_fmt == "npy" else "jpg"
                need_keys = [
                    f"l.{img_ext}", f"m.{img_ext}", f"r.{img_ext}",
                    "l.pv.npy", "m.pv.npy", "r.pv.npy",
                    "l.cam.npy", "m.cam.npy", "r.cam.npy",
                    "k.npy", "seq.txt",
                ]
                missing = [k for k in need_keys if k not in sample]
                if missing:
                    raise KeyError(f"missing keys: {missing}")

                # checks
                img_ok = (
                    a_l.shape == f_l.shape and a_m.shape == f_m.shape and a_r.shape == f_r.shape and
                    float(np.mean(np.abs(a_l.astype(np.int16) - f_l.astype(np.int16)))) < mse_tol and
                    float(np.mean(np.abs(a_m.astype(np.int16) - f_m.astype(np.int16)))) < mse_tol and
                    float(np.mean(np.abs(a_r.astype(np.int16) - f_r.astype(np.int16)))) < mse_tol
                )
                pcs_ok = (
                    np.allclose(v_l, pcs_all[:, 0, :], atol=atol, equal_nan=True) and
                    np.allclose(v_m, pcs_all[:, 1, :], atol=atol, equal_nan=True) and
                    np.allclose(v_r, pcs_all[:, 2, :], atol=atol, equal_nan=True)
                )
                K_ok = np.allclose(k_w, K, atol=atol)
                cam_ok = np.allclose(c_l, cams[0], atol=atol)
                seqtxt_ok = (bytes(sample["seq.txt"]).decode("utf-8", errors="ignore").strip() == seq_for_content)

                if img_ok and pcs_ok and K_ok and cam_ok and seqtxt_ok:
                    passed_content += 1
                else:
                    # compact diagnostics
                    if not img_ok:
                        ml = float(np.mean(np.abs(a_l.astype(np.int16) - f_l.astype(np.int16))))
                        mm = float(np.mean(np.abs(a_m.astype(np.int16) - f_m.astype(np.int16))))
                        mr = float(np.mean(np.abs(a_r.astype(np.int16) - f_r.astype(np.int16))))
                        print(f"[verify {token}] image mismatch {key} | l={ml:.2f} m={mm:.2f} r={mr:.2f}")
                    if not pcs_ok:
                        print(f"[verify {token}] pcs mismatch {key}")
                    if not K_ok:
                        print(f"[verify {token}] K mismatch {key}")
                    if not cam_ok:
                        print(f"[verify {token}] cam mismatch {key}")
                    if not seqtxt_ok:
                        print(f"[verify {token}] seq.txt mismatch {key}")

            except Exception as e:
                exceptions += 1
                try:
                    present = sorted([k for k in sample.keys() if isinstance(k, str)])
                except Exception:
                    present = []
                print(f"[verify {token}] exception on {key}: {e} | present_keys={present}")

            pbar.update(1)

        # If there are more samples in shards than expected, consume one to count “extra”
        try:
            next(it)
            # if we get here, at least one extra sample exists
            extra = 1
            for _ in it:
                extra += 1
        except StopIteration:
            pass

    finally:
        pbar.close()

    # Missing = expected but not read (due to early StopIteration or limit)
    missing = max(0, len(expected) - total_read)
    if extra:
        print(f"[verify {token}] WARNING: found {extra} extra samples beyond expected stream")

    return (len(expected), total_read, passed_content, order_mismatch, wrong_shard, exceptions)


def _verify_worker_ordered_entry(args):
    """
    Thin wrapper around _verify_worker_ordered so ProcessPoolExecutor/joblib
    can pickle the callable. 'args' is a tuple of the arguments.
    """
    (worker_id, assignments, out_pattern, split_dirs, hfov, image_fmt, cap) = args
    return _verify_worker_ordered(
        worker_id=worker_id,
        assignments=assignments,
        out_pattern=out_pattern,
        split_dirs=split_dirs,
        hfov=hfov,
        image_fmt=image_fmt,
        per_worker_limit=cap,
    )


def verify_shards_by_buckets(
    buckets: List[List[Tuple[str, str, str, List[Tuple[int,int,int]]]]],
    out_pattern: str,
    split_dirs: Tuple[Path, Path],
    hfov: float,
    image_fmt: str,
    verify_max_samples: Optional[int] = None,
) -> None:
    """
    Launch one verifier process per worker token (w00, w01, ...), mirroring the writer structure.
    Uses ProcessPoolExecutor with a top-level entry to avoid pickling errors on loky/spawn.
    """
    # Distribute a per-worker cap if a global cap is requested
    grand_total = sum(sum(len(t) for (_s, _m, _n, t) in b) for b in buckets)
    per_worker_caps: List[Optional[int]] = []
    for b in buckets:
        w_total = sum(len(t) for (_s, _m, _n, t) in b)
        if verify_max_samples is None or verify_max_samples <= 0 or grand_total == 0:
            per_worker_caps.append(None)
        else:
            cap = int(round(verify_max_samples * (w_total / grand_total)))
            per_worker_caps.append(max(1, cap) if w_total > 0 else 0)

    # Build argument tuples (all picklable)
    args_list = []
    for wid, assignments in enumerate(buckets):
        if not assignments:
            continue
        args_list.append(
            (
                wid,
                assignments,
                out_pattern,
                (split_dirs[0], split_dirs[1]),
                float(hfov),
                image_fmt,
                per_worker_caps[wid],
            )
        )

    # Fan out
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os
    agg = {"expected": 0, "read": 0, "passed": 0, "order": 0, "wrong_shard": 0, "exceptions": 0}
    if not args_list:
        print("[verify summary] nothing to verify")
        return

    max_workers = min(len(args_list), os.cpu_count() or 1) // 2
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futures = [ex.submit(_verify_worker_ordered_entry, args) for args in args_list]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="verifiers", dynamic_ncols=True):
            try:
                expd, read, passed, order, wrong, exc = fut.result()
                agg["expected"] += expd
                agg["read"] += read
                agg["passed"] += passed
                agg["order"] += order
                agg["wrong_shard"] += wrong
                agg["exceptions"] += exc
            except Exception as e:
                print(f"[verify] worker raised: {e}")
                agg["exceptions"] += 1

    print(
        "[verify summary] "
        f"expected={agg['expected']} read={agg['read']} "
        f"content_passed={agg['passed']} order_mismatch={agg['order']} "
        f"wrong_shard={agg['wrong_shard']} exceptions={agg['exceptions']}"
    )


# -----------------------------------------------------------------------------#
# DALI index generation
# -----------------------------------------------------------------------------#

def run_wds2idx_for_dir(out_pattern: str, max_workers: int = 8) -> None:
    """Run DALI's wds2idx over all shards matching the pattern (parent process)."""
    out_dir = Path(str(out_pattern)).parent
    pattern_name = Path(str(out_pattern)).name.replace("%06d", "*")
    tars = sorted(out_dir.glob(pattern_name))
    if not tars:
        return
    if shutil.which("wds2idx") is None:
        logging.info("wds2idx not found in PATH; skipping DALI index generation.")
        return
    with ProcessPoolExecutor(max_workers=min(max_workers, len(tars))) as ex:
        list(tqdm(ex.map(_make_dali_idx, tars), total=len(tars), desc="wds2idx"))

def _make_dali_idx(tar_path: Path) -> None:
    idx_path = tar_path.with_suffix(".idx")
    try:
        subprocess.run(["wds2idx", str(tar_path), str(idx_path)],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        logging.error("wds2idx not found in PATH. Install NVIDIA DALI and ensure wds2idx is available.")
    except subprocess.CalledProcessError as e:
        logging.error(f"wds2idx failed for {tar_path}:\n{e.stderr.decode('utf-8', errors='ignore')}")

# -----------------------------------------------------------------------------#
# Work partitioning for direct writers
# -----------------------------------------------------------------------------#

def _partition_sequences_for_workers(
    by_seq: Dict[str, List[Tuple[int,int,int]]],
    path_map: Dict[str, Tuple[str,str]],
    num_workers: int
) -> List[List[Tuple[str, str, str, List[Tuple[int,int,int]]]]]:
    """
    Greedy bin-packing by triplet count: sort sequences by descending triplet
    count and assign to the worker with the smallest current load.
    Returns: list of length num_workers; each element is a list of
      (seq, mp4_path, npz_path, triplets_for_seq)
    """
    num_workers = max(1, num_workers)
    buckets: List[List[Tuple[str, str, str, List[Tuple[int,int,int]]]]] = [[] for _ in range(num_workers)]
    loads = [0 for _ in range(num_workers)]

    items = [(seq, trips, len(trips)) for seq, trips in by_seq.items()]
    items.sort(key=lambda x: x[2], reverse=True)
    for seq, trips, n in items:
        wid = loads.index(min(loads))
        mp4_path, npz_path = path_map[seq]
        buckets[wid].append((seq, mp4_path, npz_path, trips))
        loads[wid] += n
    return buckets

# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Uses the same Hydra tree as your training config:
      cfg.dataset.stereo4d.{name,path,wds_dir,cache,lefteye_dir,train_split,valid_split,max_frame_window,hfov,pm_source}
    """
    # Resolve preprocessing args (we keep them under cfg.preproc.* to avoid clashing)
    pre: PreprocArgs = PreprocArgs()
    if "preproc" in cfg:
        pre = OmegaConf.merge(pre, cfg.preproc)  # allow overrides
    # split selection
    split = pre.split or cfg.dataset.stereo4d.train_split
    # folders
    ds = cfg.dataset.stereo4d
    lefteye_split_dir = Path(ds.lefteye_dir) / split
    npz_split_dir = Path(ds.path) / split
    # WIDS cache + tmp
    os.environ.setdefault("WIDS_CACHE", str(Path(ds.cache)))
    os.environ.setdefault("TMPDIR", str(Path(ds.cache)))
    os.environ.setdefault("TMP", str(Path(ds.cache)))
    os.environ.setdefault("TEMP", str(Path(ds.cache)))
    Path(os.environ["WIDS_CACHE"]).mkdir(parents=True, exist_ok=True)

    # output WDS dir
    ds_wds_dir = getattr(ds, "wds_dir", None)
    wds_dir = Path(ds_wds_dir) if (ds_wds_dir and ds_wds_dir not in ("???",)) else (Path(ds.path) / "wds" / split)
    wds_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(pre.wds_pattern or (wds_dir / "stereo4d-%06d.tar"))

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s)")

    print("== Stereo4D -> WebDataset Preprocessor (direct writers) ==")
    print(f"split                 : {split}")
    print(f"lefteye mp4 dir       : {lefteye_split_dir}")
    print(f"tracks npz dir        : {npz_split_dir}")
    print(f"out shards (base)     : {out_pattern}")
    print(f"num_workers (writers) : {pre.num_workers}")
    print(f"num_triplets          : {pre.debug_num_triplets if pre.debug else pre.num_triplets}")
    print(f"image_format          : {pre.image_format} (jpeg_quality={pre.jpeg_quality})")
    print(f"make_dali_index       : {pre.make_dali_index}")
    print(f"WIDS_CACHE            : {os.environ.get('WIDS_CACHE')}")

    # 0) Optional cleanup of existing shards/idx
    if pre.clean_output:
        out_dir = Path(str(out_pattern)).parent.resolve()

        if not out_dir.exists():
            print(f"[cleanup] nothing to delete (missing dir): {out_dir}")
        else:
            # ---- safety rails: refuse obviously dangerous deletes
            if out_dir == Path("/") or len(out_dir.parts) < 3:
                raise RuntimeError(f"Refusing to delete suspicious directory: {out_dir}")

            print(f"[cleanup] removing entire directory recursively: {out_dir}")
            try:
                shutil.rmtree(out_dir)
            except Exception as e:
                logging.warning(f"Failed to remove {out_dir}: {e}")
            else:
                # recreate the directory so writers have a target
                try:
                    out_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logging.error(f"Failed to recreate {out_dir}: {e}")
                    raise


    # 1) Discover sequences
    seqs = list_sequences(lefteye_split_dir, npz_split_dir)
    if not seqs:
        print("No sequences discovered. Check paths.")
        sys.exit(1)
    print(f"discovered sequences  : {len(seqs)}")

    # 2–3) Iteratively scan in small batches until we can presample enough triplets.
    rng = random.Random(int(pre.seed))
    seq_to_counts, pools, triplets = presample_with_minimal_scans(
        seqs, pre, int(ds.max_frame_window), rng
    )

    # 4) Group triplets by sequence (to open mp4/npz once per sequence)
    by_seq: Dict[str, List[Tuple[int,int,int]]] = {}
    for seq, l, m, r in triplets:
        by_seq.setdefault(seq, []).append((l, m, r))

    # 5) Partition sequences across writer workers and launch them
    maxsize_bytes = int(pre.shard_size_gb * (1024**3))
    path_map: Dict[str, Tuple[str,str]] = {seq: (str(mp4), str(npz)) for (seq, mp4, npz) in seqs}
    buckets = _partition_sequences_for_workers(by_seq, path_map, max(1, pre.num_workers))

    procs: List[Process] = []
    for wid, assignments in enumerate(buckets):
        if not assignments:
            continue
        p = Process(
            target=_writer_worker,
            args=(wid, assignments, out_pattern, maxsize_bytes, pre.samples_per_shard,
                  float(ds.hfov), pre.image_format, pre.jpeg_quality),
            daemon=True
        )
        procs.append(p)
        p.start()

    # Join writer workers
    for p in tqdm(procs, desc="writers", position=0, leave=True, dynamic_ncols=True):
        p.join()

    # Collect shard paths once writing is complete
    shard_glob = Path(out_pattern).name.replace("%06d", "*")
    shard_paths = sorted(Path(out_pattern).parent.glob(shard_glob))

    # 6) Ordered, per-worker verification against assignments
    if pre.verify:
        verify_shards_by_buckets(
            buckets=buckets,
            out_pattern=out_pattern,
            split_dirs=(lefteye_split_dir, npz_split_dir),
            hfov=float(ds.hfov),
            image_fmt=pre.image_format,
            verify_max_samples=min(pre.verify_max_samples, sum(len(t) for xs in buckets for (_s, _m, _n, t) in xs)),
        )


    # 6.5) Optional: build DALI .idx files (parent process, safe)
    if pre.make_dali_index:
        run_wds2idx_for_dir(out_pattern, max_workers=pre.num_workers)


if __name__ == "__main__":
    main()
