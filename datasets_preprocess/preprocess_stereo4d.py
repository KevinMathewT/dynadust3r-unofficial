#!/usr/bin/env python3
"""
preprocess_stereo4d.py
----------------------
convert stereo4d (mp4 + npz) to webdataset shards (one frame = one sample)
then benchmark random-access speed against a naïve opencv loader.

key fixes
* index json now records absolute shard paths → no “double-prefix” errors
* widsindex invoked with the proper cli (`create … -o …`)
* shardlistdataset loads directly from the index (urls already absolute)
* removed deprecated string transform arg that triggered a warning
"""

from __future__ import annotations

import gc
import os, json, math, time, random, logging, glob, io
import cv2, numpy as np, pandas as pd, torch
from decord import VideoReader, cpu
from joblib import Parallel, delayed, parallel
from tqdm import tqdm
from pathlib import Path
import webdataset as wds, wids
import hydra
from omegaconf import DictConfig
from contextlib import contextmanager
import shutil, tempfile, os
from pathlib import Path
import atexit   # ← added

# ─────────────── editable constants ─────────────────────────────────────────
ALLOW_PICKLE          = True           # flip True/False to pickled / unpickled
USE_THREADS           = False          # True → joblib uses threads instead of processes
SPLIT                 = "test"         # "train" | "test" | …
NUM_WORKERS           = 1              # exporter processes
TEST_N                = 8              # none ⇒ full run
MAX_SHARD_SIZE_GB     = 50             # shard rollover
MAX_SAMPLES_PER_SHARD = 8_000
logging.basicConfig(level=logging.WARNING)
# ────────────────────────────────────────────────────────────────────────────

# ───────── persistent shard writers (one per worker rank) ──────────
_writers: dict[int, wds.ShardWriter] = {}

def _get_sink(rank: int, out_dir: str) -> wds.ShardWriter:
    """returns a per-rank ShardWriter, creating it lazily on first use"""
    sink = _writers.get(rank)
    if sink is None:
        shard_tmpl = os.path.join(out_dir, f"stereo4d-{rank:02d}-%06d.tar")
        sink = wds.ShardWriter(
            shard_tmpl,
            maxsize=MAX_SHARD_SIZE_GB * 1e9,
            maxcount=MAX_SAMPLES_PER_SHARD,
            verbose=False,
        )
        _writers[rank] = sink
    return sink

def _close_sinks():
    for s in _writers.values():
        s.close()
atexit.register(_close_sinks)
# ────────────────────────────────────────────────────────────────────


def clear_wids_cache():
    # default location used by widsindex
    cache_dir = Path(tempfile.gettempdir()) / "_wids_cache"
    if cache_dir.is_dir():
        shutil.rmtree(cache_dir)
        print(f"cleared wids cache at {cache_dir}")

# ───────── writer helper (serialize to .npy bytes) ─────────────────────────
def to_npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32), allow_pickle=ALLOW_PICKLE)
    return buf.getvalue()

# ───────── reader helper (load np.ndarray from blob) ────────────────────────
def load_from_sample(blob) -> np.ndarray:
    if isinstance(blob, np.ndarray):
        return blob
    if isinstance(blob, (bytes, bytearray, memoryview)):
        raw = bytes(blob)
        return np.load(io.BytesIO(raw), allow_pickle=ALLOW_PICKLE)
    # assume file-like
    return np.load(blob, allow_pickle=ALLOW_PICKLE)

# ─────────────── helper funcs (unchanged) ───────────────────────────────────
def intrinsic_K(width: int, hfov: float) -> np.ndarray:
    fx = width * 0.5 / math.tan(math.radians(hfov) * 0.5)
    return np.array([[fx, 0, width/2], [0, fx, width/2], [0, 0, 1]], np.float32)


def inv(mat):
    """
    EXACT copy of the helper you provided.
    Inverts 4×4 or 3×4 [R|t] pose matrices and returns a full 4×4.
    """
    is_torch = isinstance(mat, torch.Tensor)
    h, w = mat.shape[-2:]
    if (h, w) == (4, 4):
        return torch.linalg.inv(mat) if is_torch else np.linalg.inv(mat)
    if (h, w) == (3, 4):
        if is_torch:
            R, t = mat[:, :3], mat[:, 3]
            Ri = R.transpose(-2, -1)
            ti = -Ri @ t.unsqueeze(-1)
            E = torch.eye(4, dtype=mat.dtype, device=mat.device)
            E[:3, :3], E[:3, 3] = Ri, ti.squeeze(-1)
            return E
        R, t = mat[:, :3], mat[:, 3]
        Ri, ti = R.T, -R.T @ t
        E = np.eye(4, dtype=mat.dtype)
        E[:3, :3], E[:3, 3] = Ri, ti
        return E
    raise ValueError(f"Can't invert matrix of shape {mat.shape}")


def build_pc(data: dict[str, np.ndarray], frame_idx: int) -> np.ndarray:
    """
    Byte-for-byte identical to `_pc_for_frame` in the original loader.
    Returns (N,4) float32 [x y z valid].
    """
    lengths, indices, coords = (
        data["track_lengths"],
        data["track_indices"],
        data["track_coordinates"],
    )
    num_tracks = len(lengths)
    pts = np.full((num_tracks, 3), np.nan, dtype=np.float32)
    valid = np.zeros((num_tracks, 1), dtype=np.float32)
    ptr = 0
    for i, L in enumerate(lengths):
        ids = indices[ptr : ptr + L]
        crd = coords[ptr : ptr + L]
        ptr += L
        m = ids == frame_idx
        if m.any():
            pts[i] = crd[m][0]
            valid[i] = 1
    return np.hstack([pts, valid])  # (N,4)

# ───────── data discovery (unchanged) ──────────────────────────────────────
def discover_sequences(cfg: DictConfig) -> list[str]:
    meta_csv  = os.path.join(cfg.dataset.stereo4d.meta_dir,
                             "stereo4d_id_to_time_and_fov_metadata.csv")
    stats_csv = os.path.join(cfg.dataset.stereo4d.meta_dir, "stats.csv")

    meta = pd.read_csv(meta_csv, header=0,
                       names=["vid","clipid","timestamp",
                              "start_yaw","end_yaw","start_tilt","end_tilt"])
    stats = pd.read_csv(stats_csv, skipinitialspace=True)
    stats = stats.query("displacement_percentage_50 > 0.10 and d_frame > 5*16")

    # ───────── additional split-based filtering (parallel) ────────────
    def _has_mp4(row):
        vid, cid = row["ytid"], row["clipid"]
        ts = meta.loc[(meta.vid == vid) & (meta.clipid == cid), "timestamp"]
        if ts.empty:
            return False
        seq = f"{vid}_{int(ts.values[0])}"
        return (Path(cfg.dataset.stereo4d.lefteye_dir) / SPLIT / f"{seq}-left_rectified.mp4").exists()

    exist_mask = Parallel(n_jobs=os.cpu_count())(delayed(_has_mp4)(row) for _, row in stats.iterrows())
    stats   = stats[exist_mask]
    # ──────────────────────────────────────────────────────────────────
    
    seqs = []
    for _, r in stats.iterrows():
        ts = meta.loc[(meta.vid == r["ytid"]) & (meta.clipid == r["clipid"]),
                      "timestamp"]
        if not ts.empty:
            seqs.append(f"{r['ytid']}_{int(ts.values[0])}")
    return sorted(set(seqs))

# ───────── exporter worker ─────────────────────────────────────────────────
def export_sequence(seq: str, cfg: DictConfig, out_dir: str, rank: int) -> int:
    """Write all frames of one sequence into this worker's shard set (persistent writer per rank)."""
    mp4 = os.path.join(cfg.dataset.stereo4d.lefteye_dir,
                       SPLIT, f"{seq}-left_rectified.mp4")
    npz = os.path.join(cfg.dataset.stereo4d.path, SPLIT, f"{seq}.npz")
    if not (os.path.isfile(mp4) and os.path.isfile(npz)):
        tqdm.write(f"missing assets for {seq}")
        return 0

    vr  = VideoReader(mp4, ctx=cpu(0))

    ann = np.load(npz, mmap_mode="r")
    data = {k: ann[k] for k in
            ("timestamps", "track_lengths","track_indices","track_coordinates","camera2world")}
    K = intrinsic_K(int(vr[0].shape[1]), cfg.dataset.stereo4d.hfov)
    
    lengths = data['track_lengths']
    shape = (len(lengths), len(data['timestamps']), 3)
    tracks = np.full(shape, np.nan)
    row_idx = np.repeat(np.arange(lengths.shape[0]), lengths)
    col_idx = data['track_indices']
    tracks[row_idx, col_idx] = data['track_coordinates']
    valid = (~np.isnan(tracks[..., 0])).astype(np.float32)[..., None]
    pcs = np.concatenate([tracks, valid], axis=-1).astype(np.float32)

    # ─── get (or create) persistent sink for this worker ───
    sink = _get_sink(rank, out_dir)

    written = 0
    for f_idx in range(len(vr)):
        rgb = vr[f_idx].asnumpy().astype(np.float32)
        pc = pcs[:, f_idx]
        extr = inv(data["camera2world"][f_idx]).astype(np.float32)
        key = f"{seq}_{f_idx:05d}"
        sink.write({
            "__key__":  key,
            "rgb.npy":  to_npy_bytes(rgb),
            "pc.npy":   to_npy_bytes(pc),
            "intr.npy": to_npy_bytes(K),
            "extr.npy": to_npy_bytes(extr),
        })
        written += 1
    return written

# ───────── joblib ↔ tqdm glue ───────────────────────────────────────────────
@contextmanager
def tqdm_joblib(tqdm_obj):
    old_cb = parallel.BatchCompletionCallBack
    class _TqdmCB(old_cb):
        def __call__(self, *a, **k):
            tqdm_obj.update(self.batch_size)
            return super().__call__(*a, **k)
    parallel.BatchCompletionCallBack = _TqdmCB
    try:
        yield tqdm_obj
    finally:
        parallel.BatchCompletionCallBack = old_cb
        tqdm_obj.close()

# ───────── baseline reader ─────────────────────────────────────────────────
def naive_get_frame(seq: str, frame: int, cfg: DictConfig):
    mp4 = os.path.join(cfg.dataset.stereo4d.lefteye_dir, SPLIT, f"{seq}-left_rectified.mp4")
    npz = os.path.join(cfg.dataset.stereo4d.path, SPLIT, f"{seq}.npz")

    cap = cv2.VideoCapture(mp4)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, rgb_bgr = cap.read()
    cap.release()

    ann = np.load(npz, mmap_mode="r")
    pc = build_pc({
        "track_lengths": ann["track_lengths"],
        "track_indices": ann["track_indices"],
        "track_coordinates": ann["track_coordinates"],
    }, frame)
    return (rgb_bgr[:, :, ::-1] if ok else None).astype(np.float32), pc.astype(np.float32)

# ───────── main ────────────────────────────────────────────────────────────
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print(f"ALLOW_PICKLE={ALLOW_PICKLE}")
    print(f"USE_THREADS={USE_THREADS}")
    print(f"SPLIT={SPLIT}")
    print(f"NUM_WORKERS={NUM_WORKERS}")
    print(f"TEST_N={TEST_N}")
    print(f"MAX_SHARD_SIZE_GB={MAX_SHARD_SIZE_GB}")
    print(f"MAX_SAMPLES_PER_SHARD={MAX_SAMPLES_PER_SHARD}")

    out_dir = Path(cfg.dataset.stereo4d.path) / "wds-temp" / SPLIT
    if out_dir.exists():
        for p in out_dir.glob("*.tar"):
            p.unlink()                      # delete every shard
        (out_dir / "stereo4d-idx.json").unlink(missing_ok=True)
        (out_dir / "key_to_idx.json").unlink(missing_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    clear_wids_cache()

    seqs = discover_sequences(cfg)
    print(f"full seqs: {seqs}")
    if TEST_N is not None:
        seqs = random.sample(seqs, min(TEST_N, len(seqs)))
        print(f"sampled seqs: {seqs}")
    tqdm.write(f"processing {len(seqs)} sequences → {out_dir}")

    # export
    with tqdm_joblib(tqdm(total=len(seqs), desc="export", ncols=80)):
        backend = "threading" if USE_THREADS else "loky"   # loky ⇒ separate processes
        written = Parallel(n_jobs=NUM_WORKERS, backend=backend)(
            delayed(export_sequence)(s, cfg, out_dir, i % NUM_WORKERS)
            for i, s in enumerate(seqs)
        )
    total_frames = sum(written)
    tqdm.write(f"wrote {total_frames:,} frames across {len(glob.glob(os.path.join(out_dir, '*.tar')))} shards")

    # index
    idx_json = os.path.join(out_dir, "stereo4d-idx.json")
    shard_glob_abs = os.path.abspath(os.path.join(out_dir, "stereo4d-*.tar"))
    tqdm.write(f"Indexing with {shard_glob_abs}")
    if os.path.exists(idx_json): os.remove(idx_json)
    os.system(f"widsindex create {shard_glob_abs} -o {idx_json}")
    tqdm.write(f"Indexed {shard_glob_abs} → {idx_json}")

    idx_ds = wids.ShardListDataset(idx_json, transformations=[])
    with open(idx_json) as f: dsdesc = json.load(f)
    entries = dsdesc.get("samples") or dsdesc.get("entries") or []
    if entries:
        key_to_idx = {e.get("key") or e.get("name"): e["index"] for e in tqdm(entries)}
    else:
        key_to_idx = {idx_ds[i]["__key__"]: i for i in tqdm(range(len(idx_ds)))}
    tqdm.write(f"Built key_to_idx map of size {len(key_to_idx)}")

    # persist mapping to disk and reload
    with open(os.path.join(out_dir, "key_to_idx.json"), "w") as f:
        json.dump(key_to_idx, f)
    tqdm.write(f"Saved key_to_idx map to {os.path.join(out_dir, 'key_to_idx.json')}")
    
    if TEST_N is None:
        return
    
    del key_to_idx
    gc.collect()

    with open(os.path.join(out_dir, "key_to_idx.json"), "r") as f:
        key_to_idx = json.load(f)
    get_sample = lambda k: idx_ds[key_to_idx[k]]
    pairs = random.sample(list(key_to_idx.keys()), TEST_N * 3)

    # speed benchmark
    tqdm.write(f"Benchmarking speed on keys: {pairs}")
    t0 = time.perf_counter()
    for k in tqdm(pairs, desc="fast"): get_sample(k)
    t_wds = (time.perf_counter() - t0) / len(pairs)
    t0 = time.perf_counter()
    for k in tqdm(pairs, desc="naive"):
        seq, _, fidx = k.rpartition("_")
        naive_get_frame(seq, int(fidx), cfg)
    t_naive = (time.perf_counter() - t0) / len(pairs)
    tqdm.write(f"avg wds={t_wds*1e3:.1f}ms naïve={t_naive*1e3:.1f}ms")

    # parity check
    mism = 0
    tqdm.write("Starting parity check …")
    for k in tqdm(pairs, desc="parity checking"):
        seq, _, fidx = k.rpartition("_")
        f = int(fidx)
        samp = get_sample(k)
        rgb_w = load_from_sample(samp.get(".rgb.npy"))
        pc_w  = load_from_sample(samp.get(".pc.npy"))

        rgb_n, pc_n = naive_get_frame(seq, f, cfg)
        # breakpoint()
    rgb_match = (rgb_n is not None
                 and rgb_w.shape == rgb_n.shape
                 and np.array_equal(rgb_w, rgb_n))     # exact-match uint8

    # allow tiny float32 differences & treat NaNs equal
    pc_match = (pc_w.shape == pc_n.shape
                and np.allclose(pc_w, pc_n,
                                atol=1e-6, rtol=1e-6,
                                equal_nan=True))

    if not (rgb_match and pc_match):
        mism += 1
        print(f"  MISMATCH at {k}: rgb_match={rgb_match}, pc_match={pc_match}")   
        print(f"pc_w: {pc_w}")
        print(f"pc_n: {pc_n}")
    tqdm.write(f"parity — {mism}/{len(pairs)} mismatches")

if __name__ == "__main__":
    main()