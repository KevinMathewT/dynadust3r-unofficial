#!/usr/bin/env python3
"""
Fast Stereo4D → WebDataset preprocessor (sequence-level samples).

Design:
- Discovery: CSV-based filter or scan-all, no decode; metadata-only width probe.
- Writing: per-process exclusive ShardWriter with unique shard patterns.
- Indexing: barrier then index all shards, build key→idx map.
- Optional verification & small benchmark; configurable via env toggles.

Run examples:
  Quick local:
    PRE_S4D_TEST_N=16 PRE_S4D_NUM_WORKERS=16 PRE_S4D_WRITE_MODE=per_process \
    poetry run python -m extras.preprocess_stereo4d dataset.stereo4d.wds_dir=/scratch/projects/fouheylab/km6748/stereo4d-data/wds-full

  High-throughput (Slurm; a100/h100):
    srun --partition=stake_a100_2 --nodes=1 --ntasks=1 --cpus-per-task=160 \
         --mem=0 --gres=gpu:a100:1 --time=24:00:00 bash -lc '
      export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
      PRE_S4D_USE_ALL_VIDEOS=1 PRE_S4D_SPLIT=train \
      PRE_S4D_NUM_WORKERS=${SLURM_CPUS_PER_TASK:-$(nproc)} PRE_S4D_DECORD_GPU=1 \
      PRE_S4D_WRITE_MODE=per_process PRE_S4D_MAX_SHARD_SIZE_GB=50 PRE_S4D_MAX_SAMPLES_PER_SHARD=8000 \
      PRE_S4D_CLOSE_EACH=0 PRE_S4D_VERIFY_N=0 \
      poetry run python -m extras.preprocess_stereo4d dataset.stereo4d.wds_dir=/scratch/projects/fouheylab/km6748/stereo4d-data/wds-full
    '
    # Swap to H100 if desired:
    #   --partition=stake_h100_1 --cpus-per-task=96 --gres=gpu:h100:1
    # CPU-only (no GPU decode): drop --gres and set PRE_S4D_DECORD_GPU=0
    # Tip: if dataset is on shared FS, copy to /scratch first for max I/O.

Toggles (env):
  PRE_S4D_USE_ALL_VIDEOS=0/1, PRE_S4D_SPLIT=train|test,
  PRE_S4D_NUM_WORKERS, PRE_S4D_TEST_N, PRE_S4D_VERIFY_N,
  PRE_S4D_MAX_SHARD_SIZE_GB, PRE_S4D_MAX_SAMPLES_PER_SHARD,
  PRE_S4D_WRITE_MODE=per_process|single_writer,
  PRE_S4D_CLOSE_EACH=0/1 (per-process close per sample),
  PRE_S4D_DECORD_GPU=0/1
"""

from __future__ import annotations

import os, io, math, json, time, logging, tempfile, shutil, contextlib
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from multiprocessing import Pool
import multiprocessing as mp
import queue

import webdataset as wds
from decord import VideoReader, cpu as decord_cpu

# ----------------------------------------------------------------------------
# Configurable defaults
# ----------------------------------------------------------------------------
ALLOW_PICKLE            = True
SPLIT                   = "train"
USE_ALL_VIDEOS          = False
NUM_WORKERS             = 32
TEST_N                  = None          # subset for export (None = all)
VERIFY_N                = 0             # number of seqs to verify (0 = skip)
MAX_SHARD_SIZE_GB       = 50
MAX_SAMPLES_PER_SHARD   = 8_000
WRITE_MODE              = "per_process"  # per_process | single_writer
CLOSE_EACH_SAMPLE       = False         # only for per_process; safest but slower
DECORD_GPU              = True
SPILL_THRESHOLD_BYTES   = 128 << 20     # for verify
CLEAN_SHARDS            = True          # remove existing stereo4d-*.tar & fragments on rerun

logging.basicConfig(level=logging.WARNING)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _parse_bool_env(val: str | None, default: bool) -> bool:
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _apply_env_overrides():
    global USE_ALL_VIDEOS, DECORD_GPU, WRITE_MODE, NUM_WORKERS, TEST_N, SPLIT
    global MAX_SHARD_SIZE_GB, MAX_SAMPLES_PER_SHARD, CLOSE_EACH_SAMPLE, VERIFY_N, CLEAN_SHARDS
    USE_ALL_VIDEOS = _parse_bool_env(os.environ.get("PRE_S4D_USE_ALL_VIDEOS"), USE_ALL_VIDEOS)
    DECORD_GPU = _parse_bool_env(os.environ.get("PRE_S4D_DECORD_GPU"), DECORD_GPU)
    CLOSE_EACH_SAMPLE = _parse_bool_env(os.environ.get("PRE_S4D_CLOSE_EACH"), CLOSE_EACH_SAMPLE)
    CLEAN_SHARDS = _parse_bool_env(os.environ.get("PRE_S4D_CLEAN_SHARDS"), CLEAN_SHARDS)
    if os.environ.get("PRE_S4D_SPLIT"): SPLIT = os.environ["PRE_S4D_SPLIT"]
    if os.environ.get("PRE_S4D_WRITE_MODE"): WRITE_MODE = os.environ["PRE_S4D_WRITE_MODE"]
    if os.environ.get("PRE_S4D_NUM_WORKERS"):
        with contextlib.suppress(Exception):
            NUM_WORKERS = int(os.environ["PRE_S4D_NUM_WORKERS"]) 
    if os.environ.get("PRE_S4D_TEST_N"):
        with contextlib.suppress(Exception):
            TEST_N = int(os.environ["PRE_S4D_TEST_N"]) 
    if os.environ.get("PRE_S4D_VERIFY_N"):
        with contextlib.suppress(Exception):
            VERIFY_N = int(os.environ["PRE_S4D_VERIFY_N"]) 
    if os.environ.get("PRE_S4D_MAX_SHARD_SIZE_GB"):
        with contextlib.suppress(Exception):
            MAX_SHARD_SIZE_GB = float(os.environ["PRE_S4D_MAX_SHARD_SIZE_GB"]) 
    if os.environ.get("PRE_S4D_MAX_SAMPLES_PER_SHARD"):
        with contextlib.suppress(Exception):
            MAX_SAMPLES_PER_SHARD = int(os.environ["PRE_S4D_MAX_SAMPLES_PER_SHARD"]) 


def _get_decord_ctx():
    if DECORD_GPU:
        try:
            import torch
            if torch.cuda.is_available():
                from decord import gpu as _gpu
                return _gpu(torch.cuda.current_device())
        except Exception:
            pass
    return decord_cpu(0)


def intrinsic_K(width: int, hfov: float) -> np.ndarray:
    fx = width * 0.5 / math.tan(math.radians(hfov) * 0.5)
    return np.array([[fx, 0, width/2], [0, fx, width/2], [0, 0, 1]], np.float32)


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
        vr = VideoReader(str(path), ctx=_get_decord_ctx())
        f0 = vr[0]
        return int(f0.shape[1]), int(f0.shape[0])
    except Exception:
        return None


def probe_width_height(path: str | os.PathLike) -> Tuple[int, int]:
    dims = _probe_dims_av(path) or _probe_dims_cv(path) or _probe_dims_decord(path)
    if dims is None:
        raise RuntimeError(f"Failed to probe video dimensions: {path}")
    return dims


def _probe_frame_count_cv(path: str | os.PathLike) -> int | None:
    try:
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n and n > 0:
            return n
    except Exception:
        return None
    return None


def _probe_frame_count_av(path: str | os.PathLike) -> int | None:
    try:
        import av  # type: ignore
        with av.open(str(path)) as c:
            for st in c.streams:
                if st.type == 'video':
                    if getattr(st, "frames", None) not in (None, 0):
                        return int(st.frames)
                    dur = getattr(st, "duration", None)
                    tbase = getattr(st, "time_base", None)
                    rate = getattr(st, "average_rate", None)
                    if dur is not None and tbase is not None and rate is not None:
                        seconds = float(dur * tbase)
                        est = int(seconds * float(rate))
                        if est > 0:
                            return est
    except Exception:
        return None
    return None


def _probe_frame_count_decord(path: str | os.PathLike) -> int | None:
    try:
        vr = VideoReader(str(path), ctx=_get_decord_ctx())
        return int(len(vr))
    except Exception:
        return None


def probe_frame_count(path: str | os.PathLike) -> int | None:
    """Fast frame count probe with minimal overhead and GPU-friendly fallback."""
    return (
        _probe_frame_count_cv(path)
        or _probe_frame_count_av(path)
        or _probe_frame_count_decord(path)
    )


def _cleanup_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    # remove indices and summaries
    (path / "stereo4d-idx.json").unlink(missing_ok=True)
    (path / "key_to_idx.json").unlink(missing_ok=True)
    (path / "frame_counts.csv").unlink(missing_ok=True)
    # remove stray temp/fragment files from previous runs
    for frag in list(path.glob("frame_counts-*-merged.tmp")):
        frag.unlink(missing_ok=True)
    for frag in list(path.glob("frame_counts-*-p*.csv")):
        frag.unlink(missing_ok=True)
    # optionally remove existing shards to ensure a clean rerun
    if CLEAN_SHARDS:
        for tar in list(path.glob("stereo4d-*.tar")):
            tar.unlink(missing_ok=True)


def discover_sequences(cfg: DictConfig) -> List[str]:
    if USE_ALL_VIDEOS:
        root = Path(cfg.dataset.stereo4d.lefteye_dir) / SPLIT
        mp4s = sorted(root.glob("*-left_rectified.mp4"))
        tqdm.write(f"discover: scanning {len(mp4s)} videos under {root}")
        seqs: List[str] = []
        for mp4_path in tqdm(mp4s, desc="discover (all videos)", total=len(mp4s), dynamic_ncols=True):
            seq = mp4_path.name[:-len("-left_rectified.mp4")]
            npz_path = Path(cfg.dataset.stereo4d.path) / SPLIT / f"{seq}.npz"
            if not npz_path.exists():
                continue
            # Fast path: only check presence and minimal size; detailed probing happens in workers
            try:
                if mp4_path.stat().st_size > 0 and npz_path.stat().st_size > 0:
                    seqs.append(seq)
            except Exception:
                continue
        return sorted(set(seqs))

    # CSV-guided
    meta_csv  = Path(cfg.dataset.stereo4d.meta_dir) / "stereo4d_id_to_time_and_fov_metadata.csv"
    stats_csv = Path(cfg.dataset.stereo4d.meta_dir) / "stats.csv"
    meta = pd.read_csv(meta_csv, header=0,
                       names=["vid","clipid","timestamp","start_yaw","end_yaw","start_tilt","end_tilt"])
    stats = pd.read_csv(stats_csv, skipinitialspace=True)
    stats = stats.query("displacement_percentage_50 > 0.10 and d_frame > 5*16")

    def _keep(row) -> bool:
        vid, cid = row["ytid"], row["clipid"]
        ts = meta.loc[(meta.vid == vid) & (meta.clipid == cid), "timestamp"]
        if ts.empty:
            return False
        seq = f"{vid}_{int(ts.values[0])}"
        mp4_path = Path(cfg.dataset.stereo4d.lefteye_dir) / SPLIT / f"{seq}-left_rectified.mp4"
        npz_path = Path(cfg.dataset.stereo4d.path) / SPLIT / f"{seq}.npz"
        if not (mp4_path.exists() and npz_path.exists()):
            return False
        # Fast presence/size check only; decoding/width probing deferred to writer stage
        try:
            return mp4_path.stat().st_size > 0 and npz_path.stat().st_size > 0
        except Exception:
            return False

    mask_vals: List[bool] = []
    for _, row in tqdm(stats.iterrows(), total=len(stats), desc="discover (csv check)", dynamic_ncols=True):
        mask_vals.append(_keep(row))
    exist_mask = pd.Series(mask_vals, index=stats.index)
    stats = stats[exist_mask]
    seqs: List[str] = []
    for _, r in tqdm(stats.iterrows(), total=len(stats), desc="discover (collect)", dynamic_ncols=True):
        ts = meta.loc[(meta.vid == r["ytid"]) & (meta.clipid == r["clipid"]), "timestamp"]
        if not ts.empty:
            seqs.append(f"{r['ytid']}_{int(ts.values[0])}")
    return sorted(set(seqs))


def _open_vr_from_blob(vbytes: bytes):
    if len(vbytes) > SPILL_THRESHOLD_BYTES:
        tf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=True)
        tf.write(vbytes)
        tf.flush()
        try:
            vr = VideoReader(tf.name, ctx=_get_decord_ctx())
        except Exception:
            vr = VideoReader(tf.name, ctx=decord_cpu(0))
        vr._tmpfile = tf
        return vr
    try:
        return VideoReader(io.BytesIO(vbytes), ctx=_get_decord_ctx())
    except Exception:
        return VideoReader(io.BytesIO(vbytes), ctx=decord_cpu(0))


# ----------------------------------------------------------------------------
# Writing workers
# ----------------------------------------------------------------------------
def _process_chunk(args) -> int:
    seqs, cfg_dict, out_dir_str, jobid, worker_id, progress_queue = args
    # lightweight DictConfig passthrough
    class _Cfg:
        pass
    cfg = _Cfg()
    cfg.dataset = _Cfg()
    cfg.dataset.stereo4d = _Cfg()
    for k, v in cfg_dict.items():
        setattr(cfg.dataset.stereo4d, k, v)

    out_dir = Path(out_dir_str)
    pid = os.getpid()
    pattern = out_dir / f"stereo4d-{jobid}-p{pid}-%06d.tar"
    writer = None
    total_written = 0

    # per-process progress bar
    pbar = tqdm(total=len(seqs), desc=f"worker-{worker_id}", position=worker_id+1,
                leave=False, dynamic_ncols=True)

    # per-process frame count CSV
    frame_csv_path = out_dir / f"frame_counts-{jobid}-p{pid}.csv"
    try:
        frame_csv_fh = open(frame_csv_path, "w")
    except Exception:
        frame_csv_fh = None

    try:
        for seq in seqs:
            try:
                mp4 = Path(cfg.dataset.stereo4d.lefteye_dir) / SPLIT / f"{seq}-left_rectified.mp4"
                npz = Path(cfg.dataset.stereo4d.path) / SPLIT / f"{seq}.npz"
                if not (mp4.is_file() and npz.is_file()):
                    continue

                w, _h = probe_width_height(mp4)
                # Fast frame count probe (no decode)
                n_frames = probe_frame_count(mp4)
                if n_frames is None or n_frames <= 0:
                    with contextlib.suppress(Exception):
                        ann_np = np.load(npz, allow_pickle=True)
                        if "camera2world" in ann_np:
                            n_frames = int(ann_np["camera2world"].shape[0])
                        elif "track_indices" in ann_np:
                            n_frames = int(ann_np["track_indices"].max()) + 1
                K = intrinsic_K(int(w), cfg.dataset.stereo4d.hfov)

                with open(mp4, "rb") as f:
                    video_data = f.read()
                with open(npz, "rb") as f:
                    ann_data = f.read()

                if len(video_data) == 0 or len(ann_data) == 0:
                    continue

                buf = io.BytesIO()
                np.save(buf, K.astype(np.float32))
                intr_bytes = buf.getvalue()
                if len(intr_bytes) == 0:
                    continue

                sample = {
                    "__key__": seq,
                    "video.mp4": video_data,
                    "ann.npz": ann_data,
                    "intr.npy": intr_bytes,
                }

                if writer is None:
                    writer = wds.ShardWriter(str(pattern), maxcount=MAX_SAMPLES_PER_SHARD, maxsize=MAX_SHARD_SIZE_GB*1e9)

                writer.write(sample)
                total_written += 1

                if CLOSE_EACH_SAMPLE and writer is not None:
                    writer.close(); writer = None

                # Write frame count row
                if frame_csv_fh is not None:
                    with contextlib.suppress(Exception):
                        frame_csv_fh.write(f"{seq},{int(n_frames) if n_frames else -1},{int(w)},{int(_h)}\n")

            except Exception as e:
                tqdm.write(f"[WORKER-ERROR] {seq}: {type(e).__name__}: {e}")
                continue
            finally:
                # advance both per-worker and global progress
                with contextlib.suppress(Exception):
                    pbar.update(1)
                with contextlib.suppress(Exception):
                    progress_queue.put(1)
    finally:
        if writer is not None:
            with contextlib.suppress(Exception):
                writer.close()
        if 'frame_csv_fh' in locals() and frame_csv_fh is not None:
            with contextlib.suppress(Exception):
                frame_csv_fh.close()
        with contextlib.suppress(Exception):
            pbar.close()

    return total_written


# ----------------------------------------------------------------------------
# Indexing & verify
# ----------------------------------------------------------------------------
def build_index_and_map(out_dir: Path) -> Tuple[str, dict]:
    idx_json = out_dir / "stereo4d-idx.json"
    shard_glob = str(out_dir / "stereo4d-*.tar")
    if idx_json.exists():
        idx_json.unlink()
    os.system(f"widsindex create {shard_glob} -o {idx_json}")

    import wids
    idx_ds = wids.ShardListDataset(str(idx_json), transformations=[])
    with open(idx_json) as f:
        desc = json.load(f)
    entries = desc.get("samples") or desc.get("entries") or []
    if entries:
        key_to_idx = {e.get("key") or e.get("name"): e["index"] for e in tqdm(entries)}
    else:
        key_to_idx = {idx_ds[i]["__key__"]: i for i in tqdm(range(len(idx_ds)))}
    with open(out_dir / "key_to_idx.json", "w") as f:
        json.dump(key_to_idx, f)
    return str(idx_json), key_to_idx


def _merge_frame_count_csvs(out_dir: Path, jobid: int) -> Path:
    """Merge per-process frame count CSV fragments into a single CSV with header.

    Returns the final CSV path.
    """
    final_csv = out_dir / "frame_counts.csv"
    tmp_csv = out_dir / f"frame_counts-{jobid}-merged.tmp"

    frags = sorted(out_dir.glob(f"frame_counts-{jobid}-p*.csv"))
    # Always write header
    with open(tmp_csv, "w") as out_f:
        out_f.write("seq,d_frame,width,height\n")
        for frag in frags:
            try:
                with open(frag, "r") as in_f:
                    shutil.copyfileobj(in_f, out_f)
            except Exception:
                continue

    # Atomic finalize
    tmp_csv.replace(final_csv)

    # Clean up fragments
    for frag in frags:
        frag.unlink(missing_ok=True)

    return final_csv


def quick_benchmark(cfg: DictConfig, keys: List[str], key_to_idx: dict, idx_json: str):
    import wids
    idx_ds = wids.ShardListDataset(idx_json, transformations=[])
    pairs = keys
    t0 = time.perf_counter()
    for s in pairs:
        _ = idx_ds[key_to_idx[s]]
    t_fast = (time.perf_counter() - t0) / max(len(pairs), 1)

    t0 = time.perf_counter()
    for s in pairs:
        open(Path(cfg.dataset.stereo4d.lefteye_dir) / SPLIT / f"{s}-left_rectified.mp4", "rb").read(1)
    t_naive = (time.perf_counter() - t0) / max(len(pairs), 1)
    tqdm.write(f"avg webdataset seek={t_fast*1e3:.1f} ms  disk open={t_naive*1e3:.1f} ms")


def verify_samples(idx_json: str, key_to_idx: dict, verify_n: int):
    if verify_n <= 0:
        return
    import wids
    idx_ds = wids.ShardListDataset(idx_json, transformations=[])
    bad: List[Tuple[str, str]] = []
    keys = list(key_to_idx.keys())[:verify_n]
    for s in tqdm(keys, desc="verify"):
        try:
            sample = idx_ds[key_to_idx[s]]
        except Exception as e:
            bad.append((s, f"tar read: {e}")); continue
        try:
            vdat = sample.get(".video.mp4")
            if vdat is None: bad.append((s, f"missing .video.mp4 key, has: {list(sample.keys())}")); continue
            vbytes = vdat.read() if hasattr(vdat, "read") else vdat
            if len(vbytes) == 0: bad.append((s, "video empty")); continue
            vr = _open_vr_from_blob(vbytes); _ = vr[0]
        except Exception as e:
            bad.append((s, f"video decode: {type(e).__name__}: {e}"))
        try:
            import numpy as _np
            _np.load(sample[".ann.npz"], allow_pickle=True)
        except Exception as e:
            bad.append((s, f"ann npz: {e}"))
        try:
            import numpy as _np
            _np.load(sample[".intr.npy"], allow_pickle=True)
        except Exception as e:
            bad.append((s, f"intr npy: {e}"))
    if bad:
        tqdm.write(f"\n{len(bad)} / {len(keys)} sequences failed verification:")
        for seq, msg in bad:
            tqdm.write(f"  {seq}: {msg}")
    else:
        tqdm.write("\nverification passed ✔︎")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    _apply_env_overrides()

    # set WIDS cache from config if provided; fallback to scratch tmp
    os.environ.setdefault("TMPDIR", "/scratch/km6748/tmp")
    os.environ.setdefault("TMP",    "/scratch/km6748/tmp")
    os.environ.setdefault("TEMP",   "/scratch/km6748/tmp")
    cache_dir = str(getattr(cfg.dataset.stereo4d, "cache", os.path.join(os.environ.get("TMP", "/tmp"), "_wids_cache")))
    os.environ.setdefault("WIDS_CACHE", cache_dir)
    Path(os.environ["WIDS_CACHE"]).mkdir(parents=True, exist_ok=True)

    out_dir = Path(cfg.dataset.stereo4d.wds_dir) / SPLIT
    _cleanup_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"SPLIT={SPLIT}")
    print(f"USE_ALL_VIDEOS={USE_ALL_VIDEOS}")
    print(f"WRITE_MODE={WRITE_MODE}")
    print(f"NUM_WORKERS={NUM_WORKERS}")
    print(f"TEST_N={TEST_N}")
    print(f"VERIFY_N={VERIFY_N}")
    print(f"MAX_SHARD_SIZE_GB={MAX_SHARD_SIZE_GB}")
    print(f"MAX_SAMPLES_PER_SHARD={MAX_SAMPLES_PER_SHARD}")

    seqs = discover_sequences(cfg)
    if TEST_N is not None:
        import random
        random.seed(0)
        seqs = random.sample(seqs, min(TEST_N, len(seqs)))
    tqdm.write(f"processing {len(seqs)} sequences → {out_dir}")

    # partition into chunks per worker
    if len(seqs) == 0:
        tqdm.write("no sequences found; exiting")
        return
    chunks: List[List[str]] = []
    for i in range(NUM_WORKERS):
        chunks.append([])
    for j, s in enumerate(seqs):
        chunks[j % NUM_WORKERS].append(s)
    chunks = [c for c in chunks if c]

    # thin cfg dict for workers (avoid pickling Hydra objects)
    cfg_dict = dict(
        path=str(cfg.dataset.stereo4d.path),
        lefteye_dir=str(cfg.dataset.stereo4d.lefteye_dir),
        meta_dir=str(cfg.dataset.stereo4d.meta_dir),
        hfov=float(cfg.dataset.stereo4d.hfov),
    )

    jobid = int(time.time())

    if WRITE_MODE == "per_process":
        manager = mp.Manager()
        progress_queue = manager.Queue()
        total_items = sum(len(c) for c in chunks)
        with Pool(processes=len(chunks)) as pool:
            tasks = []
            for worker_id, c in enumerate(chunks):
                tasks.append(pool.apply_async(_process_chunk, [(c, cfg_dict, str(out_dir), jobid, worker_id, progress_queue)]))

            overall = tqdm(total=total_items, desc="overall", position=0, dynamic_ncols=True)
            finished = 0
            while finished < len(tasks):
                try:
                    n = progress_queue.get(timeout=0.1)
                    overall.update(n)
                except queue.Empty:
                    pass
                finished = sum(1 for t in tasks if t.ready())

            # Drain any remaining updates
            while True:
                try:
                    n = progress_queue.get_nowait()
                    overall.update(n)
                except queue.Empty:
                    break
            with contextlib.suppress(Exception):
                overall.close()

            written_counts = [t.get() for t in tasks]
    else:
        # single_writer mode: fallback to serial write with a single per-process bar
        # Local queue to keep signature compatibility
        _local_q = mp.Queue()
        written_counts = [_process_chunk((seqs, cfg_dict, str(out_dir), jobid, 0, _local_q))]

    total_written = int(sum(written_counts))
    tqdm.write(f"wrote {total_written:,} sequence-level samples")

    # index & keymap
    idx_json, key_to_idx = build_index_and_map(out_dir)
    tqdm.write(f"Indexed shards → {idx_json}")

    # merge per-process frame count CSVs
    final_fc = _merge_frame_count_csvs(out_dir, jobid)
    tqdm.write(f"Frame counts → {final_fc}")

    # quick benchmark & optional verify
    keys = list(key_to_idx.keys())
    keys_small = keys[: min(VERIFY_N, len(keys))] if VERIFY_N and VERIFY_N > 0 else keys[: min(32, len(keys))]
    if len(keys_small) > 0:
        quick_benchmark(cfg, keys_small, key_to_idx, idx_json)
    if VERIFY_N and VERIFY_N > 0:
        verify_samples(idx_json, key_to_idx, VERIFY_N)


if __name__ == "__main__":
    main()


