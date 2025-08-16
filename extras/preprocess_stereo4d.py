#!/usr/bin/env python3
"""
preprocess_stereo4d.py
----------------------
NEW LAYOUT (sequence-level samples)
• one WebDataset sample per sequence
• stores the raw video, the raw annotation .npz and intrinsics
The rest of the pipeline (indexing, benchmarking) is kept identical,
except the benchmark now uses `seqs` directly instead of regenerating
keys from the index.
"""

from __future__ import annotations
import os

# put all python tempfiles and wids cache on your scratch
os.environ.setdefault("TMPDIR", "/scratch/km6748/tmp")
os.environ.setdefault("TMP",    "/scratch/km6748/tmp")
os.environ.setdefault("TEMP",   "/scratch/km6748/tmp")
# ensure wids's cache dir lives under scratch
os.environ.setdefault("WIDS_CACHE", os.path.join(os.environ["TMP"], "_wids_cache"))

# create the cache directory if it doesn't exist
os.makedirs(os.environ["WIDS_CACHE"], exist_ok=True)

# create a symlink so /tmp/shard.tar lives on scratch
scratch_tmp = "/scratch/km6748/tmp"
os.makedirs(scratch_tmp, exist_ok=True)
shard_link = "/tmp/shard.tar"
shard_target = os.path.join(scratch_tmp, "shard.tar")

# remove any existing file or link, then symlink
if os.path.exists(shard_link) or os.path.islink(shard_link):
    os.remove(shard_link)
os.symlink(shard_target, shard_link)

from utils.train_utils import create_symlink_for_wids_cache, seed_everything

import gc
import os, json, math, time, random, logging, io, contextlib, tarfile
import cv2, numpy as np, pandas as pd, torch
from decord import VideoReader, cpu
from tqdm import tqdm
from pathlib import Path
import webdataset as wds, wids
import hydra
from omegaconf import DictConfig
import shutil, tempfile

from functools import partial
from multiprocessing import Pool, current_process, Lock
from multiprocessing.managers import SyncManager

# suppress noisy tarfile _Stream errors
def _silent_stream_del(self):            # type: ignore[func-no-untyped-def]
    with contextlib.suppress(Exception):
        if getattr(self, "fileobj", None):
            self.fileobj = None
tarfile._Stream.__del__ = _silent_stream_del  # type: ignore[attr-defined]

def clear_wids_cache():
    cache_dir = Path(tempfile.gettempdir()) / "_wids_cache"
    if cache_dir.is_dir():
        shutil.rmtree(cache_dir)
        print(f"cleared wids cache at {cache_dir}")

# editable constants
ALLOW_PICKLE          = True
USE_THREADS           = False
SPLIT                 = "train"
NUM_WORKERS           = 32
TEST_N                = None
MAX_SHARD_SIZE_GB     = 50
MAX_SAMPLES_PER_SHARD = 8_000
logging.basicConfig(level=logging.WARNING)

# shard writer and manager for multiprocessing
class MyShardWriter(wds.ShardWriter):
    def __init__(self, pattern, maxcount=MAX_SAMPLES_PER_SHARD, maxsize=MAX_SHARD_SIZE_GB*1e9, **kw):
        super().__init__(pattern, maxcount=maxcount, maxsize=maxsize, **kw)
        self.verbose = False
    def get_shards(self):
        return self.shard
    def get_count(self):
        return self.count if self.count < self.maxcount else 0
    def get_total(self):
        return self.total

class MyManager(SyncManager):
    pass

# Register shared classes
MyManager.register('MyShardWriter', MyShardWriter)
MyManager.register('Lock', Lock)

# helper funcs (unchanged)
def intrinsic_K(width: int, hfov: float) -> np.ndarray:
    fx = width * 0.5 / math.tan(math.radians(hfov) * 0.5)
    return np.array([[fx, 0, width/2], [0, fx, width/2], [0, 0, 1]], np.float32)

# data discovery (unchanged)
def discover_sequences(cfg: DictConfig) -> list[str]:
    meta_csv  = os.path.join(cfg.dataset.stereo4d.meta_dir,
                             "stereo4d_id_to_time_and_fov_metadata.csv")
    stats_csv = os.path.join(cfg.dataset.stereo4d.meta_dir, "stats.csv")

    meta = pd.read_csv(meta_csv, header=0,
                       names=["vid","clipid","timestamp",
                              "start_yaw","end_yaw","start_tilt","end_tilt"])
    stats = pd.read_csv(stats_csv, skipinitialspace=True)
    stats = stats.query("displacement_percentage_50 > 0.10 and d_frame > 5*16")

    def _has_mp4(row):
        vid, cid = row["ytid"], row["clipid"]
        ts = meta.loc[(meta.vid == vid) & (meta.clipid == cid), "timestamp"]
        if ts.empty: return False
        seq = f"{vid}_{int(ts.values[0])}"
        mp4_path = Path(cfg.dataset.stereo4d.lefteye_dir) / SPLIT / f"{seq}-left_rectified.mp4"
        npz_path = Path(cfg.dataset.stereo4d.path) / SPLIT / f"{seq}.npz"
        if not (mp4_path.exists() and npz_path.exists()):
            return False
        # probe that the mp4 is actually readable
        try:
            VideoReader(str(mp4_path), ctx=cpu(0))  # will raise if corrupt
            return True
        except Exception:
            return False

    exist_mask = pd.Series([_has_mp4(row) for _, row in stats.iterrows()],
                           index=stats.index)
    stats = stats[exist_mask]

    seqs = []
    for _, r in stats.iterrows():
        ts = meta.loc[(meta.vid == r["ytid"]) & (meta.clipid == r["clipid"]), "timestamp"]
        if not ts.empty:
            seqs.append(f"{r['ytid']}_{int(ts.values[0])}")
    return sorted(set(seqs))

def export_sequence(seq: str, cfg: DictConfig, out_dir: str, shared_sink, shared_lock) -> int:
    try:
        mp4 = os.path.join(cfg.dataset.stereo4d.lefteye_dir, SPLIT, f"{seq}-left_rectified.mp4")
        npz = os.path.join(cfg.dataset.stereo4d.path, SPLIT, f"{seq}.npz")
        
        if not (os.path.isfile(mp4) and os.path.isfile(npz)):
            tqdm.write(f"[MISSING] {seq}: missing assets")
            return 0

        # Video validation
        try:
            vr = VideoReader(mp4, ctx=cpu(0))
            K = intrinsic_K(int(vr[0].shape[1]), cfg.dataset.stereo4d.hfov)
            del vr
        except Exception as e:
            tqdm.write(f"[VIDEO-ERROR] {seq}: {e}")
            return 0

        # File reading with explicit error handling and size checking
        try:
            with open(mp4, "rb") as f:
                video_data = f.read()
            with open(npz, "rb") as f:
                ann_data = f.read()
                
            # DEBUG: Check data sizes
            video_size = len(video_data)
            ann_size = len(ann_data)
            
            if video_size == 0:
                tqdm.write(f"[ZERO-VIDEO] {seq}: video file is 0 bytes")
                return 0
            if ann_size == 0:
                tqdm.write(f"[ZERO-ANN] {seq}: annotation file is 0 bytes") 
                return 0
                
            # DEBUG: Log data sizes for problematic sequences
            problematic = ['8BpB5PhtDQ8_15081748', 'GyOUUDMnENE_255321989', 'JMIcmLexxjA_519085752', 
                          'OhPOB7M-_6U_108508509', 'bpacvEMOgjc_188588589', 'jt9tYywMCf0_15566667',
                          'ppg9V8JMo1E_505605606', 'zyv5UYyVZVA_358625292', 'bzkC7Cay_4A_698732065']
            if seq in problematic:
                tqdm.write(f"[DEBUG] {seq}: video={video_size:,} bytes, ann={ann_size:,} bytes")
                
        except Exception as e:
            tqdm.write(f"[FILE-READ-ERROR] {seq}: {e}")
            return 0

        # Buffer creation with size checking
        try:
            buf = io.BytesIO()
            np.save(buf, K.astype(np.float32))
            intr_bytes = buf.getvalue()
            
            intr_size = len(intr_bytes)
            if intr_size == 0:
                tqdm.write(f"[ZERO-INTR] {seq}: intrinsics buffer is 0 bytes")
                return 0
                
            # DEBUG: Log intrinsics size for problematic sequences  
            if seq in problematic:
                tqdm.write(f"[DEBUG] {seq}: intrinsics={intr_size:,} bytes")
                
        except Exception as e:
            tqdm.write(f"[BUFFER-ERROR] {seq}: {e}")
            return 0

        # Sample creation with validation
        try:
            sample = {
                "__key__": seq,
                "video.mp4": video_data,
                "ann.npz": ann_data,
                "intr.npy": intr_bytes,
            }
            
            # DEBUG: Validate sample contents
            if seq in problematic:
                tqdm.write(f"[DEBUG] {seq}: sample keys={list(sample.keys())}")
                tqdm.write(f"[DEBUG] {seq}: sample sizes={[len(v) if isinstance(v, (bytes, bytearray)) else type(v) for k,v in sample.items() if k != '__key__']}")
                
        except Exception as e:
            tqdm.write(f"[SAMPLE-CREATE-ERROR] {seq}: {e}")
            return 0

        # Sink operations using shared objects
        try:
            # Add pre-write validation for problematic sequences
            if seq in problematic:
                tqdm.write(f"[DEBUG-WRITE] {seq}: about to write sample with keys {list(sample.keys())}")
                tqdm.write(f"[DEBUG-WRITE] {seq}: video size={len(sample['video.mp4'])}, ann size={len(sample['ann.npz'])}")

            shared_lock.acquire()
            try:
                shared_sink.write(sample)
                
                # Post-write validation for problematic sequences
                if seq in problematic:
                    tqdm.write(f"[DEBUG-WRITE] {seq}: write completed successfully")
                    
                pbar = globals().get("_pbar")
                if pbar is None:
                    globals()["_pbar"] = tqdm(total=None, desc="export", position=0)
                    pbar = globals()["_pbar"]
                pbar.update(1)
                pbar.set_postfix_str(f"shard {shared_sink.get_shards()}")
            finally:
                shared_lock.release()
                
        except Exception as e:
            tqdm.write(f"[WRITE-ERROR] {seq}: {e}")
            import traceback
            tqdm.write(f"[WRITE-TRACEBACK] {seq}: {traceback.format_exc()}")
            return 0

        return 1

    except Exception as e:
        tqdm.write(f"[UNKNOWN-ERROR] {seq}: {type(e).__name__}: {e}")
        return 0


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    create_symlink_for_wids_cache()
    seed_everything(cfg.seed)

    global ALLOW_PICKLE, USE_THREADS, SPLIT, NUM_WORKERS, TEST_N, MAX_SHARD_SIZE_GB, MAX_SAMPLES_PER_SHARD

    print(f"ALLOW_PICKLE={ALLOW_PICKLE}")
    print(f"USE_THREADS={USE_THREADS}")
    print(f"SPLIT={SPLIT}")
    print(f"NUM_WORKERS={NUM_WORKERS}")
    print(f"TEST_N={TEST_N}")
    print(f"MAX_SHARD_SIZE_GB={MAX_SHARD_SIZE_GB}")
    print(f"MAX_SAMPLES_PER_SHARD={MAX_SAMPLES_PER_SHARD}")

    out_dir = Path(cfg.dataset.stereo4d.path) / "wds" / SPLIT
    if out_dir.exists():
        for p in out_dir.glob("*.tar"):
            p.unlink()
        (out_dir / "stereo4d-idx.json").unlink(missing_ok=True)
        (out_dir / "key_to_idx.json").unlink(missing_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    clear_wids_cache()

    seqs = discover_sequences(cfg)
    if TEST_N is not None:
        seqs = random.sample(seqs, min(TEST_N, len(seqs)) if TEST_N is not None else len(seqs))
    tqdm.write(f"processing {len(seqs)} sequences → {out_dir}")

    # Create shared manager and process with shared ShardWriter
    with MyManager() as manager:
        shared_sink = manager.MyShardWriter(
            pattern=str(out_dir / "stereo4d-%06d.tar"),
            maxcount=MAX_SAMPLES_PER_SHARD,
            maxsize=MAX_SHARD_SIZE_GB * 1e9
        )
        shared_lock = manager.Lock()
        
        with Pool(processes=NUM_WORKERS) as pool:
            written = pool.starmap(
                export_sequence,
                [(seq, cfg, str(out_dir), shared_sink, shared_lock) for seq in seqs]
            )

        shared_sink.close()

    total = sum(written)
    tqdm.write(f"wrote {total:,} sequence-level samples")

    idx_json       = os.path.join(out_dir, "stereo4d-idx.json")
    shard_glob_abs = os.path.abspath(os.path.join(out_dir, "stereo4d-*.tar"))
    tqdm.write(f"Indexing with {shard_glob_abs}")
    if os.path.exists(idx_json):
        os.remove(idx_json)
    os.system(f"widsindex create {shard_glob_abs} -o {idx_json}")
    tqdm.write(f"Indexed {shard_glob_abs} → {idx_json}")

    idx_ds = wids.ShardListDataset(idx_json, transformations=[])
    with open(idx_json) as f:
        desc = json.load(f)
    entries = desc.get("samples") or desc.get("entries") or []
    if entries:
        key_to_idx = {e.get("key") or e.get("name"): e["index"] for e in tqdm(entries)}
    else:
        key_to_idx = {idx_ds[i]["__key__"]: i for i in tqdm(range(len(idx_ds)))}
    tqdm.write(f"Built key_to_idx map of size {len(key_to_idx)}")

    with open(out_dir / "key_to_idx.json", "w") as f:
        json.dump(key_to_idx, f)
    tqdm.write(f"Saved key_to_idx map to {out_dir / 'key_to_idx.json'}")

    # if TEST_N is None:
    #     return

    del key_to_idx
    gc.collect()

    # TEST_N = 100000000

    with open(out_dir / "key_to_idx.json", "r") as f:
        key_to_idx = json.load(f)

    written_seqs = [s for s in seqs if s in key_to_idx]
    pairs = random.sample(written_seqs, min(TEST_N, len(written_seqs)) if TEST_N is not None else len(written_seqs))
    get_sample = lambda k: idx_ds[key_to_idx[k]]

    tqdm.write(f"Testing with {len(pairs)} sequences (out of {len(seqs)} discovered, {len(written_seqs)} written)")

    t0 = time.perf_counter()
    for s in tqdm(pairs, desc="fast"):
        get_sample(s)
    t_fast = (time.perf_counter() - t0) / len(pairs)

    t0 = time.perf_counter()
    for s in tqdm(pairs, desc="naive"):
        open(os.path.join(cfg.dataset.stereo4d.lefteye_dir, SPLIT, f"{s}-left_rectified.mp4"), "rb").read(1)
    t_naive = (time.perf_counter() - t0) / len(pairs)

    tqdm.write(f"avg webdataset seek={t_fast*1e3:.1f} ms  disk open={t_naive*1e3:.1f} ms")

    # ❷ read every sequence **fully** (video + ann + intr) and report failures
    # -----------------------------------------------------------------------
    def _safe_load_npy(blob):
        """
        Robust .npy loader that copes with the three payload types returned
        by WebDataset:
          • raw bytes / bytearray / memoryview
          • TarSubFile-like objects (have .read())
          • ordinary file paths
        """
        if isinstance(blob, (bytes, bytearray, memoryview)):
            return np.load(io.BytesIO(bytes(blob)), allow_pickle=True)
        if hasattr(blob, "read"):
            return np.load(io.BytesIO(blob.read()), allow_pickle=True)
        return np.load(blob, allow_pickle=True)

    bad: list[tuple[str, str]] = []
    check_seqs = seqs if TEST_N is None else pairs     # respect TEST_N limit

    for s in tqdm(check_seqs, desc="verify-all"):
        if s not in key_to_idx:
            bad.append((s, "missing in key_to_idx"))
            continue

        # try to pull the raw sample
        try:
            sample = idx_ds[key_to_idx[s]]
        except Exception as e:
            bad.append((s, f"tar read: {e}"))
            continue

        # --- video ------------------------------------------------------------------
        try:
            if ".video.mp4" not in sample:
                bad.append((s, f"missing .video.mp4 key, has: {list(sample.keys())}"))
                continue
            
            vdat = sample[".video.mp4"]
            vbytes = vdat.read() if hasattr(vdat, "read") else vdat
            
            if len(vbytes) == 0:
                bad.append((s, "video data is empty (0 bytes)"))
                continue
                
            vr = VideoReader(io.BytesIO(vbytes), ctx=cpu(0))
            _ = vr[0]
        except Exception as e:
            bad.append((s, f"video decode: {type(e).__name__}: {str(e)}"))

        # --- annotation -------------------------------------------------------------
        try:
            np.load(sample[".ann.npz"], allow_pickle=True)
        except Exception as e:
            bad.append((s, f"ann npz: {e}"))
            continue

        # --- intrinsics -------------------------------------------------------------
        try:
            _safe_load_npy(sample[".intr.npy"])
        except Exception as e:
            bad.append((s, f"intrinsics npy: {e}"))
            continue
        # sample OK -----------------------------------------------------------

    # summary -----------------------------------------------------------------
    if bad:
        tqdm.write(f"\n{len(bad)} / {len(check_seqs)} sequences failed to load fully:")
        for seq, msg in bad:
            tqdm.write(f"  {seq}: {msg}")
    else:
        tqdm.write("\nall checked sequences loaded successfully ✔︎")


if __name__ == "__main__":
    main()
