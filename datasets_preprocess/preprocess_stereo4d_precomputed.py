#!/usr/bin/env python3
"""
preprocess_stereo4d_triplets.py
-------------------------------
Convert stereo4d dataset to webdataset format where each sample is a complete triplet.
Uses parallel processing with no shared state between workers.
"""

import os, json, io, time, glob, tempfile, shutil
import cv2, numpy as np, pandas as pd
from decord import VideoReader, cpu
from pathlib import Path
import webdataset as wds
import hydra
from omegaconf import DictConfig, open_dict
from tqdm import tqdm
import atexit

# ─────────────── Configuration ─────────────────────────────────────────
SPLIT = "train"            # "train" | "test"
NUM_WORKERS = 32           # number of parallel workers
ALLOW_PICKLE = True
MAX_SHARD_SIZE_GB = 10
MAX_SAMPLES_PER_SHARD = 5000

N_SAMPLES = None           # ← NEW: None → cfg-based lengths
USE_COMPRESSION = False    # ← NEW: True → .npz w/ np.savez_compressed
EXT = "npz" if USE_COMPRESSION else "npy"      # helper
# ────────────────────────────────────────────────────────────────────────



# Track persistent writers
_writers = {}

def _get_sink(rank: int, out_dir: str) -> wds.ShardWriter:
    """Get or create a ShardWriter for this worker rank"""
    sink = _writers.get(rank)
    if sink is None:
        shard_tmpl = os.path.join(out_dir, f"stereo4d-triplets-{rank:02d}-%06d.tar")
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

def to_bytes(arr: np.ndarray) -> bytes:
    """serialize array as .npy  or compressed .npz, depending on USE_COMPRESSION"""
    buf = io.BytesIO()
    if USE_COMPRESSION:
        np.savez_compressed(buf, arr=arr, allow_pickle=ALLOW_PICKLE)   # (arr)
    else:
        np.save(buf, arr, allow_pickle=ALLOW_PICKLE)
    return buf.getvalue()


def compress_track_data(track_lengths, track_indices, track_coordinates, frame_idx):
    """
    Compress track data for a specific frame using the efficient format.
    Returns (tracks, coords) where tracks is list of track IDs and coords is list of 3D points.
    """
    # Build track assignment array
    track_assignment = np.repeat(np.arange(len(track_lengths)), track_lengths)
    
    # Find which tracks are visible in this frame
    mask = track_indices == frame_idx
    tracks = track_assignment[mask].tolist()
    coords = track_coordinates[mask].tolist()
    
    return tracks, coords

def process_triplet(triplet_info, cfg, out_dir, rank, sink):
    """
    Process a single triplet and write to the worker's shard.
    
    Args:
        triplet_info: tuple of (triplet_idx, seq_path, left_frame, mid_frame, right_frame)
        cfg: Hydra config
        out_dir: Output directory
        rank: Worker rank
    """
    triplet_idx, seq_path, left_frame, mid_frame, right_frame = triplet_info
    
    # Construct file paths
    mp4_path = os.path.join(cfg.dataset.stereo4d.lefteye_dir, SPLIT, f"{seq_path}-left_rectified.mp4")
    npz_path = os.path.join(cfg.dataset.stereo4d.path, SPLIT, f"{seq_path}.npz")
    
    if not (os.path.isfile(mp4_path) and os.path.isfile(npz_path)):
        print(f"Missing files for {seq_path}")
        return False
    
    try:
        # Load video
        vr = VideoReader(mp4_path, ctx=cpu(0))
        
        # Load annotations
        ann_data = np.load(npz_path, allow_pickle=True)
        
        # Extract annotation arrays
        track_lengths = ann_data['track_lengths']
        track_indices = ann_data['track_indices']
        track_coordinates = ann_data['track_coordinates']
        camera2world = ann_data['camera2world']
        timestamps = ann_data.get('timestamps', np.arange(len(vr)))
        
        # Load frames
        left_img = vr[left_frame].asnumpy()
        mid_img = vr[mid_frame].asnumpy() 
        right_img = vr[right_frame].asnumpy()
        
        # Convert to RGB
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        mid_img = cv2.cvtColor(mid_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        
        # Compress track data for each frame
        left_tracks, left_coords = compress_track_data(
            track_lengths, track_indices, track_coordinates, left_frame
        )
        mid_tracks, mid_coords = compress_track_data(
            track_lengths, track_indices, track_coordinates, mid_frame
        )
        right_tracks, right_coords = compress_track_data(
            track_lengths, track_indices, track_coordinates, right_frame
        )
        
        # Camera intrinsics (same for all frames)
        width = vr[0].shape[1]
        hfov = cfg.dataset.stereo4d.hfov
        fx = width * 0.5 / np.tan(np.radians(hfov) * 0.5)
        K = np.array([[fx, 0, width/2], [0, fx, width/2], [0, 0, 1]], np.float32)
        
        # Camera extrinsics for each frame
        left_cam2world = camera2world[left_frame]
        mid_cam2world = camera2world[mid_frame]
        right_cam2world = camera2world[right_frame]
        
        # Create sample dict
        sample = {
            "__key__": f"triplet_{triplet_idx:08d}",
            
            # images
            f"left_img.{EXT}":  to_bytes(left_img),
            f"mid_img.{EXT}":   to_bytes(mid_img),
            f"right_img.{EXT}": to_bytes(right_img),
            
            # Compressed track data as JSON (small enough to store efficiently)
            "left_tracks.json": json.dumps({"tracks": left_tracks, "coords": left_coords}),
            "mid_tracks.json": json.dumps({"tracks": mid_tracks, "coords": mid_coords}),
            "right_tracks.json": json.dumps({"tracks": right_tracks, "coords": right_coords}),
            
            # camera
            "intrinsics." + EXT:          to_bytes(K),
            "left_cam2world." + EXT:      to_bytes(left_cam2world.astype(np.float32)),
            "mid_cam2world." + EXT:       to_bytes(mid_cam2world.astype(np.float32)),
            "right_cam2world." + EXT:     to_bytes(right_cam2world.astype(np.float32)),
            
            # Metadata
            "metadata.json": json.dumps({
                "sequence": seq_path,
                "frames": [left_frame, mid_frame, right_frame],
                "instance_left": f"{seq_path}_{left_frame:05d}",
                "instance_mid": f"{seq_path}_{mid_frame:05d}",
                "instance_right": f"{seq_path}_{right_frame:05d}",
            })
        }
        
        # Write to shard
        sink.write(sample)
        
        # Clean up
        del vr
        
        return True
        
    except Exception as e:
        print(f"Error processing triplet {triplet_idx}: {e}")
        return False

def compute_all_triplets(sequence_paths, frame_counts, config,
                         is_valid=False, n_samples=None):
    """Compute (left, mid, right) triplets.

    If n_samples is given it overrides cfg-based lengths.
    """
    np.random.seed(config.seed)

    triplets = []
    target_triplets = (
        n_samples
        if n_samples is not None
        else (config.data.valid_len if is_valid else config.data.len)
    )
    max_frame_window = config.dataset.stereo4d.max_frame_window
    
    while len(triplets) < target_triplets:
        seq_idx = np.random.randint(0, len(sequence_paths))
        frame_count = frame_counts[seq_idx]
        
        if frame_count < 3:
            continue
            
        max_left = frame_count - 3
        if max_left < 0:
            continue
            
        left_frame = np.random.randint(0, max_left + 1)
        min_right = left_frame + 2
        max_right = min(left_frame + max_frame_window, frame_count - 1)
        right_frame = np.random.randint(min_right, max_right + 1)
        mid_frame = np.random.randint(left_frame + 1, right_frame)
        
        triplets.append((seq_idx, left_frame, mid_frame, right_frame))
    
    if len(triplets) > target_triplets:
        triplets = triplets[:target_triplets]
    
    return triplets

from pathlib import Path
import os
import pandas as pd
from joblib import Parallel, delayed
import joblib
from tqdm import tqdm
from contextlib import contextmanager

# ─────────────────────────── helpers ────────────────────────────
@contextmanager
def tqdm_joblib(tqdm_object):
    """link joblib progress to a tqdm bar"""
    class _BatchCB(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(self.batch_size)
            return super().__call__(*args, **kwargs)

    old_cb = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _BatchCB          # patch
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb        # restore
        tqdm_object.close()

# ─────────────────────────── main util ──────────────────────────
def discover_sequences_with_counts(cfg, split, n_jobs=-1, backend="loky"):
    """
    discover sequences + frame counts (stereo4dv5 logic)  
    now: joblib-parallel + tqdm bar + stats filtering
    """
    meta_csv  = os.path.join(cfg.dataset.stereo4d.meta_dir,
                             "stereo4d_id_to_time_and_fov_metadata.csv")
    stats_csv = os.path.join(cfg.dataset.stereo4d.meta_dir, "stats.csv")

    meta_df  = pd.read_csv(
        meta_csv,
        header=0,
        names=["vid", "clipid", "timestamp",
               "start_yaw", "end_yaw", "start_tilt", "end_tilt"],
    )

    stats_df = (
        pd.read_csv(stats_csv, skipinitialspace=True)
          .query("displacement_percentage_50 > 0.10 and d_frame > 5*16")
          .reset_index(drop=True)
    )

    # ---------------------- worker ------------------------------
    def _check_row(row):
        """return (seq, frame_cnt) if assets exist else None"""
        ytid, clipid = row["ytid"], row["clipid"]

        ts_rows = meta_df[(meta_df.vid == ytid) & (meta_df.clipid == clipid)]
        if ts_rows.empty:
            return None

        timestamp = int(ts_rows.iloc[0]["timestamp"])
        seq       = f"{ytid}_{timestamp}"

        mp4_path = Path(cfg.dataset.stereo4d.lefteye_dir) / split / f"{seq}-left_rectified.mp4"
        npz_path = Path(cfg.dataset.stereo4d.path)       / split / f"{seq}.npz"

        if mp4_path.exists() and npz_path.exists():
            return seq, int(row["d_frame"])
        return None

    # ---------------- parallel loop with bar --------------------
    with tqdm_joblib(tqdm(total=len(stats_df), desc="scanning sequences")):
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_check_row)(row) for _, row in stats_df.iterrows()
        )

    # ---------------- aggregate --------------------------------
    available_seqs     = []
    seq_to_frame_count = {}
    for res in results:
        if res is not None:
            seq, frame_cnt = res
            available_seqs.append(seq)
            seq_to_frame_count[seq] = frame_cnt

    return available_seqs, seq_to_frame_count


def worker_process_triplets(worker_triplets, cfg, out_dir, rank):
    tmpl = os.path.join(out_dir, f"stereo4d-triplets-{rank:02d}-%06d.tar")
    with wds.ShardWriter(tmpl,
                         maxsize=MAX_SHARD_SIZE_GB*1e9,
                         maxcount=MAX_SAMPLES_PER_SHARD,
                         verbose=False) as sink:
        successes = 0
        for t in tqdm(worker_triplets,
                      desc=f"worker {rank}", position=rank):
            if process_triplet(t, cfg, out_dir, rank, sink=sink):
                successes += 1
    return successes


from glob import glob
import os, io, hashlib
import webdataset as wds
import numpy as np
import cv2
from decord import VideoReader, cpu
from pathlib import Path

def _load_arr(b: bytes) -> np.ndarray:
    if USE_COMPRESSION:
        with np.load(io.BytesIO(b)) as z:
            return z["arr"]
    return np.load(io.BytesIO(b))

def _md5(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes()).hexdigest()

def _verify_dataset(out_dir: Path, triplet_infos, cfg):
    """Cross-check written shards against raw video/labels."""
    shard_paths = sorted(glob(os.path.join(out_dir, "stereo4d-triplets-*.tar")))
    if not shard_paths:
        print(f"❌  no shards found in {out_dir}; skip verification")
        return

    ds = (
        wds.WebDataset(shard_paths, shardshuffle=False)      # list, **no wildcard**
          .to_tuple("__key__", "left_img.npy", "mid_img.npy", "right_img.npy")
    )

    for (key, lbytes, mbytes, rbytes), (_, seq, lf, mf, rf) in zip(ds, triplet_infos):
        mp4 = Path(cfg.dataset.stereo4d.lefteye_dir) / SPLIT / f"{seq}-left_rectified.mp4"
        vr  = VideoReader(str(mp4), ctx=cpu(0))
        raw  = [cv2.cvtColor(vr[i].asnumpy(), cv2.COLOR_BGR2RGB) for i in (lf, mf, rf)]
        saved = [_load_arr(b) for b in (lbytes, mbytes, rbytes)]
        if any(_md5(r) != _md5(s) for r, s in zip(raw, saved)):
            print(f"❌  mismatch in sample {key}")
            return
    print("✅  verification passed")



@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print(f"Preprocessing Stereo4D triplets")
    print(f"SPLIT={SPLIT}")
    print(f"NUM_WORKERS={NUM_WORKERS}")
    print(f"MAX_SHARD_SIZE_GB={MAX_SHARD_SIZE_GB}")
    print(f"MAX_SAMPLES_PER_SHARD={MAX_SAMPLES_PER_SHARD}")
    
    # Patch config to compute correct number of samples
    with open_dict(cfg):
        cfg.data.len = cfg.train.iterations * cfg.data.batch_size
        cfg.data.valid_len = cfg.data.valid_len * cfg.data.batch_size
    
    # Setup output directory
    out_dir = Path(cfg.dataset.stereo4d.path) / "wds-triplets" / SPLIT
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover sequences and frame counts
    print("Discovering sequences...")
    is_valid = (SPLIT != "train")
    sequence_paths, seq_to_frame_count = discover_sequences_with_counts(cfg, SPLIT)
    
    # Filter to sequences with frame counts
    sequence_paths = [s for s in sequence_paths if s in seq_to_frame_count]
    frame_counts = [seq_to_frame_count[s] for s in sequence_paths]
    
    print(f"Found {len(sequence_paths)} sequences")
    
    # Sample sequences if needed (following Stereo4Dv5 logic)
    limit = min(len(sequence_paths), cfg.data.valid_len if is_valid else cfg.data.len)
    np.random.seed(cfg.seed)
    selected_indices = np.random.choice(len(sequence_paths), size=limit, replace=False)
    
    selected_sequences = [sequence_paths[i] for i in selected_indices]
    selected_frame_counts = [frame_counts[i] for i in selected_indices]
    
    # Compute all triplets
    print("Computing triplets…")
    triplets = compute_all_triplets(
        selected_sequences,
        selected_frame_counts,
        cfg,
        is_valid,
        n_samples=N_SAMPLES                # ← NEW
    )
    
    # Convert triplets to full info (idx, seq_path, left, mid, right)
    triplet_infos = []
    for idx, (seq_idx, left, mid, right) in enumerate(triplets):
        seq_path = selected_sequences[seq_idx]
        triplet_infos.append((idx, seq_path, left, mid, right))
    
    print(f"Generated {len(triplet_infos)} triplets")
    
    # Distribute triplets across workers
    triplets_per_worker = len(triplet_infos) // NUM_WORKERS
    worker_assignments = []
    
    for i in range(NUM_WORKERS):
        start_idx = i * triplets_per_worker
        if i == NUM_WORKERS - 1:
            # Last worker gets any remainder
            end_idx = len(triplet_infos)
        else:
            end_idx = (i + 1) * triplets_per_worker
        
        worker_triplets = triplet_infos[start_idx:end_idx]
        worker_assignments.append((worker_triplets, cfg, str(out_dir), i))
    
    # Process in parallel
    print(f"Processing triplets with {NUM_WORKERS} workers...")
    from multiprocessing import Pool
    
    with Pool(NUM_WORKERS) as pool:
        results = pool.starmap(worker_process_triplets, worker_assignments)
    
    total_processed = sum(results)
    print(f"Successfully processed {total_processed}/{len(triplet_infos)} triplets")
    
    # Close all writers
    _close_sinks()


    # ------------------------------------------------------------------
    # OPTIONAL verification (only when we purposely limited N_SAMPLES)
    # ------------------------------------------------------------------
    if N_SAMPLES is not None:
        print("Verifying WebDataset contents…")
        _verify_dataset(out_dir, triplet_infos, cfg)
    
    # Create index
    print("Creating index...")
    idx_json = out_dir / "stereo4d-triplets-idx.json"
    shard_glob = str(out_dir / "stereo4d-triplets-*.tar")
    os.system(f"widsindex create {shard_glob} -o {idx_json}")
    print(f"Index created at {idx_json}")
    
    print("Done!")

if __name__ == "__main__":
    main()