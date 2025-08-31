"""
Author: Kevin Mathew T
Date: 2025-08-17
LinkedIn: https://www.linkedin.com/in/kevinmathewt/
"""

import json, io, time, tempfile, os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import wids
from decord import VideoReader, cpu, gpu
import albumentations as A

from .stereo_motion_base import StereoMotionBase
# import utils.geometry as geom
import utils.torch_geometry as geom


# --- decord gpu helpers ---
def _get_decord_ctx():
    """Prefer current CUDA device for Decord; fallback to CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            dev_id = torch.cuda.current_device()
            return gpu(dev_id)
    except Exception:
        pass
    return cpu(0)


def _to_torch_cuda(decord_obj):
    """Zero-copy convert a Decord NDArray/Frame to torch (keeps on GPU if decoded there)."""
    import torch
    try:
        return torch.utils.dlpack.from_dlpack(decord_obj.to_dlpack())
    except Exception:
        # Fallback best-effort conversion (may copy)
        print("Warning: Failed to convert Decord NDArray to torch tensor on GPU, falling back to CPU. Copying data instead...")
        return torch.as_tensor(decord_obj)


class Stereo4DWDS(StereoMotionBase):
    """
    ultra-lightweight Stereo-4D dataloader optimized for 8TB+ datasets.

    no caching – optimized for single-access patterns where each frame
    is accessed exactly once. minimizes memory usage and maximizes throughput.

    key optimizations in this version:
    - single-frame annotations use the vectorized scatter (no full (T,F,3) tensor)
    - get_frame_info loads ann/intr once
    - multi-frame decode uses decord.get_batch
    - consistent bgr→rgb via channel reverse
    - fewer large temporaries in pcs construction
    """

    # spill very large mp4 payloads to a temp file to cap peak ram
    _spill_threshold_bytes = 128 << 20  # 128mb

    def __init__(self, config, valid: bool = False):
        super().__init__(config, valid)
        print("loading stereo4d v5 dataset...")

        # ensure WIDS cache uses config-provided cache path if available
        try:
            cache_dir = getattr(config.dataset.stereo4d, "cache", None)
            if cache_dir:
                os.environ.setdefault("WIDS_CACHE", str(cache_dir))
        except Exception:
            pass

        self.dataset_label = config.dataset.stereo4d.name
        split = (
            config.dataset.stereo4d.valid_split
            if valid
            else config.dataset.stereo4d.train_split
        )
        self.hfov = config.dataset.stereo4d.hfov
        self.max_frame_window = config.dataset.stereo4d.max_frame_window
        self.config = config

        # color augmentation (train) or identity (valid)
        if not valid:
            aug_list = [
                A.RandomBrightnessContrast(p=0.2),
                A.HueSaturationValue(p=0.2),
                A.ToGray(p=0.2),
                A.ImageCompression(quality_lower=30, quality_upper=100, p=0.5),
                A.OneOf(
                    [
                        A.MotionBlur(p=0.2),
                        A.MedianBlur(blur_limit=3, p=0.1),
                        A.Blur(blur_limit=3, p=0.1),
                    ],
                    p=0.2,
                ),
            ]
            self.color_aug = A.Compose(aug_list)
        else:
            self.color_aug = A.NoOp()

        # webdataset setup
        wds_dir = Path(config.dataset.stereo4d.wds_dir) / split
        idx_json = wds_dir / "stereo4d-idx.json"
        map_json = wds_dir / "key_to_idx.json"

        print(f"loading wds index from {idx_json}...")
        self.idx_ds = wids.ShardListDataset(str(idx_json), transformations=[])
        print(f"loaded {len(self.idx_ds)} sequence samples")

        with open(map_json) as f:
            self.key_to_idx = json.load(f)

        # load metadata
        available_seqs = list(self.key_to_idx.keys())
        print(f"found {len(available_seqs)} sequences in WebDataset")

        stats_csv = Path(config.dataset.stereo4d.meta_dir) / "stats.csv"
        stats_df = pd.read_csv(stats_csv, skipinitialspace=True)

        meta_csv = Path(config.dataset.stereo4d.meta_dir) / "stereo4d_id_to_time_and_fov_metadata.csv"
        meta_df = pd.read_csv(
            meta_csv,
            header=0,
            names=["vid", "clipid", "timestamp", "start_yaw", "end_yaw", "start_tilt", "end_tilt"],
        )

        # process metadata
        available_seqs_set = set(available_seqs)
        merged = stats_df.merge(
            meta_df[["vid", "clipid", "timestamp"]],
            left_on=["ytid", "clipid"],
            right_on=["vid", "clipid"],
            how="inner",
        )
        merged = merged.groupby(["ytid", "clipid"]).first().reset_index()
        merged["seq"] = merged["ytid"] + "_" + merged["timestamp"].astype(int).astype(str)
        filtered = merged[merged["seq"].isin(available_seqs_set)]
        seq_to_frame_count = dict(zip(filtered["seq"], filtered["d_frame"].astype(int)))

        print(f"mapped {len(seq_to_frame_count)} sequences to frame counts")

        # sample sequences
        available_with_counts = list(seq_to_frame_count.keys())
        limit = min(len(available_with_counts), config.data.len if not valid else config.data.valid_len)

        np.random.seed(config.seed)
        selected_seqs = np.random.choice(available_with_counts, size=limit, replace=False)

        # build sequence list
        for seq in selected_seqs:
            self.sequence_paths.append(seq)
            self.frame_counts.append(seq_to_frame_count[seq])

        print(f"total sequences: {len(self.sequence_paths)}")
        print(f"total frames   : {sum(self.frame_counts)}")

        # precompute frame triplets
        self._compute_triplets()

    # ----------------------------- helpers ---------------------------------

    def _video_bytes_from_sample(self, sample) -> bytes:
        """read raw mp4 bytes out of the tar entry."""  # (bytes)
        video_obj = sample[".video.mp4"]
        return video_obj.read() if hasattr(video_obj, "read") else video_obj  # (B,)

    def _open_vr_from_bytes(self, vbytes: bytes) -> VideoReader:
        """open a decord reader, spilling huge payloads to a tmpfile to bound peak ram."""
        ctx = _get_decord_ctx()
        if len(vbytes) > self._spill_threshold_bytes:
            tf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=True)
            tf.write(vbytes)
            tf.flush()
            # keep a reference so file doesn't disappear while vr is alive
            try:
                vr = VideoReader(tf.name, ctx=ctx)
            except Exception:
                print("Failed to use GPU for Decord, falling back to CPU...")
                vr = VideoReader(tf.name, ctx=cpu(0))
            vr._tmpfile = tf  # attach for lifecycle
            return vr
        try:
            return VideoReader(io.BytesIO(vbytes), ctx=ctx)
        except Exception:
            print("Failed to use GPU for Decord, falling back to CPU...")
            return VideoReader(io.BytesIO(vbytes), ctx=cpu(0))

    # ----------------------------- core loaders -----------------------------

    def _load_single_frame(self, seq: str, idx: int, sample=None, vbytes: bytes | None = None):
        """load only the specific frame needed."""  # (H, W, 3)
        sample = sample if sample is not None else self.idx_ds[self.key_to_idx[seq]]
        vbytes = vbytes if vbytes is not None else self._video_bytes_from_sample(sample)
        vr = self._open_vr_from_bytes(vbytes)
        frame = _to_torch_cuda(vr[idx])              # (H, W, 3) torch tensor (GPU if available)
        del vr
        return frame

    def _load_single_frame_annotations(self, seq: str, idx: int, ann_data=None):
        """vectorized scatter for just one frame."""  # (T, 4)
        sample = self.idx_ds[self.key_to_idx[seq]] if ann_data is None else None
        ann = ann_data if ann_data is not None else np.load(sample[".ann.npz"], allow_pickle=True)
        pcs1 = optimized_pcs(
            ann["track_lengths"],
            ann["track_indices"],
            ann["track_coordinates"],
            np.array([idx], dtype=np.int32),
        )[:, 0, :].astype(np.float32)  # (T, 4)
        return pcs1

    def _load_camera_data(self, seq: str, idx: int, ann_data=None, intrinsics=None):
        """load intrinsics (once per seq) and per-frame extrinsics."""  # ((3,3), (4,4))
        if ann_data is None or intrinsics is None:
            sample = self.idx_ds[self.key_to_idx[seq]]
            if intrinsics is None:
                intrinsics = np.load(sample[".intr.npy"], allow_pickle=True)  # (3, 3)
            if ann_data is None:
                ann_data = np.load(sample[".ann.npz"], allow_pickle=True)

        extrinsics = geom.inv(ann_data["camera2world"][idx])  # (4, 4)
        return intrinsics, extrinsics

    # ------------------------------ public api ------------------------------

    def get_frame_info(self, seq: str, idx: int):
        """
        loads only the exact data needed with minimal memory usage.  # (image: (H,W,3), world_pc_valid: (T,4))
        """
        sample = self.idx_ds[self.key_to_idx[seq]]

        # video
        vbytes = self._video_bytes_from_sample(sample)
        img = self._load_single_frame(seq, idx, sample=sample, vbytes=vbytes)  # (H, W, 3) torch tensor

        # ann + intr once
        ann = np.load(sample[".ann.npz"], allow_pickle=True)
        # device follows decoded image
        import torch
        device = img.device
        lengths = torch.from_numpy(ann["track_lengths"].astype(np.int32)).to(device)
        track_indices = torch.from_numpy(ann["track_indices"].astype(np.int32)).to(device)
        track_coordinates = torch.from_numpy(ann["track_coordinates"].astype(np.float32)).to(device)
        world_pc_all = optimized_pcs_torch(lengths, track_indices, track_coordinates, torch.tensor([idx], dtype=torch.int32, device=device))  # (T, 1, 4)
        world_pc = world_pc_all[:, 0, :]  # (T, 4)
        intrinsics = torch.from_numpy(np.load(sample[".intr.npy"], allow_pickle=True).astype(np.float32)).to(device)
        cam2world = torch.from_numpy(ann["camera2world"][idx].astype(np.float32)).to(device)
        extrinsics = geom.inv(cam2world)  # (4, 4)

        return dict(
            image=img,                              # (H, W, 3) torch tensor (GPU if available)
            world_pc_valid=world_pc,                # (T, 4) torch
            cam=(intrinsics, extrinsics),           # ((3,3), (4,4)) torch
            dm=None,
            instance=f"{seq}_{idx:05d}",
        )

    def get_frame_infos(self, seq: str, idxs: list[int]):
        """
        optimized batch loading of multiple frames from the same sequence.  # list of dicts
        """
        # pull sample once
        sample = self.idx_ds[self.key_to_idx[seq]]

        # video bytes -> single VR
        vbytes = self._video_bytes_from_sample(sample)
        vr = self._open_vr_from_bytes(vbytes)

        # batch decode with get_batch
        frames_nd = _to_torch_cuda(vr.get_batch(idxs))      # (F, H, W, 3) torch tensor (GPU if available)
        del vr

        # load ann + intr once
        ann = np.load(sample[".ann.npz"], allow_pickle=True)
        import torch
        device = frames_nd.device
        lengths = torch.from_numpy(ann["track_lengths"].astype(np.int32)).to(device)
        track_indices = torch.from_numpy(ann["track_indices"].astype(np.int32)).to(device)
        track_coordinates = torch.from_numpy(ann["track_coordinates"].astype(np.float32)).to(device)
        intrinsics = torch.from_numpy(np.load(sample[".intr.npy"], allow_pickle=True).astype(np.float32)).to(device)

        # pcs for selected frames (torch)
        frame_idxs = torch.tensor(idxs, dtype=torch.int32, device=device)
        pcs_all = optimized_pcs_torch(lengths, track_indices, track_coordinates, frame_idxs)  # (T, F, 4)

        # extrinsics for selected frames
        extrinsics_list = []
        for i in idxs:
            cam2world_i = torch.from_numpy(ann["camera2world"][i].astype(np.float32)).to(device)
            extrinsics_list.append(geom.inv(cam2world_i))

        # package
        results = []
        for j, fidx in enumerate(idxs):
            results.append(dict(
                image=frames_nd[j],                  # (H, W, 3) torch tensor (GPU if available)
                world_pc_valid=pcs_all[:, j, :],     # (T, 4) torch
                cam=(intrinsics, extrinsics_list[j]),
                dm=None,
                instance=f"{seq}_{fidx:05d}",
            ))

        return results


import hydra
from omegaconf import DictConfig, open_dict
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data._utils.collate import default_collate

# --- user-controlled setting ------------------------------------------------
TARGET_SAMPLES = 10  # override: NUM_SAMPLES=25 poetry run python -m loaders.stereo4dv5
# ---------------------------------------------------------------------------

def add_batch_size_wrapper(batch):
    batch = default_collate(batch)
    batch["batch_size"] = len(batch["left_pm"])
    return batch

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig):
    # replicate training-time patches so dataset builds
    with open_dict(config):
        config.data.len = config.train.iterations * config.data.batch_size
        config.data.valid_len = config.data.valid_len * config.data.batch_size

    dataset = Stereo4DWDS(config, valid=False)
    dist = torch.distributed.is_initialized()
    world = torch.distributed.get_world_size() if dist else 1
    sampler = DistributedSampler(dataset) if dist else None

    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size // world,
        shuffle=(sampler is None),
        num_workers=config.data.num_workers,
        sampler=sampler,
        collate_fn=add_batch_size_wrapper,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # iterate and time
    seen = 0
    pbar = tqdm(total=TARGET_SAMPLES, desc="loading samples")
    t0 = time.perf_counter()
    for batch in loader:
        bs = batch["batch_size"]
        seen += bs
        pbar.update(bs if seen <= TARGET_SAMPLES else bs - (seen - TARGET_SAMPLES))
        if seen >= TARGET_SAMPLES:
            break
    pbar.close()
    print(f"pulled {min(seen, TARGET_SAMPLES)} samples in {time.perf_counter() - t0:.2f}s")

if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# implementations
# ---------------------------------------------------------------------------

def naive_pcs(lengths, track_indices, track_coordinates, idxs):
    num_f = int(track_indices.max()) + 1
    n_tracks = int(len(lengths))
    row_idx = np.repeat(np.arange(n_tracks), lengths)
    tracks = np.full((n_tracks, num_f, 3), np.nan, np.float32)
    tracks[row_idx, track_indices] = track_coordinates
    pcs = np.empty((n_tracks, num_f, 4), np.float32)
    pcs[..., :3] = tracks
    pcs[..., 3] = (~np.isnan(tracks[..., 0])).astype(np.float32)
    sel = np.stack([pcs[:, idx, :] for idx in idxs], axis=1)  # (T, Fsel, 4)
    return sel


def optimized_pcs(lengths, track_indices, track_coordinates, idxs):
    """
    vectorized scatter into (tracks, len(idxs), 3) with a valid mask appended.  # (T, Fsel, 4)
    chooses between dense and dict mapping for frame->column.
    """
    frame_idxs = np.asarray(idxs, dtype=np.int32)            # (Fsel,)
    col_idx_full = track_indices                              # (Nobs,)
    keep = np.isin(col_idx_full, frame_idxs)                  # (Nobs,)

    if keep.any():
        obs_idx = np.flatnonzero(keep)                        # (K,)
        row_s = np.searchsorted(lengths.cumsum(), obs_idx, side="right")  # (K,)

        frames_kept = col_idx_full[keep]                      # (K,)
        # heuristic: dense map only if range is not much larger than selection
        max_frame = int(col_idx_full.max())
        if (max_frame + 1) <= 8 * len(frame_idxs):
            frame2col = np.full(max_frame + 1, -1, np.int32)
            frame2col[frame_idxs] = np.arange(len(frame_idxs), dtype=np.int32)
            col_s = frame2col[frames_kept]                    # (K,)
        else:
            small_map = {f: j for j, f in enumerate(frame_idxs)}
            col_s = np.fromiter((small_map[f] for f in frames_kept), dtype=np.int32, count=len(frames_kept))

        coord_s = track_coordinates[keep].astype(np.float32)  # (K, 3)
    else:
        row_s = np.empty((0,), np.int32)
        col_s = np.empty((0,), np.int32)
        coord_s = np.empty((0, 3), np.float32)

    T = int(len(lengths))
    F = int(len(frame_idxs))
    tracks = np.full((T, F, 3), np.nan, np.float32)          # (T, F, 3)
    if len(row_s):
        tracks[row_s, col_s] = coord_s

    pcs = np.empty((T, F, 4), np.float32)                    # (T, F, 4)
    pcs[..., :3] = tracks
    pcs[..., 3] = (~np.isnan(tracks[..., 0])).astype(np.float32)
    return pcs



def optimized_pcs_torch(lengths: 'torch.Tensor', track_indices: 'torch.Tensor', track_coordinates: 'torch.Tensor', idxs: 'torch.Tensor'):
    """
    Torch equivalent of optimized_pcs. All inputs are torch tensors on same device.
    lengths: (T,), int32; track_indices: (Nobs,), int32; track_coordinates: (Nobs,3), float32; idxs: (Fsel,), int32
    Returns: (T, Fsel, 4) float32
    """
    import torch
    device = track_indices.device
    frame_idxs = idxs.to(torch.int32)  # (Fsel,)
    col_idx_full = track_indices  # (Nobs,)
    keep = torch.isin(col_idx_full, frame_idxs)  # (Nobs,)

    if torch.any(keep):
        obs_idx = torch.nonzero(keep, as_tuple=True)[0]  # (K,)
        cum = lengths.cumsum(0)
        row_s = torch.searchsorted(cum, obs_idx, right=True)  # (K,)

        frames_kept = col_idx_full[keep]  # (K,)
        max_frame = int(col_idx_full.max().item()) if col_idx_full.numel() > 0 else -1
        Fsel = int(len(frame_idxs))
        if (max_frame + 1) <= 8 * max(Fsel, 1):
            frame2col = torch.full((max_frame + 1,), -1, dtype=torch.int32, device=device)
            frame2col[frame_idxs] = torch.arange(Fsel, dtype=torch.int32, device=device)
            col_s = frame2col[frames_kept]
        else:
            # fallback mapping via search in a dict-like manner
            # create (frame_idx -> col) tensor and gather via matching
            # This branch rarely triggers in practice for selected frames.
            col_s = torch.empty_like(frames_kept)
            for j, f in enumerate(frame_idxs):
                col_s[frames_kept == f] = int(j)

        coord_s = track_coordinates[keep].to(torch.float32)
    else:
        row_s = torch.empty((0,), dtype=torch.int64, device=device)
        col_s = torch.empty((0,), dtype=torch.int64, device=device)
        coord_s = torch.empty((0, 3), dtype=torch.float32, device=device)

    T = int(lengths.numel())
    F = int(frame_idxs.numel())
    tracks = torch.full((T, F, 3), float('nan'), dtype=torch.float32, device=device)
    if row_s.numel() > 0:
        tracks[row_s.long(), col_s.long()] = coord_s

    pcs = torch.empty((T, F, 4), dtype=torch.float32, device=device)
    pcs[..., :3] = tracks
    pcs[..., 3] = (~torch.isnan(tracks[..., 0])).to(torch.float32)
    return pcs
