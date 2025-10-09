# stereo4dv6.py

from typing import NoReturn  # unused sentinel to avoid empty import block
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from decord import VideoReader, cpu
import albumentations as A
import torch

from .stereo_motion_base import StereoMotionBase
import utils.geometry as geom


def _intrinsic_K(width: int, hfov_deg: float) -> np.ndarray:
    # compute pinhole intrinsics from hfov and image width
    fx = width * 0.5 / np.tan(np.deg2rad(hfov_deg) * 0.5)
    return np.array([[fx, 0, width / 2],
                     [0, fx, width / 2],
                     [0,  0,          1]], dtype=np.float32)  # (3, 3)


def _optimized_pcs(lengths: np.ndarray,
                   track_indices: np.ndarray,
                   track_coordinates: np.ndarray,
                   idxs: np.ndarray) -> np.ndarray:
    # vectorized scatter into (tracks, len(idxs), 3), with valid mask appended
    frame_idxs = np.asarray(idxs)
    col_idx_full = track_indices
    keep = np.isin(col_idx_full, frame_idxs)

    if keep.any():
        obs_idx = np.flatnonzero(keep)
        row_s = np.searchsorted(lengths.cumsum(), obs_idx, side='right')

        frames_kept = col_idx_full[keep]
        # ensure mapping array can index both tracked and requested frames
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

    tracks = np.full((len(lengths), len(frame_idxs), 3), np.nan, np.float32)  # (T, F, 3)
    if len(row_s):
        tracks[row_s, col_s] = coord_s
    valid = (~np.isnan(tracks[..., 0]))[..., None].astype(np.float32)  # (T, F, 1)
    return np.concatenate([tracks, valid], axis=-1)  # (T, F, 4)


class Stereo4D(StereoMotionBase):
    """
    ultra-lightweight Stereo-4D dataloader (direct-from-disk).
    reads sequence-level mp4/npz files without webdataset.

    design:
    - single-frame path opens & seeks once, no persistent caches
    - multi-frame path opens once per sequence, batches seeks
    - intrinsics computed from hfov + width (memoized per seq)
    """

    def __init__(self, config, valid: bool = False):
        super().__init__(config, valid)
        print("loading stereo4d dataset (direct-from-disk)...")

        self.dataset_label = config.dataset.stereo4d.name
        self.split = config.dataset.stereo4d.valid_split if valid else config.dataset.stereo4d.train_split
        self.hfov = config.dataset.stereo4d.hfov
        self.max_frame_window = config.dataset.stereo4d.max_frame_window
        
        self.config = config

        # color augmentation (train) or identity (valid)
        if not valid and not getattr(config, "debug", False):
            aug_list = [
                A.RandomBrightnessContrast(p=0.2),
                A.HueSaturationValue(p=0.2),
                A.ToGray(p=0.2),
                A.ImageCompression(quality_range=(30, 100), p=0.5),
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

        # dirs
        self.left_mp4_dir = Path(config.dataset.stereo4d.lefteye_dir) / self.split
        self.npz_dir = Path(config.dataset.stereo4d.path) / self.split

        # lightweight per-seq memo for intrinsics width
        self._intrinsics_by_seq: Dict[str, np.ndarray] = {}

        # build available sequences + frame counts from precomputed CSV (from config)
        sequences_csv = Path(config.dataset.stereo4d.sequences_csv)
        df = pd.read_csv(sequences_csv)

        # keep only seqs that have both mp4 + npz on disk
        def _exists(seq: str) -> bool:
            mp4 = self.left_mp4_path(seq)
            npz = self.npz_path(seq)
            return mp4.is_file() and npz.is_file()

        df = df[df["sequence_id"].map(_exists)]

        # frame counts and per-sequence width (if present)
        seq_to_frame_count = dict(zip(df["sequence_id"], df["length"].astype(int)))
        # width column may exist in CSV, but we do not use it for intrinsics

        print(f"found {len(seq_to_frame_count)} sequences on disk (from CSV)")

        # sample sequences
        available = list(seq_to_frame_count.keys())
        limit = min(len(available), config.data.len if not valid else config.data.valid_len)

        np.random.seed(config.seed)
        selected = np.random.choice(available, size=limit, replace=False)

        for seq in selected:
            self.sequence_paths.append(seq)
            self.frame_counts.append(int(seq_to_frame_count[seq]))

        print(f"total sequences: {len(self.sequence_paths)}")
        print(f"total frames   : {sum(self.frame_counts)}")

        # precompute triplets as in base impl
        self._compute_triplets()

    # ----------------------------- path helpers -----------------------------

    def left_mp4_path(self, seq: str) -> Path:
        return self.left_mp4_dir / f"{seq}-left_rectified.mp4"

    def npz_path(self, seq: str) -> Path:
        return self.npz_dir / f"{seq}.npz"

    # ----------------------------- core loaders -----------------------------

    def _load_single_frame(self, seq: str, idx: int) -> np.ndarray:
        vr = VideoReader(str(self.left_mp4_path(seq)), ctx=cpu(0))
        frame = vr[idx].asnumpy()  # (H, W, 3)
        del vr
        return frame  # (H, W, 3)

    def _load_single_frame_annotations(self, seq: str, idx: int) -> np.ndarray:
        ann = np.load(self.npz_path(seq), allow_pickle=True)
        pcs_all = _optimized_pcs(
            lengths=ann["track_lengths"],
            track_indices=ann["track_indices"],
            track_coordinates=ann["track_coordinates"],
            idxs=np.array([idx], dtype=np.int32),
        )  # (T, 1, 4)
        pcs = pcs_all[:, 0, :]  # (T, 4)
        return pcs.astype(np.float32)  # (T, 4)

    def _get_intrinsics(self, seq: str) -> np.ndarray:
        K = self._intrinsics_by_seq.get(seq)
        if K is not None:
            return K  # (3, 3)
        vr = VideoReader(str(self.left_mp4_path(seq)), ctx=cpu(0))
        _, w = vr[0].shape[0], vr[0].shape[1]
        del vr
        K = _intrinsic_K(w, self.hfov)  # (3, 3)
        self._intrinsics_by_seq[seq] = K
        return K  # (3, 3)

    def _load_camera_data(self, seq: str, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        K = self._get_intrinsics(seq)  # (3, 3)
        ann = np.load(self.npz_path(seq), allow_pickle=True)
        extr = geom.inv(ann["camera2world"][idx])  # (4, 4) or (3, 4) depending on your utils
        return K, extr  # (3, 3), (4, 4)

    # ------------------------------ public api ------------------------------

    def get_frame_info(self, seq: str, idx: int) -> dict:
        img = self._load_single_frame(seq, idx)  # (H, W, 3)
        world_pc = self._load_single_frame_annotations(seq, idx)  # (T, 4)
        intr, extr = self._load_camera_data(seq, idx)  # (3, 3), (4, 4)

        return dict(
            image=torch.from_numpy(img),                     # (H, W, 3)
            world_pc_valid=torch.from_numpy(world_pc),       # (T, 4)
            cam=(torch.from_numpy(intr), torch.from_numpy(extr)),  # ((3,3), (4,4))
            dm=None,
            instance=f"{seq}_{idx:05d}",
        )

    def get_frame_infos(self, seq: str, idxs: List[int]) -> List[dict]:
        # open video once
        vr = VideoReader(str(self.left_mp4_path(seq)), ctx=cpu(0))

        # batch load frames
        frames: List[np.ndarray] = []
        for i in idxs:
            frames.append(vr[i].asnumpy())  # (H, W, 3)
        del vr
        frames = [torch.from_numpy(f) for f in frames]

        # load ann once
        ann = np.load(self.npz_path(seq), allow_pickle=True)

        # intrinsics once
        intr = self._get_intrinsics(seq)  # (3, 3)
        intr = torch.from_numpy(intr.astype(np.float32))

        # annotations for selected frames
        pcs_all = _optimized_pcs(
            lengths=ann["track_lengths"],
            track_indices=ann["track_indices"],
            track_coordinates=ann["track_coordinates"],
            idxs=np.asarray(idxs, dtype=np.int32),
        )  # (T, F, 4)

        # extrinsics for selected frames
        c2w = ann["camera2world"]
        extr_list = [geom.inv(c2w[i]) for i in idxs]  # list of (4, 4)
        extr_list = [torch.from_numpy(e.astype(np.float32)) for e in extr_list]

        # col lookup
        col_of_frame = {f: j for j, f in enumerate(idxs)}

        # package
        out: List[dict] = []
        for j, fidx in enumerate(idxs):
            out.append(dict(
                image=frames[j],                                    # (H, W, 3)
                world_pc_valid=torch.from_numpy(
                    pcs_all[:, col_of_frame[fidx], :].astype(np.float32)
                ),                                                  # (T, 4)
                cam=(intr, extr_list[j]),                           # ((3,3), (4,4))
                dm=None,
                instance=f"{seq}_{fidx:05d}",
            ))

        return out
