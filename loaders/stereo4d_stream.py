# loaders/stereo4d_stream.py
from __future__ import annotations
import io
import re
from pathlib import Path
import glob
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info
import webdataset as wds
import albumentations as A

# torch geometry (your GPU-friendly port)
import utils.torch_geometry as geo


# ----------------------------
# Small helpers (readability)
# ----------------------------

_KEY_RE = re.compile(r"^(?P<seq>.+)_(?P<l>\d{5})_(?P<m>\d{5})_(?P<r>\d{5})$")


def _np_from_npy_bytes(b) -> np.ndarray:
    """Load a true .npy byte buffer into a numpy array."""
    return np.load(io.BytesIO(bytes(b)), allow_pickle=False)


def _parse_sample_key(key: str) -> Tuple[str, int, int, int]:
    """Parse '__key__' -> (sequence, left_idx, mid_idx, right_idx)."""
    m = _KEY_RE.match(key)
    if not m:
        raise ValueError(f"Bad key format: {key}")
    return m.group("seq"), int(m.group("l")), int(m.group("m")), int(m.group("r"))


def _to_chw_float(img_hwc_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    """HWC uint8 [0..255] -> CHW float32 [0..1] on device."""
    t = torch.from_numpy(img_hwc_uint8).permute(2, 0, 1).contiguous()
    return (t.to(device=device, dtype=torch.float32) / 255.0)


def _stack_cam_tuples(samples: List[Dict[str, Any]]) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
                                                              Tuple[torch.Tensor, torch.Tensor],
                                                              Tuple[torch.Tensor, torch.Tensor]]:
    """Stack (K, E) camera tuples across the batch -> ((B,3,3),(B,4,4)) for left/mid/right."""
    K_left  = torch.stack([s["cam"][0]       for s in samples], dim=0)
    E_left  = torch.stack([s["cam"][1]       for s in samples], dim=0)
    K_mid   = torch.stack([s["cam_mid"][0]   for s in samples], dim=0)
    E_mid   = torch.stack([s["cam_mid"][1]   for s in samples], dim=0)
    K_right = torch.stack([s["cam_right"][0] for s in samples], dim=0)
    E_right = torch.stack([s["cam_right"][1] for s in samples], dim=0)
    return (K_left, E_left), (K_mid, E_mid), (K_right, E_right)


# ---------------------------------
# Main streaming WebDataset loader
# ---------------------------------

class Stereo4DWDSStream(IterableDataset):
    """
    Streaming Stereo-4D loader over WebDataset shards (image_format='npy').

    Geometry placement:
      - if config.do_geo_calc_on_gpu == False → compute in _decode_wds_sample on CPU (parallel across workers)
      - else → compute in collate on GPU

    Albumentations color aug (left/right) only in collate to keep behavior identical across modes.
    """

    # ---------- init ----------
    def __init__(self, config, valid: bool = False):
        super().__init__()
        self.config: Any = config
        self.is_valid_split: bool = bool(valid)
        self._geo_on_gpu: bool = bool(getattr(config, "do_geo_calc_on_gpu", True))

        # where shards live
        ds = config.dataset.stereo4d
        split = ds.valid_split if self.is_valid_split else ds.train_split
        if getattr(ds, "wds_dir", None) and str(ds.wds_dir) not in ("", "???"):
            base = Path(ds.wds_dir)
            if base.name not in (split, "train", "test", "valid", "validation"):
                base = base / split
        else:
            base = Path(ds.path) / "wds" / split
        self._shard_glob: str = str(base / "stereo4d-*.tar")

        # epoch length
        self._epoch_len: int = int(config.data.valid_len if self.is_valid_split else config.data.len)

        # dataset-scoped knobs
        loader_key = getattr(config.data, "loader", "stereo4d").lower()
        ds_loader_cfg = getattr(config.dataset, loader_key, None)
        self._pm_source: str = getattr(ds_loader_cfg, "pm_source", "3d_tracks") if ds_loader_cfg else "3d_tracks"
        self._min_valid_pc: int = int(getattr(ds_loader_cfg, "min_valid_pc_points", 0) if ds_loader_cfg else 0)
        self._min_valid_mm: int = int(getattr(ds_loader_cfg, "min_valid_mm_points", 0) if ds_loader_cfg else 0)

        # Albumentations (train only)
        if (not self.is_valid_split) and (not getattr(config, "debug", False)):
            self._albu = A.Compose([
                A.RandomBrightnessContrast(p=0.2),
                A.HueSaturationValue(p=0.2),
                A.ToGray(p=0.2),
                A.ImageCompression(quality_range=(30, 100), p=0.5),  # webdataset warning fixed
                A.OneOf(
                    [
                        A.MotionBlur(p=0.2),
                        A.MedianBlur(blur_limit=3, p=0.1),
                        A.Blur(blur_limit=3, p=0.1),
                    ],
                    p=0.2,
                ),
            ])
        else:
            self._albu = None

        # stable per-process mapping: sequence -> integer index
        self._seq_to_int: Dict[str, int] = {}

        # expose collate for DataLoader
        self.custom_collate_fn = self._collate_build_batch

        # no debug prints

    # ---------- IterableDataset protocol ----------
    def __len__(self) -> int:
        return self._epoch_len

    def __iter__(self):
        # Expand glob to explicit shard paths; avoid passing wildcard to WebDataset
        shard_paths = sorted(glob.glob(self._shard_glob))
        if len(shard_paths) == 0:
            raise RuntimeError(f"No shards found for pattern: {self._shard_glob}")

        shard_source = (
            wds.ResampledShards(shard_paths)    # infinite stochastic stream over explicit paths
            if not self.is_valid_split
            else wds.SimpleShardList(shard_paths)
        )

        pipeline = wds.DataPipeline(
            shard_source,
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.ignore_and_continue),
            (wds.shuffle(1000) if not self.is_valid_split else (lambda x: x)),
            (lambda src: (self._decode_wds_sample(s) for s in src)),
            (lambda src: (s for s in src if s is not None)),
        )

        for sample in pipeline:
            yield sample

    # ---------- shared geometry helpers (no duplication) ----------
    def _prepare_geom_inputs(
        self, s: Dict[str, Any], device: torch.device
    ) -> Tuple[int, int,
               torch.Tensor, torch.Tensor, torch.Tensor,
               Tuple[torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor],
               str, int, int, int]:
        """Convert numpy fields in `s` to torch tensors on `device` for geometry."""
        seq = s["seq"]
        l_idx, m_idx, r_idx = s["idxs"]
        H, W = s["image_left"].shape[:2]

        # tracks
        wpv_left_t  = torch.from_numpy(s["wpv_left"]).to(device)
        wpv_mid_t   = torch.from_numpy(s["wpv_mid"]).to(device)
        wpv_right_t = torch.from_numpy(s["wpv_right"]).to(device)
        if wpv_left_t.numel() == 0 or wpv_mid_t.numel() == 0 or wpv_right_t.numel() == 0:
            raise AssertionError(f"missing tracks in {seq}:{l_idx}-{m_idx}-{r_idx}")

        # cameras (same intrinsics K for all three)
        K_left,  E_left  = s["cam_left"]
        K_mid,   E_mid   = s["cam_mid"]
        K_right, E_right = s["cam_right"]
        cam_left_t  = (torch.from_numpy(K_left ).to(device), torch.from_numpy(E_left ).to(device))
        cam_mid_t   = (torch.from_numpy(K_mid  ).to(device), torch.from_numpy(E_mid  ).to(device))
        cam_right_t = (torch.from_numpy(K_right).to(device), torch.from_numpy(E_right).to(device))

        return (H, W,
                wpv_left_t, wpv_mid_t, wpv_right_t,
                cam_left_t, cam_mid_t, cam_right_t,
                seq, l_idx, m_idx, r_idx)

    def _compute_geometry(
        self,
        *,
        H: int,
        W: int,
        wpv_left_t: torch.Tensor,
        wpv_mid_t: torch.Tensor,
        wpv_right_t: torch.Tensor,
        cam_left_t: Tuple[torch.Tensor, torch.Tensor],
        cam_mid_t: Tuple[torch.Tensor, torch.Tensor],
        cam_right_t: Tuple[torch.Tensor, torch.Tensor],
        seq: str,
        l_idx: int,
        m_idx: int,
        r_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute point maps and motion maps (torch ops) on current device."""
        # split xyz/valid
        left_xyz,  left_valid  = wpv_left_t[:, :3],  wpv_left_t[:, 3:4]
        mid_xyz,   mid_valid   = wpv_mid_t[:, :3],   wpv_mid_t[:, 3:4]
        right_xyz, right_valid = wpv_right_t[:, :3], wpv_right_t[:, 3:4]

        # point maps
        if self._pm_source == "3d_tracks":
            pm_left  = geo.create_pm_in_ref_frame(left_xyz,  left_valid,  cam_left_t,  cam_left_t,  (H, W), pm_source="3d_tracks")
            pm_mid   = geo.create_pm_in_ref_frame(mid_xyz,   mid_valid,   cam_mid_t,   cam_left_t,  (H, W), pm_source="3d_tracks")
            pm_right = geo.create_pm_in_ref_frame(right_xyz, right_valid, cam_right_t, cam_left_t,  (H, W), pm_source="3d_tracks")
        elif self._pm_source == "dm":
            raise NotImplementedError("pm_source='dm' requires depth maps in shards.")
        else:
            raise NotImplementedError("Unknown pm_source")

        # thresholds: point maps
        if torch.sum(pm_left[..., 3])  < self._min_valid_pc:
            raise AssertionError(f"left point map {seq}:{l_idx} has too few valid points")
        if torch.sum(pm_mid[..., 3])   < self._min_valid_pc:
            raise AssertionError(f"mid  point map {seq}:{m_idx} has too few valid points")
        if torch.sum(pm_right[..., 3]) < self._min_valid_pc:
            raise AssertionError(f"right point map {seq}:{r_idx} has too few valid points")

        # motion maps
        motion = geo.compute_all_motion_maps(
            wpv_left_t, wpv_mid_t, wpv_right_t,
            cam_left_t, cam_mid_t, cam_right_t, (H, W)
        )

        # thresholds: motion maps
        if torch.sum(motion["l2m"][..., 3]) < self._min_valid_mm:
            raise AssertionError(f"l2m motion map {seq}:{l_idx}->{m_idx} has too few valid points")
        if torch.sum(motion["r2m"][..., 3]) < self._min_valid_mm:
            raise AssertionError(f"r2m motion map {seq}:{r_idx}->{m_idx} has too few valid points")
        if torch.sum(motion["l2r"][..., 3]) < self._min_valid_mm:
            raise AssertionError(f"l2r motion map {seq}:{l_idx}->{r_idx} has too few valid points")
        if torch.sum(motion["r2l"][..., 3]) < self._min_valid_mm:
            raise AssertionError(f"r2l motion map {seq}:{r_idx}->{l_idx} has too few valid points")

        return pm_left.float(), pm_mid.float(), pm_right.float(), {k: v.float() for k, v in motion.items()}

    def _maybe_compute_geometry_in_place(self, s: Dict[str, Any], device: torch.device) -> None:
        """Attach geometry tensors to `s` if they are missing."""
        if ("left_pm" in s) and ("motion_gt" in s):
            return
        H, W, wpv_l, wpv_m, wpv_r, cam_l, cam_m, cam_r, seq, li, mi, ri = self._prepare_geom_inputs(s, device)
        pm_l, pm_m, pm_r, motion = self._compute_geometry(
            H=H, W=W,
            wpv_left_t=wpv_l, wpv_mid_t=wpv_m, wpv_right_t=wpv_r,
            cam_left_t=cam_l, cam_mid_t=cam_m, cam_right_t=cam_r,
            seq=seq, l_idx=li, m_idx=mi, r_idx=ri,
        )
        s["left_pm"] = pm_l
        s["mid_pm"] = pm_m
        s["right_pm"] = pm_r
        s["motion_gt"] = motion

    # ---------- decoding (tiny) ----------
    def _decode_wds_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Decode raw WDS sample -> light numpy dict; optionally compute geometry on CPU."""
        try:
            seq, left_idx, mid_idx, right_idx = _parse_sample_key(sample["__key__"])

            # images saved as .npy uint8 HWC
            img_left  = _np_from_npy_bytes(sample["l.npy"]).astype(np.uint8)
            img_mid   = _np_from_npy_bytes(sample["m.npy"]).astype(np.uint8)
            img_right = _np_from_npy_bytes(sample["r.npy"]).astype(np.uint8)

            # tracks (T,4) float32
            wpv_left  = _np_from_npy_bytes(sample["l.pv.npy"]).astype(np.float32)
            wpv_mid   = _np_from_npy_bytes(sample["m.pv.npy"]).astype(np.float32)
            wpv_right = _np_from_npy_bytes(sample["r.pv.npy"]).astype(np.float32)

            # cameras
            K = _np_from_npy_bytes(sample.get("k.npy") or sample.get("K.npy")).astype(np.float32)
            E_left  = _np_from_npy_bytes(sample["l.cam.npy"]).astype(np.float32)
            E_mid   = _np_from_npy_bytes(sample["m.cam.npy"]).astype(np.float32)
            E_right = _np_from_npy_bytes(sample["r.cam.npy"]).astype(np.float32)

            s: Dict[str, Any] = dict(
                seq=seq,
                idxs=(left_idx, mid_idx, right_idx),
                image_left=img_left,
                image_mid=img_mid,
                image_right=img_right,
                wpv_left=wpv_left,
                wpv_mid=wpv_mid,
                wpv_right=wpv_right,
                cam_left=(K, E_left),
                cam_mid=(K, E_mid),
                cam_right=(K, E_right),
                # placeholders to match your interface
                instance_left=None,
                instance_right=None,
            )

            # geometry on CPU (parallel across workers) if requested
            if not self._geo_on_gpu:
                self._maybe_compute_geometry_in_place(s, torch.device("cpu"))

            return s

        except Exception:
            # Let WebDataset handler decide (we ignore and continue).
            return None

    # ---------- collate + packaging ----------
    def _collate_build_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a full batch:
          - Albumentations (CPU) on left/right
          - If geometry not precomputed: compute now on GPU
          - Stack to batch tensors
        """
        # Use CUDA in collate only if we're on the main process (no workers)
        # to avoid "Cannot re-initialize CUDA in forked subprocess".
        device = (
            torch.device("cuda")
            if (self._geo_on_gpu and torch.cuda.is_available() and get_worker_info() is None)
            else torch.device("cpu")
        )
        per_sample_outputs: List[Dict[str, Any]] = []

        for s in samples:
            seq = s["seq"]
            left_idx, mid_idx, right_idx = s["idxs"]
            H, W = s["image_left"].shape[:2]

            # Albumentations on left/right only (numpy)
            img_left_np  = s["image_left"]
            img_mid_np   = s["image_mid"]    # unchanged
            img_right_np = s["image_right"]
            if self._albu is not None:
                img_left_np  = self._albu(image=img_left_np)["image"]
                img_right_np = self._albu(image=img_right_np)["image"]

            # images -> device (CHW float)
            img_left_chw  = _to_chw_float(img_left_np,  device)
            img_mid_chw   = _to_chw_float(img_mid_np,   device)
            img_right_chw = _to_chw_float(img_right_np, device)

            # cameras to device (tuples)
            K_left,  E_left  = s["cam_left"]
            K_mid,   E_mid   = s["cam_mid"]
            K_right, E_right = s["cam_right"]
            cam_left  = (torch.from_numpy(K_left ).to(device), torch.from_numpy(E_left ).to(device))
            cam_mid   = (torch.from_numpy(K_mid  ).to(device), torch.from_numpy(E_mid  ).to(device))
            cam_right = (torch.from_numpy(K_right).to(device), torch.from_numpy(E_right).to(device))

            # geometry: use precomputed or compute now
            if ("left_pm" in s) and ("motion_gt" in s):
                pm_left  = s["left_pm"].to(device, non_blocking=True)
                pm_mid   = s["mid_pm"].to(device, non_blocking=True)
                pm_right = s["right_pm"].to(device, non_blocking=True)
                motion   = {k: v.to(device, non_blocking=True) for k, v in s["motion_gt"].items()}
            else:
                # compute on GPU using the same helpers (no duplication)
                self._maybe_compute_geometry_in_place(s, device)
                pm_left, pm_mid, pm_right = s["left_pm"], s["mid_pm"], s["right_pm"]
                motion = s["motion_gt"]

            # sequence index mapping
            seq_int = self._seq_to_int.setdefault(seq, len(self._seq_to_int))

            # pack per-sample output (only tensors, dicts, tuples for Accelerate concatenation)
            per_sample_outputs.append(dict(
                left_pm=pm_left, mid_pm=pm_mid, right_pm=pm_right,
                motion_gt=motion,
                left_image=img_left_chw.contiguous(),
                mid_image=img_mid_chw.contiguous(),
                right_image=img_right_chw.contiguous(),
                idxs=torch.tensor((left_idx, mid_idx, right_idx), dtype=torch.float32, device=device),
                query_times=torch.tensor(
                    [(mid_idx - left_idx) / max(1, (right_idx - left_idx)), 1.0, 0.0],
                    dtype=torch.float32, device=device
                ),
                sequence_idx=torch.tensor(seq_int, dtype=torch.float32, device=device),
                # Provide instance-identifying integers for model symmetrization logic
                seq_left_idx=torch.tensor(seq_int * 100000 + left_idx, dtype=torch.long, device=device),
                seq_right_idx=torch.tensor(seq_int * 100000 + right_idx, dtype=torch.long, device=device),
                cam=cam_left, cam_mid=cam_mid, cam_right=cam_right,
                # drop non-tensor instance strings to avoid Accelerate concat issues
            ))

        # ---------- stack where shapes agree ----------
        batch: Dict[str, Any] = {}
        keys_to_stack = [
            "left_pm", "mid_pm", "right_pm",
            "left_image", "mid_image", "right_image",
            "idxs", "query_times", "sequence_idx",
            "seq_left_idx", "seq_right_idx",
        ]
        for k in keys_to_stack:
            batch[k] = torch.stack([o[k] for o in per_sample_outputs], dim=0)

        # motion maps: dict of four (H,W,4)
        batch["motion_gt"] = {
            name: torch.stack([o["motion_gt"][name] for o in per_sample_outputs], dim=0)
            for name in per_sample_outputs[0]["motion_gt"].keys()
        }

        # cameras: stack tuple fields to ((B,3,3),(B,4,4))
        batch["cam"], batch["cam_mid"], batch["cam_right"] = _stack_cam_tuples(per_sample_outputs)

        # expose instance identifiers as tensors for safe concatenation by Accelerate
        batch["left_instance"]  = batch.pop("seq_left_idx")
        batch["right_instance"] = batch.pop("seq_right_idx")

        return batch
