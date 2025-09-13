import os
import sys
import math
import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf


# Ensure repository root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loaders.stereo4d_wds import Stereo4DWDS


def tensor_on_same_device(objs, device):
    for obj in objs:
        if not isinstance(obj, torch.Tensor):
            return False
        if obj.device != device:
            return False
    return True


def get_intr_extr(cam):
    intr, extr = cam
    return intr, extr


def proj_pixels(K, Pc):
    # Pc: (...,3), returns (...,2) float pixels and z
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x, y, z = Pc[..., 0], Pc[..., 1], Pc[..., 2]
    u = fx * x / z + cx
    v = fy * y / z + cy
    return u, v, z


def round_to_int_uv(u, v):
    return torch.round(u).to(torch.int64), torch.round(v).to(torch.int64)


def leftref_to_right_cam(P_left_ref, E_left, E_right):
    # E_*: world->cam 4x4, P_left_ref: (...,3) in left cam coords
    Rl, tl = E_left[:3, :3], E_left[:3, 3]
    Rr, tr = E_right[:3, :3], E_right[:3, 3]
    Pw = (Rl.transpose(-2, -1) @ (P_left_ref - tl).transpose(-2, -1)).transpose(-2, -1)
    PcR = (Rr @ Pw.transpose(-2, -1)).transpose(-2, -1) + tr
    return PcR


def check_rotation(R, atol=1e-4, rtol=1e-4):
    I = torch.eye(3, device=R.device, dtype=R.dtype)
    return torch.allclose(R.transpose(-2, -1) @ R, I, atol=atol, rtol=rtol) and (torch.det(R) > 0.0)


def main():
    parser = argparse.ArgumentParser("Stereo4DWDS consistency checks")
    parser.add_argument("--samples", type=int, default=3, help="number of triplets to check")
    parser.add_argument("--per_sample_points", type=int, default=200, help="per-sample correspondences to test")
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--wds_dir", type=str, default=None, help="path to WebDataset root containing train/test subdirs")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Build minimal config (no Hydra) using dataset YAML
    try:
        ds_yaml = OmegaConf.load(str(ROOT / "config" / "dataset" / "stereo4d.yaml"))
    except Exception as e:
        print(f"FAIL: could not load dataset YAML: {e}")
        sys.exit(2)

    cfg = OmegaConf.create({
        "seed": args.seed,
        "debug": True,
        "data": {
            "loader": "stereo4d",
            "len": max(2, args.samples),
            "valid_len": max(2, args.samples),
            "batch_size": 1,
            "num_workers": 0,
        },
        "dataset": {
            "stereo4d": ds_yaml,
        }
    })

    # Determine wds_dir
    split = cfg.dataset.stereo4d.train_split
    wds_dir_cli = args.wds_dir or os.environ.get("STEREO4D_WDS_DIR")
    wds_dir_cfg = cfg.dataset.stereo4d.get("wds_dir")

    def infer_wds_dir_from_cfg():
        base = cfg.dataset.stereo4d.get("path")
        if base:
            guess = Path(base) / "wds"
            if (guess / split / "stereo4d-idx.json").is_file():
                return str(guess)
            cand = list(Path(base).rglob("stereo4d-idx.json"))
            if len(cand):
                return str(cand[0].parents[1])
        return None

    resolved_wds = wds_dir_cli or (wds_dir_cfg if isinstance(wds_dir_cfg, str) and wds_dir_cfg.strip(" ?") else None) or infer_wds_dir_from_cfg()
    if not resolved_wds:
        print("FAIL: dataset.stereo4d.wds_dir is not set and could not be inferred. Use --wds_dir or STEREO4D_WDS_DIR.")
        sys.exit(2)
    idx_json = Path(resolved_wds) / split / "stereo4d-idx.json"
    if not idx_json.is_file():
        print(f"FAIL: missing WDS index: {idx_json}")
        sys.exit(2)
    cfg.dataset.stereo4d.wds_dir = resolved_wds

    # Build dataset
    try:
        ds = Stereo4DWDS(cfg, valid=False)
    except Exception as e:
        print(f"FAIL: could not construct Stereo4DWDS: {e}")
        sys.exit(2)

    n = min(args.samples, len(ds))
    if n == 0:
        print("FAIL: dataset contains 0 samples to test")
        sys.exit(2)

    total_samples = 0
    totals = {
        "type_device_ok": 0,
        "shape_ok": 0,
        "cams_ok": 0,
        "idxs_ok": 0,
        "query_times_ok": 0,
        "pm_motion_valid_nonzero": 0,
        "l2r_r2l_inverse_ok": 0,
        "pm_l2r_alignment_ok": 0,
        "l2r_compose_via_mid_ok": 0,
    }

    failures = []

    for i in range(n):
        try:
            sample = ds[i]
        except Exception as e:
            failures.append(f"[{i}] exception getting sample: {e}")
            continue

        total_samples += 1

        # Unpack
        left_pm = sample["left_pm"]
        mid_pm = sample["mid_pm"]
        right_pm = sample["right_pm"]
        motion = sample["motion_gt"]
        left_img = sample["left_image"]
        mid_img = sample["mid_image"]
        right_img = sample["right_image"]
        idxs = sample["idxs"]      # float tensor (left, mid, right)
        qtimes = sample["query_times"]
        camL = sample["cam"]
        camM = sample["cam_mid"]
        camR = sample["cam_right"]

        device = left_pm.device

        # 1) Type/device/dtype checks
        ok_types = True
        tensors_expected = [
            left_pm, mid_pm, right_pm,
            motion["l2m"], motion["r2m"], motion["l2r"], motion["r2l"],
            left_img, mid_img, right_img, idxs, qtimes,
            camL[0], camL[1], camM[0], camM[1], camR[0], camR[1],
        ]
        if not tensor_on_same_device(tensors_expected, device):
            ok_types = False
        else:
            for t in tensors_expected:
                if t.dtype != torch.float32:
                    ok_types = False
                    break
        if ok_types:
            totals["type_device_ok"] += 1
        else:
            failures.append(f"[{i}] type/device/dtype mismatch")

        # 2) Shape/contiguity checks
        H, W = left_pm.shape[:2]
        ok_shapes = (
            left_pm.shape == (H, W, 4)
            and mid_pm.shape == (H, W, 4)
            and right_pm.shape == (H, W, 4)
            and motion["l2m"].shape == (H, W, 4)
            and motion["r2m"].shape == (H, W, 4)
            and motion["l2r"].shape == (H, W, 4)
            and motion["r2l"].shape == (H, W, 4)
            and left_img.shape[1:] == (H, W)
            and mid_img.shape[1:] == (H, W)
            and right_img.shape[1:] == (H, W)
            and left_img.shape[0] == 3
            and mid_img.shape[0] == 3
            and right_img.shape[0] == 3
        )
        ok_shapes = ok_shapes and left_img.is_contiguous() and mid_img.is_contiguous() and right_img.is_contiguous()
        if ok_shapes:
            totals["shape_ok"] += 1
        else:
            failures.append(f"[{i}] shape/contiguity mismatch")

        # 3) Camera sanity
        K_L, E_L = get_intr_extr(camL)
        K_M, E_M = get_intr_extr(camM)
        K_R, E_R = get_intr_extr(camR)

        cams_ok = (
            K_L.shape == (3, 3) and K_M.shape == (3, 3) and K_R.shape == (3, 3)
            and E_L.shape == (4, 4) and E_M.shape == (4, 4) and E_R.shape == (4, 4)
            and check_rotation(E_L[:3, :3]) and check_rotation(E_M[:3, :3]) and check_rotation(E_R[:3, :3])
            and K_L[0, 0] > 0 and K_L[1, 1] > 0 and K_R[0, 0] > 0 and K_R[1, 1] > 0
            and 0 <= K_L[0, 2] <= W and 0 <= K_L[1, 2] <= H
            and 0 <= K_R[0, 2] <= W and 0 <= K_R[1, 2] <= H
        )
        if cams_ok:
            totals["cams_ok"] += 1
        else:
            failures.append(f"[{i}] camera matrices invalid")

        # 4) Index / query_times
        left_idx, mid_idx, right_idx = int(idxs[0].item()), int(idxs[1].item()), int(idxs[2].item())
        idxs_ok = (left_idx < mid_idx < right_idx)
        qt = (mid_idx - left_idx) / max(1, (right_idx - left_idx))
        qtimes_ok = (
            torch.allclose(qtimes[0], torch.tensor(qt, device=device), atol=1e-6)
            and torch.allclose(qtimes[1], torch.tensor(1.0, device=device))
            and torch.allclose(qtimes[2], torch.tensor(0.0, device=device))
        )
        if idxs_ok:
            totals["idxs_ok"] += 1
        else:
            failures.append(f"[{i}] idxs not strictly increasing")
        if qtimes_ok:
            totals["query_times_ok"] += 1
        else:
            failures.append(f"[{i}] query_times incorrect")

        # 5) Minimal valid coverage
        valid_counts = {
            "left_pm": int(torch.sum(left_pm[..., 3]).item()),
            "mid_pm": int(torch.sum(mid_pm[..., 3]).item()),
            "right_pm": int(torch.sum(right_pm[..., 3]).item()),
            "l2m": int(torch.sum(motion["l2m"][..., 3]).item()),
            "r2m": int(torch.sum(motion["r2m"][..., 3]).item()),
            "l2r": int(torch.sum(motion["l2r"][..., 3]).item()),
            "r2l": int(torch.sum(motion["r2l"][..., 3]).item()),
        }
        if all(v > 0 for v in valid_counts.values()):
            totals["pm_motion_valid_nonzero"] += 1
        else:
            failures.append(f"[{i}] some pm/motion validity sums are zero: {valid_counts}")

        # 6) Geometry checks on sampled correspondences
        # Sample pixels valid in left pm and l2r
        l2r_valid = (motion["l2r"][..., 3] > 0) & (left_pm[..., 3] > 0)
        coords = torch.nonzero(l2r_valid)
        if coords.numel() == 0:
            failures.append(f"[{i}] no valid l2r pixels to test")
            continue

        # Random subset
        num_test = min(args.per_sample_points, coords.shape[0])
        sel = coords[torch.randperm(coords.shape[0], device=device)[:num_test]]
        v_sel, u_sel = sel[:, 0], sel[:, 1]

        # Extract left ref 3D and deltas
        P_left_ref = left_pm[v_sel, u_sel, :3]
        d_lr = motion["l2r"][v_sel, u_sel, :3]
        P_right_ref_pred = P_left_ref + d_lr

        # Project predicted right 3D into right camera pixels
        PcR = leftref_to_right_cam(P_right_ref_pred, E_L, E_R)
        u_r_f, v_r_f, z_r = proj_pixels(K_R, PcR)
        in_front = z_r > 0
        u_r, v_r = round_to_int_uv(u_r_f[in_front], v_r_f[in_front])
        P_right_ref_pred = P_right_ref_pred[in_front]
        v_sel2 = v_sel[in_front]
        u_sel2 = u_sel[in_front]

        in_bounds = (u_r >= 0) & (u_r < W) & (v_r >= 0) & (v_r < H)
        u_r = u_r[in_bounds]
        v_r = v_r[in_bounds]
        P_right_ref_pred = P_right_ref_pred[in_bounds]
        v_sel2 = v_sel2[in_bounds]
        u_sel2 = u_sel2[in_bounds]

        if u_r.numel() == 0:
            failures.append(f"[{i}] all projected right pixels out-of-bounds or behind camera")
            continue

        # 6a) Right PM alignment at target pixel (allow ambiguity skip if pixel collision)
        rp_valid = right_pm[v_r, u_r, 3] > 0
        v_r_ok = v_r[rp_valid]
        u_r_ok = u_r[rp_valid]
        P_right_ref_ok = P_right_ref_pred[rp_valid]
        if v_r_ok.numel() == 0:
            failures.append(f"[{i}] projected right pixels not marked valid in right_pm")
        else:
            P_right_ref_pm = right_pm[v_r_ok, u_r_ok, :3]
            abs_err = torch.max(torch.abs(P_right_ref_pm - P_right_ref_ok), dim=1).values
            rel_denom = torch.clamp(P_right_ref_ok.abs().max(dim=1).values, min=1e-6)
            rel_err = abs_err / rel_denom
            pm_alignment_pass = torch.mean(((abs_err <= 1e-2) | (rel_err <= 1e-2)).to(torch.float32))
            if pm_alignment_pass >= 0.9:
                totals["pm_l2r_alignment_ok"] += 1
            else:
                failures.append(f"[{i}] right_pm 3D alignment low pass rate: {float(pm_alignment_pass):.2f}")

        # 6b) r2l inverse at those right pixels: r2l ≈ -l2r (for the same 3D point)
        r2l_vec = motion["r2l"][v_r, u_r, :3]
        inv_ok = torch.mean((torch.norm(r2l_vec + d_lr[in_front][in_bounds], dim=1) <= 5e-3).to(torch.float32))
        if inv_ok >= 0.9:
            totals["l2r_r2l_inverse_ok"] += 1
        else:
            failures.append(f"[{i}] r2l inverse consistency low pass rate: {float(inv_ok):.2f}")

        # 6c) l2r ≈ l2m + m2r; m2r = -r2m at corresponding right pixel
        l2m_vec = motion["l2m"][v_sel2, u_sel2, :3]
        r2m_on_right = motion["r2m"][v_r, u_r, :3]
        composed = l2m_vec + (-r2m_on_right)
        comp_ok = torch.mean((torch.norm(composed - d_lr[in_front][in_bounds], dim=1) <= 5e-3).to(torch.float32))
        if comp_ok >= 0.85:
            totals["l2r_compose_via_mid_ok"] += 1
        else:
            failures.append(f"[{i}] l2r composition via mid low pass rate: {float(comp_ok):.2f}")

    # Print summary
    print("Stereo4DWDS consistency check summary")
    print(f"- samples tested: {total_samples}")
    for k, v in totals.items():
        mark = "PASS" if v == total_samples else ("WARN" if v > 0 else "FAIL")
        print(f"{mark:5s} {k}: {v}/{total_samples}")

    if len(failures) == 0:
        print("All checks passed at required thresholds.")
        sys.exit(0)
    else:
        print("\nIssues:")
        for f in failures:
            print(f" - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()


