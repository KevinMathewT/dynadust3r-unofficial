import math
import torch
import pytest

from models.dust3r.dyna_dust3r import DynaDUSt3R


def build_small_model():
    # Small, CPU-friendly model; dec_depth >= 10 due to DPT head assertion
    model = DynaDUSt3R(
        output_mode="pts3d",
        motion_output_mode="pts3d",
        head_type="dpt",
        motion_head_type="dpt",
        depth_mode=("linear", -float("inf"), float("inf")),
        conf_mode=("exp", 1, float("inf")),
        motion_depth_mode=("linear", -float("inf"), float("inf")),
        motion_conf_mode=("exp", 1, float("inf")),
        depth_post_mode=("linear", -float("inf"), float("inf")),
        motion_depth_post_mode=("linear", -float("inf"), float("inf")),
        freeze="none",
        landscape_only=True,
        patch_embed_cls="ManyAR_PatchEmbed",
        time_pos_emb_dim=16,
        img_size=(32, 32),
        patch_size=16,
        enc_embed_dim=64,
        enc_depth=2,
        enc_num_heads=4,
        dec_embed_dim=64,
        dec_depth=10,
        dec_num_heads=4,
        mlp_ratio=2,
        norm_im2_in_dec=True,
        pos_embed="RoPE100",
    )
    return model


def test_forward_keys_and_shapes_cpu():
    torch.manual_seed(0)
    model = build_small_model()
    B, H, W = 1, 32, 32
    batch = {
        "left_image": torch.zeros(B, 3, H, W),
        "right_image": torch.zeros(B, 3, H, W),
        "query_times": torch.tensor([[0.0, 0.5, 1.0]]),
        "left_instance": ["a"],
        "right_instance": ["b"],
    }

    outputs = model.forward(batch)

    # Base predictions and confidences
    assert outputs["left_map_pred"].shape == (B, H, W, 3)
    assert outputs["right_map_pred_in_left_frame"].shape == (B, H, W, 3)
    assert outputs["left_map_pred_conf"].shape == (B, H, W, 1)
    assert outputs["right_map_pred_conf"].shape == (B, H, W, 1)

    # Motion key semantics for times [0.0, 0.5, 1.0]
    mp = outputs["motion_pred"]
    assert "l_to_r" in mp and "r_to_l" in mp
    # middle value at index 1 → t1 keys should exist
    assert "l_to_t1" in mp and "r_to_t1" in mp
    # extremes are handled by special keys, not index keys
    assert "l_to_t0" not in mp
    assert "r_to_t2" not in mp


def test_get_loss_mid_time_key_missing_raises_keyerror():
    torch.manual_seed(0)
    model = build_small_model()
    B, H, W = 1, 32, 32
    batch_fw = {
        "left_image": torch.zeros(B, 3, H, W),
        "right_image": torch.zeros(B, 3, H, W),
        # Only 0 and 1 → forward makes only r_to_l and l_to_r, no l_to_t0/r_to_t0
        "query_times": torch.tensor([[0.0, 1.0]]),
        "left_instance": ["a"],
        "right_instance": ["b"],
    }

    outputs = model.forward(batch_fw)

    # Build a minimal batch for loss (shapes only); values won't be used before the KeyError
    ones_mask = torch.ones(B, H, W, 1)
    zeros_pc = torch.zeros(B, H, W, 3)
    batch_loss = {
        "left_pm": torch.cat([zeros_pc, ones_mask], dim=-1),
        "right_pm": torch.cat([zeros_pc, ones_mask], dim=-1),
        "motion_gt": {
            "l2m": torch.cat([zeros_pc, ones_mask], dim=-1),
            "r2m": torch.cat([zeros_pc, ones_mask], dim=-1),
            "l2r": torch.cat([zeros_pc, ones_mask], dim=-1),
            "r2l": torch.cat([zeros_pc, ones_mask], dim=-1),
        },
        "left_image": batch_fw["left_image"],
        "right_image": batch_fw["right_image"],
        "query_times": batch_fw["query_times"],
    }

    with pytest.raises(KeyError):
        model.get_loss(batch_loss, outputs)


def test_get_loss_self_consistency_zero_l2_means():
    torch.manual_seed(0)
    model = build_small_model()
    B, H, W = 1, 32, 32
    # Put the middle time at index 0 so get_loss() will look for l_to_t0/r_to_t0
    batch_fw = {
        "left_image": torch.zeros(B, 3, H, W),
        "right_image": torch.zeros(B, 3, H, W),
        "query_times": torch.tensor([[0.5, 0.0, 1.0]]),
        "left_instance": ["a"],
        "right_instance": ["b"],
    }

    outputs = model.forward(batch_fw)

    # Build GT directly from predictions to enforce zero L2 error
    ones_mask = torch.ones(B, H, W, 1)
    left_base = outputs["left_map_pred"].detach()
    right_base = outputs["right_map_pred_in_left_frame"].detach()
    mp = outputs["motion_pred"]

    # Motion keys required by get_loss() with tq_mid_idx=0
    l2m_disp = mp["l_to_t0"].detach()
    r2m_disp = mp["r_to_t0"].detach()
    l2r_disp = mp["l_to_r"].detach()
    r2l_disp = mp["r_to_l"].detach()

    batch_loss = {
        "left_pm": torch.cat([left_base, ones_mask], dim=-1),
        "right_pm": torch.cat([right_base, ones_mask], dim=-1),
        "motion_gt": {
            "l2m": torch.cat([l2m_disp, ones_mask], dim=-1),
            "r2m": torch.cat([r2m_disp, ones_mask], dim=-1),
            "l2r": torch.cat([l2r_disp, ones_mask], dim=-1),
            "r2l": torch.cat([r2l_disp, ones_mask], dim=-1),
        },
        "left_image": batch_fw["left_image"],
        "right_image": batch_fw["right_image"],
        "query_times": batch_fw["query_times"],
    }

    total_loss, details = model.get_loss(batch_loss, outputs)

    # Check L2 statistics are ~0 for all components (left, right, l2m, r2m, l2r, r2l)
    for name in ["left", "right", "l2m", "r2m", "l2r", "r2l"]:
        assert abs(details[f"{name}_l2_mean"]) < 1e-6


