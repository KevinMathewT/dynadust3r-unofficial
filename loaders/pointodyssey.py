"""
Author: Kevin Mathew T
Date: 2025-03-10
"""

import cv2
import numpy as np
import loaders.utils.geometry as geo


class PointOdyssey:
    def __init__(
        self,
        dataset_location="data/pointodyssey",
        dset="train",
    ):
        print("loading pointodyssey dataset...")

        self.dataset_label = "pointodyssey"
        self.split = dset

    def get_frame_info(self, sequence_path, frame_index):
        image = cv2.imread(f"{sequence_path}/rgbs/rgb_{frame_index:05d}.jpg")[
            :, :, ::-1
        ]  # (H, W, 3)
        dm_16bit = cv2.imread(
            f"{sequence_path}/depths/depth_{frame_index:05d}.png", cv2.IMREAD_ANYDEPTH
        )  # (H, W)
        dm = dm_16bit.astype(np.float32) / 65535.0 * 1000.0

        annotations = np.load(f"{sequence_path}/anno.npz", allow_pickle=True)
        intrinsics = annotations["intrinsics"][frame_index]  # (3, 3)
        extrinsics = annotations["extrinsics"][frame_index]  # (4, 4) homogeneous
        cam = (intrinsics, extrinsics)
        world_pc = annotations["trajs_3d"][frame_index]  # (N, 3)
        validity = annotations["visibs"][frame_index][..., np.newaxis]  # (N, 1)
        # validity = annotations["valids"][frame_index][..., np.newaxis]  # (N, 1)
        world_pc_valid = np.concatenate([world_pc, validity], axis=1)  # (N, 4)

        return {
            "image": image,  # (H, W, 3)
            "world_pc_valid": world_pc_valid ,  # (N, 4) scaled by factor "scale"
            "cam": cam,  # ((3, 3), (4, 4) homogeneous)
            "dm": dm,  # convert to meters, (H, W)
        }


def main():
    import loaders.utils.viz as viz

    # viz.test_rerun()
    # viz.test_visualize_pc()

    dataset = PointOdyssey(dset="sample")

    seq_path = "data/pointodyssey/sample/r4_new_f"
    frame_info = dataset.get_frame_info(seq_path, 0)

    for k, v in frame_info.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v[0].shape, v[1].shape)

    # viz.visualize_cam_movement_in_world(dataset, seq_path, num_frames=10)

    world_pc_valid = frame_info["world_pc_valid"]
    dm = frame_info["dm"]
    cam = frame_info["cam"]
    world_pc = np.asarray(world_pc_valid[:, :3])
    cam_pc = geo.world_pc_to_cam_pc(world_pc, cam)
    cam_dm_pc = geo.dm_to_cam_pc(dm, cam)
    cam_pm = geo.cam_pc_to_cam_pm(cam_pc, cam, dm.shape)
    cam_pm2 = geo.cam_pc_to_cam_pm(cam_dm_pc, cam, dm.shape)

    print(f"world_pc_valid shape: {world_pc_valid.shape}")
    print(f"world_pc_valid sample: {world_pc_valid[:5]}")
    print(f"Depth map min/max: {dm.min()}, {dm.max()}")
    print(f"Camera PC min/max: {cam_pc.min()}, {cam_pc.max()}")
    print(f"Camera PC sample: {cam_pc[:5]}")
    print(f"Camera DM PC min/max: {cam_dm_pc.min()}, {cam_dm_pc.max()}")
    print(f"Camera DM PC sample: {cam_dm_pc[:5]}")

    print("Min depth:", np.min(dm))
    print("Max depth:", np.max(dm))
    print("Number of (0,0,0) points:", np.sum((cam_pc == [0, 0, 0]).all(axis=1)))
    print("Number of (0,0,0) points:", np.sum((cam_dm_pc == [0, 0, 0]).all(axis=1)))

    print("Visualizing Point Cloud...")
    viz.visualize_pc(
        cam_pc,
        image=frame_info["image"],
        cam=cam,
        valid=True,
        pc_in_cam_coords=True,
    )
    print("Visualizing Point Cloud Done.")

    print("Visualizing Point Cloud...")
    viz.visualize_pc(
        cam_dm_pc,
        image=frame_info["image"],
        cam=cam,
        valid=True,
        pc_in_cam_coords=True,
        # name="cam_dm_pc",
    )
    print("Visualizing Point Cloud Done.")

    print("Visualizing Point Map...")
    viz.visualize_pm(
        cam_pm,
        image=frame_info["image"],
        cam=cam,
        valid=True,
        pc_in_cam_coords=True,
    )


    print("Visualizing Point Map...")
    viz.visualize_pm(
        cam_pm2,
        image=frame_info["image"],
        cam=cam,
        valid=True,
        pc_in_cam_coords=True,
        # name="cam_dm_pc",
    )

    t = 50

    frame_infos = [dataset.get_frame_info(seq_path, i) for i in range(t)]
    cams = [f["cam"] for f in frame_infos]
    images = [f["image"] for f in frame_infos]

    world_pcs = [f["world_pc_valid"][:, :3] for f in frame_infos]
    valid_flags = [f["world_pc_valid"][..., 3:4] for f in frame_infos]

    cam_pcs = [geo.world_pc_to_cam_pc(world_pcs[i], cams[0]) for i in range(t)]
    cam_pc_valids = [np.concatenate([cam_pcs[i], valid_flags[i]], axis=1) for i in range(t)]

    print(f"valid in cam_pc: {[int(cpv[:, -1].sum()) for cpv in cam_pc_valids]}")

    cam_pms = [geo.cam_pc_to_cam_pm(cam_pc_valids[i], (cams[i][0], None), frame_infos[i]["dm"].shape, valid=True) for i in range(t)]
    cam_dm_pms = [geo.dm_to_cam_pm(frame_infos[i]["dm"], cams[i]) for i in range(t)]
    print(f"valid in cam_pm: {[cp[..., -1].sum() for cp in cam_pms]}")

    motion_map = geo.get_motion_map_from_cam_pc(cam_pc_valids, cams[0][0], frame_infos[0]["dm"].shape)

    print("--------------------------- total points -> valid points -> motion valid points")
    for ti in range(motion_map.shape[0]):
        print(f"valid in motion_map at t={ti}: {frame_infos[ti]['world_pc_valid'].shape[0]} + {frame_infos[ti + 1]['world_pc_valid'].shape[0]} -> {int(cam_pc_valids[ti][:, -1].sum())} + {int(cam_pc_valids[ti + 1][:, -1].sum())} -> {int(motion_map[ti, ..., -1].sum())}")

    viz.visualize_sequence_from_pms(np.asarray(cam_pms), motion_map, images, name="cam_pms_motion_map")
    viz.visualize_sequence_from_pms(np.asarray(cam_dm_pms), motion_map, images, name="cam_dm_pms_motion_map")


if __name__ == "__main__":
    main()
