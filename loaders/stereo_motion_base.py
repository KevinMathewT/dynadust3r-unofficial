"""
Author: Kevin Mathew T
Date: 2025-08-17
LinkedIn: https://www.linkedin.com/in/kevinmathewt/
"""

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

# import utils.geometry as geo
# import utils.geometry as geo_motion
import utils.torch_geometry as geo


class StereoMotionBase(Dataset):
    def __init__(self, config, valid: bool = False):
        """
        Base class for stereo motion datasets that handles loading and processing of
        multi-view time series data for stereo and motion analysis.

        This class manages frame triplets (left, middle, right) across sequences, handles
        dataset indexing, and provides the infrastructure for 3D point cloud processing
        between camera views and timestamps. Intended to be subclassed for specific
        dataset implementations.

        Args:
            config: Configuration object containing dataset parameters including:
                - dataset.<dataset>.name: Name identifier for the dataset
                - dataset.<dataset>.path: Base directory path to dataset files
                - dataset.<dataset>.max_frame_window: Maximum temporal window size for frame triplets
            valid: Whether this is a validation dataset (affects triplet count)
        """
        # initialize common parameters
        self.config = config
        self.is_valid = valid  # Store whether this is validation dataset

        self.sequence_paths = []

        # populated by child classes
        self.frame_counts = []

    def _compute_triplets(self):
        """
        Compute a list of triplets through uniform sampling:
        1. Uniformly sample a sequence (each sequence has equal probability)
        2. Uniformly sample a left frame (ensuring room for right frame at min distance 2)
        3. Sample right frame between [left+2, left+max_frame_window]
        4. Sample mid frame between left and right

        The triplets are stored as (seq_idx, left, mid, right) tuples.
        """
        import numpy as np

        self.triplets = []  # initialize triplets list
        
        # Use appropriate target based on whether this is validation or training
        target_triplets = self.config.data.valid_len if self.is_valid else self.config.data.len
        
        while (
            len(self.triplets) < target_triplets
        ):  # generate triplets through uniform sampling
            seq_idx = np.random.randint(0, len(self.sequence_paths))  # sample sequence
            frame_count = self.frame_counts[seq_idx]

            if frame_count < 3:  # skip if too short
                continue
            
            max_left = frame_count - 3  # sample left frame
            if max_left < 0:
                continue
            left_frame = np.random.randint(0, max_left + 1)
            min_right = left_frame + 2  # sample right frame
            max_right = min(left_frame + self.max_frame_window, frame_count - 1)
            right_frame = np.random.randint(min_right, max_right + 1)
            mid_frame = np.random.randint(
                left_frame + 1, right_frame
            )  # sample mid frame
            self.triplets.append(
                (seq_idx, left_frame, mid_frame, right_frame)
            )  # add to list

        if len(self.triplets) > target_triplets:  # trim to exact size if needed
            self.triplets = self.triplets[:target_triplets]

        self.num_triplets = target_triplets  # set final size

    def __len__(self):
        """
        Returns the total number of triplets in the dataset.

        Returns:
            int: Total number of triplets in the dataset
        """
        return self.num_triplets

    def _apply_color_augmentation(self, image):
        """
        Apply dataset-provided color augmentation to a single image.
        Preserves input type (torch.Tensor or numpy.ndarray) and shape (H, W, 3).
        """
        import torch as _torch
        # Expect self.color_aug to be provided by child datasets (NoOp for valid)
        if isinstance(image, _torch.Tensor):
            device = image.device
            img_np = image.detach().cpu().numpy()
            aug_np = self.color_aug(image=img_np)["image"]
            return _torch.from_numpy(aug_np).to(device=device)
        else:
            # numpy path
            return self.color_aug(image=image)["image"]

    def _process_images(self, left_info, mid_info, right_info):
        """
        Process and normalize images for training.
        
        Returns torch tensors (3,H,W) on the same device as input images.
        """
        m = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=left_info["image"].device).view(3, 1, 1)
        s = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=left_info["image"].device).view(3, 1, 1)

        def _proc(info):
            # info["image"]: (H,W,3) torch tensor; convert to CHW float
            img = info["image"].to(dtype=torch.float32)
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)
            img = img / 255.0
            img = (img - m) / s
            return img

        return _proc(left_info), _proc(mid_info), _proc(right_info)

    def __getitem__(self, index):
        """
        fetch a triplet of frames and compute associated point maps, motion maps,
        and images for training.

        this method returns a dict containing:
        - left_pm, mid_pm, right_pm: point maps in reference frame (H, W, 4)
        - motion_gt: dict of motion ground truths with keys:
            - 'l2m': left->mid motion map (H, W, 3)
            - 'r2m': right->mid motion map (H, W, 3)
            - 'l2r': left->right motion map (H, W, 3)
            - 'r2l': right->left motion map (H, W, 3)
        - left_image, mid_image, right_image: normalized RGB images (3, H, W)
        - idxs: tensor of frame indices (3,)
        - query_times: tensor of temporal query positions [mid, right, left] (3,)
        - sequence_idx: tensor identifying sequence index ()
        - left_instance, right_instance: optional instance identifiers (str or None)
        - cam, cam_mid, cam_right: camera extrinsic & intrinsic tuples for each view
        """
        import numpy as np

        while True:
            # get triplet info
            seq_idx, left_frame, mid_frame, right_frame = self.triplets[index]
            sequence_path = self.sequence_paths[seq_idx]

            try:
                # optimized multi-frame fetch
                left_info, mid_info, right_info = self.get_frame_infos(sequence_path, [left_frame, mid_frame, right_frame])

                # apply color augmentation to left and right images only
                left_info["image"] = self._apply_color_augmentation(left_info["image"])  # (H, W, 3)
                right_info["image"] = self._apply_color_augmentation(right_info["image"])  # (H, W, 3)

                # Ensure frames have tracks
                assert len(left_info["world_pc_valid"]) > 0, f"left frame {sequence_path}:{left_frame} has no tracks"
                assert len(mid_info["world_pc_valid"]) > 0, f"mid frame {sequence_path}:{mid_frame} has no tracks"  
                assert len(right_info["world_pc_valid"]) > 0, f"right frame {sequence_path}:{right_frame} has no tracks"

                reference_cam = left_info["cam"]  # Use left camera as reference
                
                # Extract world points and validity (torch)
                left_world_pc = left_info["world_pc_valid"][:, :3]
                mid_world_pc = mid_info["world_pc_valid"][:, :3]
                right_world_pc = right_info["world_pc_valid"][:, :3]
                
                left_valid = left_info["world_pc_valid"][:, 3:4]
                mid_valid = mid_info["world_pc_valid"][:, 3:4]
                right_valid = right_info["world_pc_valid"][:, 3:4]
                
                # Get dataset config
                dataset_config = getattr(self.config.dataset, self.config.data.loader.lower(), None)
                pm_source = getattr(dataset_config, "pm_source", "") if dataset_config else ""
                min_valid_pc_points = getattr(dataset_config, "min_valid_pc_points", 0) if dataset_config else 0
                min_valid_mm_points = getattr(dataset_config, "min_valid_mm_points", 0) if dataset_config else 0
                
                if pm_source == "3d_tracks":
                    # Create point maps with torch ops
                    left_pm = geo.create_pm_in_ref_frame(
                        left_world_pc, left_valid, left_info["cam"], reference_cam,
                        left_info["image"].shape[:2], pm_source="3d_tracks"
                    )
                    mid_pm = geo.create_pm_in_ref_frame(
                        mid_world_pc, mid_valid, mid_info["cam"], reference_cam,
                        mid_info["image"].shape[:2], pm_source="3d_tracks"
                    )
                    right_pm = geo.create_pm_in_ref_frame(
                        right_world_pc, right_valid, right_info["cam"], reference_cam,
                        right_info["image"].shape[:2], pm_source="3d_tracks"
                    )
                elif pm_source == "dm":
                    left_pm = geo.dm_to_cam_pm(left_info["dm"], reference_cam)
                    mid_pm = geo.create_pm_from_dm_in_ref_frame(
                        mid_info["dm"], mid_info["cam"], reference_cam
                    )
                    right_pm = geo.create_pm_from_dm_in_ref_frame(
                        right_info["dm"], right_info["cam"], reference_cam
                    )
                else:
                    raise NotImplementedError("point map source not implemented")

                # Assert that point maps have valid points (torch)
                assert torch.sum(left_pm[:, :, 3]) >= min_valid_pc_points, f"left point map {sequence_path}:{left_frame} has {torch.sum(left_pm[:, :, 3]).item()} valid points, need at least {min_valid_pc_points}"
                assert torch.sum(mid_pm[:, :, 3]) >= min_valid_pc_points, f"mid point map {sequence_path}:{mid_frame} has {torch.sum(mid_pm[:, :, 3]).item()} valid points, need at least {min_valid_pc_points}"
                assert torch.sum(right_pm[:, :, 3]) >= min_valid_pc_points, f"right point map {sequence_path}:{right_frame} has {torch.sum(right_pm[:, :, 3]).item()} valid points, need at least {min_valid_pc_points}"

                # Compute motion maps with torch
                H, W = left_info["image"].shape[:2]
                motion_gt = geo.compute_all_motion_maps(
                    left_info["world_pc_valid"],
                    mid_info["world_pc_valid"],
                    right_info["world_pc_valid"],
                    left_info["cam"],
                    mid_info["cam"],
                    right_info["cam"],
                    (H, W)
                )

                assert torch.sum(motion_gt["l2m"][..., 3]) >= min_valid_mm_points, f"l2m motion map {sequence_path}:{left_frame}->{mid_frame} has {torch.sum(motion_gt['l2m'][..., 3]).item()} valid points, need at least {min_valid_mm_points}"
                assert torch.sum(motion_gt["r2m"][..., 3]) >= min_valid_mm_points, f"r2m motion map {sequence_path}:{right_frame}->{mid_frame} has {torch.sum(motion_gt['r2m'][..., 3]).item()} valid points, need at least {min_valid_mm_points}"
                assert torch.sum(motion_gt["l2r"][..., 3]) >= min_valid_mm_points, f"l2r motion map {sequence_path}:{left_frame}->{right_frame} has {torch.sum(motion_gt['l2r'][..., 3]).item()} valid points, need at least {min_valid_mm_points}"
                assert torch.sum(motion_gt["r2l"][..., 3]) >= min_valid_mm_points, f"r2l motion map {sequence_path}:{right_frame}->{left_frame} has {torch.sum(motion_gt['r2l'][..., 3]).item()} valid points, need at least {min_valid_mm_points}"

                break
            except Exception as e:
                import traceback
                
                print(f"error loading triplet: {seq_idx}:{sequence_path}, {left_frame}, {mid_frame}, {right_frame}")
                print(f"error: {e}")
                traceback.print_exc()
                
                index = np.random.randint(0, len(self.triplets))

        # Process images (torch, returns CHW)
        left_image, mid_image, right_image = self._process_images(left_info, mid_info, right_info)

        # Build return dictionary (torch tensors)
        result = {
            "left_pm": left_pm.float(),  # (H, W, 4)
            "mid_pm": mid_pm.float(),  # (H, W, 4)
            "right_pm": right_pm.float(),  # (H, W, 4)
            
            # Generic motion ground truths (torch)
            "motion_gt": {k: v.float() for k, v in motion_gt.items()},
            
            "left_image": left_image.contiguous(),  # (3, H, W)
            "mid_image": mid_image.contiguous(),  # (3, H, W)
            "right_image": right_image.contiguous(),  # (3, H, W)
            
            "idxs": torch.tensor((left_frame, mid_frame, right_frame), dtype=torch.float32, device=left_image.device),
            "query_times": torch.tensor([(mid_frame - left_frame) / (right_frame - left_frame), 1.0, 0.0], dtype=torch.float32, device=left_image.device),
            "sequence_idx": torch.tensor(seq_idx, dtype=torch.float32, device=left_image.device),
            
            "left_instance": left_info.get("instance", None),
            "right_instance": right_info.get("instance", None),
            
            "cam": left_info["cam"],
            "cam_mid": mid_info["cam"],
            "cam_right": right_info["cam"],
        }
        
        return result


    def crop_data(self, image, dm, cam, output_resolution):
        """
        Apply consistent cropping and resizing to image, depth map, and camera parameters.

        Args:
            image: RGB image array (H, W, 3)
            dm: Depth map array (H, W)
            cam: Camera parameters tuple ((3,3) intrinsics, (4,4) extrinsics)
            output_resolution: Target output resolution (width, height)

        Returns:
            tuple: (cropped_image, cropped_dm, adjusted_cam)
        """
        import utils.geometry as geo

        intrinsics, extrinsics = cam  # ((3,3), (4,4))

        cropped_image, cropped_dm, adjusted_intrinsics = geo.crop_resize_if_necessary(
            image, dm, intrinsics, output_resolution
        )  # apply transformations

        if isinstance(cropped_image, Image.Image):
            cropped_image = np.array(cropped_image)  # (H, W, 3)

        adjusted_cam = (adjusted_intrinsics, extrinsics)  # ((3,3), (4,4))

        return cropped_image, cropped_dm, adjusted_cam

    def get_frame_info(self, sequence_path, frame_index):
        """
        Abstract method to retrieve comprehensive information for a specific frame.

        This method must be implemented by child classes to handle dataset-specific
        frame loading and processing. It should load all necessary data for a frame
        including images, depth maps, camera parameters, and point clouds.

        Args:
            sequence_path (str): Path to the sequence directory containing frame data
            frame_index (int): Index of the frame to retrieve within the sequence

        Returns:
            dict: Dictionary containing frame information with at minimum these keys:
                - image: RGB image array (H, W, 3)
                - dm: Depth map array (H, W)
                - cam: Camera parameters tuple ((3,3) intrinsic matrix, (4,4) extrinsic matrix)
                - world_pc_valid: World point cloud with validity flags (N, 4)
                  where the first 3 columns are XYZ coordinates and the 4th is a validity flag

        Raises:
            NotImplementedError: This method must be overridden by child classes
        """
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )
    

    def get_frame_infos(self, sequence_path, frame_idxs):
        """
        Optimized batch loading of multiple frames from the same sequence.
        
        This method provides an efficient alternative to multiple get_frame_info calls
        by opening files only once and loading all requested frames in a single pass.
        This significantly reduces I/O overhead when loading frame triplets or other
        multi-frame batches from the same sequence.

        Args:
            sequence_path (str): Path to the sequence directory containing frame data
            frame_idxs (list): List of frame indices to retrieve within the sequence

        Returns:
            list: List of dictionaries, one for each requested frame, containing frame 
                  information with at minimum these keys:
                - image: RGB image array (H, W, 3)
                - dm: Depth map array (H, W) or None if not available
                - cam: Camera parameters tuple ((3,3) intrinsic matrix, (4,4) extrinsic matrix)
                - world_pc_valid: World point cloud with validity flags (N, 4)
                  where the first 3 columns are XYZ coordinates and the 4th is a validity flag
                - instance: Optional instance identifier (str or None)

        Raises:
            NotImplementedError: This method must be overridden by child classes
        """
