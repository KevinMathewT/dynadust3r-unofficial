import numpy as np
from PIL import Image
import time

import torch
from torch.utils.data import Dataset

import loaders.utils.geometry as geo


class StereoMotionBase(Dataset):
    def __init__(self, config, valid: bool = False, time_debug: bool = False):
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
            time_debug: Whether to print timing information for each operation
        """
        # initialize common parameters
        self.config = config
        self.is_valid = valid  # Store whether this is validation dataset
        self.time_debug = time_debug  # Store time debug flag

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

    def _process_images(self, left_info, mid_info, right_info):
        """
        Process and normalize images for training.

        Args:
            left_info: Dictionary containing left frame information including "image" key
            mid_info: Dictionary containing mid frame information including "image" key  
            right_info: Dictionary containing right frame information including "image" key

        Returns:
            tuple: (left_image, mid_image, right_image) - normalized images with values in [-1, 1]
        """
        m = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        s = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        left_image = (left_info["image"].astype(np.float32) / 255 - m) / s
        mid_image = (mid_info["image"].astype(np.float32) / 255 - m) / s
        right_image = (right_info["image"].astype(np.float32) / 255 - m) / s

        return left_image, mid_image, right_image

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
        if self.time_debug:
            total_start = time.perf_counter()

        while True:
            if self.time_debug:
                loop_start = time.perf_counter()
                
            # get triplet info
            if self.time_debug:
                t0 = time.perf_counter()
            seq_idx, left_frame, mid_frame, right_frame = self.triplets[index]
            sequence_path = self.sequence_paths[seq_idx]
            if self.time_debug:
                print(f"[TIME] Get triplet info: {(time.perf_counter() - t0)*1000:.2f}ms")

            try:
                # fetch frame info
                if self.time_debug:
                    t0 = time.perf_counter()
                    
                # ~21 mins for 200 batches
                # left_info = self.get_frame_info(sequence_path, left_frame)
                # mid_info = self.get_frame_info(sequence_path, mid_frame)
                # right_info = self.get_frame_info(sequence_path, right_frame)

                # ~ 8 mins for 200 batches
                left_info, mid_info, right_info = self.get_frame_infos(sequence_path, [left_frame, mid_frame, right_frame])
                if self.time_debug:
                    print(f"[TIME] Get frame infos: {(time.perf_counter() - t0)*1000:.2f}ms")
                
                # Assert that all frames have tracks
                if self.time_debug:
                    t0 = time.perf_counter()
                assert len(left_info["world_pc_valid"]) > 0, f"left frame {sequence_path}:{left_frame} has no tracks"
                assert len(mid_info["world_pc_valid"]) > 0, f"mid frame {sequence_path}:{mid_frame} has no tracks"  
                assert len(right_info["world_pc_valid"]) > 0, f"right frame {sequence_path}:{right_frame} has no tracks"
                if self.time_debug:
                    print(f"[TIME] Assert world_pc_valid: {(time.perf_counter() - t0)*1000:.2f}ms")
                
                if self.time_debug:
                    t0 = time.perf_counter()
                reference_cam = left_info["cam"]  # Use left camera as reference
                
                # Extract world points and validity
                left_world_pc = left_info["world_pc_valid"][:, :3]
                mid_world_pc = mid_info["world_pc_valid"][:, :3]
                right_world_pc = right_info["world_pc_valid"][:, :3]
                
                left_valid = left_info["world_pc_valid"][:, 3:4]
                mid_valid = mid_info["world_pc_valid"][:, 3:4]
                right_valid = right_info["world_pc_valid"][:, 3:4]
                if self.time_debug:
                    print(f"[TIME] Extract world points: {(time.perf_counter() - t0)*1000:.2f}ms")

                # Get dataset config
                if self.time_debug:
                    t0 = time.perf_counter()
                dataset_config = getattr(self.config.dataset, self.config.data.loader.lower(), None)
                pm_source = getattr(dataset_config, "pm_source", "") if dataset_config else ""
                min_valid_pc_points = getattr(dataset_config, "min_valid_pc_points", 0) if dataset_config else 0
                min_valid_mm_points = getattr(dataset_config, "min_valid_mm_points", 0) if dataset_config else 0
                if self.time_debug:
                    print(f"[TIME] Get dataset config: {(time.perf_counter() - t0)*1000:.2f}ms")
                
                if pm_source == "3d_tracks":
                    if self.time_debug:
                        t0 = time.perf_counter()
                    # Create point maps with proper coordinate transforms
                    left_pm = geo.create_pm_in_ref_frame(
                        left_world_pc, left_valid, left_info["cam"], reference_cam, 
                        left_info["image"].shape[:2], pm_source="3d_tracks"
                    )
                    if self.time_debug:
                        print(f"[TIME] Create left_pm: {(time.perf_counter() - t0)*1000:.2f}ms")
                    
                    if self.time_debug:
                        t0 = time.perf_counter()
                    mid_pm = geo.create_pm_in_ref_frame(
                        mid_world_pc, mid_valid, mid_info["cam"], reference_cam, 
                        mid_info["image"].shape[:2], pm_source="3d_tracks"
                    )
                    if self.time_debug:
                        print(f"[TIME] Create mid_pm: {(time.perf_counter() - t0)*1000:.2f}ms")
                    
                    if self.time_debug:
                        t0 = time.perf_counter()
                    right_pm = geo.create_pm_in_ref_frame(
                        right_world_pc, right_valid, right_info["cam"], reference_cam, 
                        right_info["image"].shape[:2], pm_source="3d_tracks"
                    )
                    if self.time_debug:
                        print(f"[TIME] Create right_pm: {(time.perf_counter() - t0)*1000:.2f}ms")
                    
                elif pm_source == "dm":
                    if self.time_debug:
                        t0 = time.perf_counter()
                    # Create point maps from depth maps
                    left_pm = geo.dm_to_cam_pm(left_info["dm"], reference_cam)
                    if self.time_debug:
                        print(f"[TIME] Create left_pm from dm: {(time.perf_counter() - t0)*1000:.2f}ms")
                    
                    if self.time_debug:
                        t0 = time.perf_counter()
                    mid_pm = geo.create_pm_from_dm_in_ref_frame(
                        mid_info["dm"], mid_info["cam"], reference_cam
                    )
                    if self.time_debug:
                        print(f"[TIME] Create mid_pm from dm: {(time.perf_counter() - t0)*1000:.2f}ms")
                    
                    if self.time_debug:
                        t0 = time.perf_counter()
                    right_pm = geo.create_pm_from_dm_in_ref_frame(
                        right_info["dm"], right_info["cam"], reference_cam
                    )
                    if self.time_debug:
                        print(f"[TIME] Create right_pm from dm: {(time.perf_counter() - t0)*1000:.2f}ms")
                    
                else:
                    raise NotImplementedError("point map source not implemented")

                # Assert that point maps have valid points
                if self.time_debug:
                    t0 = time.perf_counter()
                assert np.sum(left_pm[:, :, 3]) >= min_valid_pc_points, f"left point map {sequence_path}:{left_frame} has {np.sum(left_pm[:, :, 3])} valid points, need at least {min_valid_pc_points}"
                assert np.sum(mid_pm[:, :, 3]) >= min_valid_pc_points, f"mid point map {sequence_path}:{mid_frame} has {np.sum(mid_pm[:, :, 3])} valid points, need at least {min_valid_pc_points}"
                assert np.sum(right_pm[:, :, 3]) >= min_valid_pc_points, f"right point map {sequence_path}:{right_frame} has {np.sum(right_pm[:, :, 3])} valid points, need at least {min_valid_pc_points}"
                if self.time_debug:
                    print(f"[TIME] Assert point maps: {(time.perf_counter() - t0)*1000:.2f}ms")

                # ---------------------------------------------------------------
                # Compute motion maps generically
                # ---------------------------------------------------------------
                
                # Calculate normalized temporal positions
                if self.time_debug:
                    t0 = time.perf_counter()
                mid_tq = (mid_frame - left_frame) / (right_frame - left_frame)
                
                # Query times list: [mid_tq, 1.0, 0.0]
                # Corresponding to: [position of mid, position of right from left, position of left from right]
                query_times = torch.tensor([mid_tq, 1.0, 0.0]).float()
                
                # Store all frame infos and point maps for generic motion computation
                frame_infos = [left_info, mid_info, right_info]
                view_infos = [left_info, right_info]  # Only the stereo views (0 and 1)
                if self.time_debug:
                    print(f"[TIME] Compute temporal positions: {(time.perf_counter() - t0)*1000:.2f}ms")
                
                # Compute motion maps for all required pairs
                # Format: motion_gt["i_j"] where i is source view index, j is query time
                # IMPORTANT: All motions must be expressed in the left camera (reference) frame
                motion_gt = {}
                
                # For query 0 (mid_tq): both views to mid position
                # l2m: left to mid (left camera is already reference)
                if self.time_debug:
                    t0 = time.perf_counter()
                motion_maps_0 = geo.get_motion_map_from_world_pc(
                    [left_info["world_pc_valid"], mid_info["world_pc_valid"]],
                    [left_info["cam"], mid_info["cam"]],
                    left_info["image"].shape[:2],
                )
                motion_gt["l2m"] = motion_maps_0[0]  # left to mid, in left frame
                if self.time_debug:
                    print(f"[TIME] Compute l2m motion: {(time.perf_counter() - t0)*1000:.2f}ms")
                
                # r2m: right to mid (ensure left camera is reference by putting it first)
                # We need a dummy left entry to maintain left as reference
                if self.time_debug:
                    t0 = time.perf_counter()
                motion_maps_1 = geo.get_motion_map_from_world_pc(
                    [left_info["world_pc_valid"], right_info["world_pc_valid"], mid_info["world_pc_valid"]],
                    [left_info["cam"], right_info["cam"], mid_info["cam"]],
                    right_info["image"].shape[:2],
                )
                motion_gt["r2m"] = motion_maps_1[1]  # right to mid (index 1), in left frame
                if self.time_debug:
                    print(f"[TIME] Compute r2m motion: {(time.perf_counter() - t0)*1000:.2f}ms")
                
                # For query 1 (1.0): view 0 to position 1 (left to right)
                # l2r: left to right (left camera is already reference)
                if self.time_debug:
                    t0 = time.perf_counter()
                motion_maps_2 = geo.get_motion_map_from_world_pc(
                    [left_info["world_pc_valid"], right_info["world_pc_valid"]],
                    [left_info["cam"], right_info["cam"]],
                    left_info["image"].shape[:2],
                )
                motion_gt["l2r"] = motion_maps_2[0]  # left to right, in left frame
                if self.time_debug:
                    print(f"[TIME] Compute l2r motion: {(time.perf_counter() - t0)*1000:.2f}ms")
                
                # For query 2 (0.0): view 1 to position 0 (right to left)
                # r2l: right to left (ensure left camera is reference)
                # We include left as dummy first entry to maintain it as reference
                if self.time_debug:
                    t0 = time.perf_counter()
                motion_maps_3 = geo.get_motion_map_from_world_pc(
                    [left_info["world_pc_valid"], right_info["world_pc_valid"], left_info["world_pc_valid"]],
                    [left_info["cam"], right_info["cam"], left_info["cam"]],
                    right_info["image"].shape[:2],
                )
                # Motion from right (index 1) to left (last frame)
                motion_gt["r2l"] = motion_maps_3[1]  # right to left, in left frame
                if self.time_debug:
                    print(f"[TIME] Compute r2l motion: {(time.perf_counter() - t0)*1000:.2f}ms")

                # Assert that motion maps have valid motion vectors
                if self.time_debug:
                    t0 = time.perf_counter()
                assert np.sum(motion_gt["l2m"][:, :, 3]) >= min_valid_mm_points, f"l2m motion map {sequence_path}:{left_frame}->{mid_frame} has {np.sum(motion_gt['l2m'][:, :, 3])} valid points, need at least {min_valid_mm_points}"
                assert np.sum(motion_gt["r2m"][:, :, 3]) >= min_valid_mm_points, f"r2m motion map {sequence_path}:{right_frame}->{mid_frame} has {np.sum(motion_gt['r2m'][:, :, 3])} valid points, need at least {min_valid_mm_points}"
                assert np.sum(motion_gt["l2r"][:, :, 3]) >= min_valid_mm_points, f"l2r motion map {sequence_path}:{left_frame}->{right_frame} has {np.sum(motion_gt['l2r'][:, :, 3])} valid points, need at least {min_valid_mm_points}"
                assert np.sum(motion_gt["r2l"][:, :, 3]) >= min_valid_mm_points, f"r2l motion map {sequence_path}:{right_frame}->{left_frame} has {np.sum(motion_gt['r2l'][:, :, 3])} valid points, need at least {min_valid_mm_points}"
                if self.time_debug:
                    print(f"[TIME] Assert motion maps: {(time.perf_counter() - t0)*1000:.2f}ms")

                break
            except Exception as e:
                import traceback
                
                print(f"error loading triplet: {seq_idx}:{sequence_path}, {left_frame}, {mid_frame}, {right_frame}")
                print(f"error: {e}")
                traceback.print_exc()
                
                index = np.random.randint(0, len(self.triplets))
                if self.time_debug:
                    print(f"[TIME] Exception handling loop iteration: {(time.perf_counter() - loop_start)*1000:.2f}ms")

        # Process images
        if self.time_debug:
            t0 = time.perf_counter()
        left_image, mid_image, right_image = self._process_images(left_info, mid_info, right_info)
        if self.time_debug:
            print(f"[TIME] Process images: {(time.perf_counter() - t0)*1000:.2f}ms")

        # Convert motion_gt dict to tensors
        if self.time_debug:
            t0 = time.perf_counter()
        motion_gt_tensors = {k: torch.from_numpy(v.copy()).float() for k, v in motion_gt.items()}
        if self.time_debug:
            print(f"[TIME] Convert motion_gt to tensors: {(time.perf_counter() - t0)*1000:.2f}ms")

        # Build return dictionary
        if self.time_debug:
            t0 = time.perf_counter()
        result = {
            "left_pm": torch.from_numpy(left_pm.copy()).float(),  # (H, W, 4)
            "mid_pm": torch.from_numpy(mid_pm.copy()).float(),  # (H, W, 4)
            "right_pm": torch.from_numpy(right_pm.copy()).float(),  # (H, W, 4)
            
            # Generic motion ground truths
            "motion_gt": motion_gt_tensors,  # Dict[str, Tensor] with keys "i_j"
            
            "left_image": torch.from_numpy(left_image.copy()).permute(2, 0, 1).float(),  # (3, H, W)
            "mid_image": torch.from_numpy(mid_image.copy()).permute(2, 0, 1).float(),  # (3, H, W)
            "right_image": torch.from_numpy(right_image.copy()).permute(2, 0, 1).float(),  # (3, H, W)
            
            "idxs": torch.tensor((left_frame, mid_frame, right_frame)).float(),  # tensor of shape (3,)
            "query_times": query_times,  # tensor of shape (3,) - [mid_tq, 1.0, 0.0]
            "sequence_idx": torch.tensor(seq_idx).float(),  # scalar tensor
            
            "left_instance": left_info.get("instance", None),  # string
            "right_instance": right_info.get("instance", None),  # string
            
            "cam": left_info["cam"],  # ((3,3), (4,4))
            "cam_mid": mid_info["cam"],  # ((3,3), (4,4))
            "cam_right": right_info["cam"],  # ((3,3), (4,4))
        }
        if self.time_debug:
            print(f"[TIME] Build return dict: {(time.perf_counter() - t0)*1000:.2f}ms")
            print(f"[TIME] TOTAL __getitem__: {(time.perf_counter() - total_start)*1000:.2f}ms")
            print("-" * 60)
        
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
        import loaders.utils.geometry as geo

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