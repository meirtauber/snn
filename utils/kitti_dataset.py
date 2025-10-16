import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob


def read_calib_file(filepath):
    """Reads a calibration file and returns a dictionary of matrices."""
    data = {}
    with open(filepath, "r") as f:
        for line in f.readlines():
            if (
                ":" not in line
            ):  # Skip lines that don't contain a colon (e.g., comments)
                continue
            key, value = line.split(":", 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass  # skip non-numeric lines
    return data


class KittiDataset(Dataset):
    def __init__(
        self,
        kitti_root_dir,
        processed_depth_dir,
        split,
        img_height=384,
        img_width=1280,
        augmentation_params=None,
        sequence_length=4,  # Number of frames per temporal sequence
        sequence_stride=1,  # Stride between consecutive sequences
    ):
        super(KittiDataset, self).__init__()
        self.kitti_root_dir = kitti_root_dir
        self.processed_depth_dir = processed_depth_dir
        self.split = split
        self.img_height = img_height
        self.img_width = img_width
        self.augmentation_params = augmentation_params
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride

        self.samples = self._load_samples()
        self.to_tensor = transforms.ToTensor()

        # Define transformations for data augmentation
        self.transform_rgb = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.transform_no_aug_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _load_samples(self):
        """
        Loads paths to RGB images and their corresponding depth maps and calibration files.
        Creates temporal sequences of length `sequence_length`.

        Assumes the following structure:
        kitti_root_dir/
            date/
                date_drive_sync/
                    image_02/data/
                        0000000000.png
                    calib_cam_to_cam.txt
        processed_depth_dir/
            date/
                date_drive_sync/
                    depth_maps/
                        0000000000.npy

        Returns a list of samples where each sample is a temporal sequence.
        """
        sequences = []

        # Find all 'date_drive_sync' directories in the kitti_root_dir
        # First, search for drive_sync directories directly under kitti_root_dir
        # E.g., kitti_root_dir/2011_09_26_drive_0000_sync
        direct_drive_dirs = sorted(
            glob.glob(os.path.join(self.kitti_root_dir, "*_sync"))
        )

        # Second, search for drive_sync directories within date folders under kitti_root_dir
        # E.g., kitti_root_dir/2011_09_26/2011_09_26_drive_0000_sync
        nested_drive_dirs = sorted(
            glob.glob(os.path.join(self.kitti_root_dir, "*/", "*_sync"))
        )

        all_drive_dirs = direct_drive_dirs + nested_drive_dirs
        # Remove duplicates, if any drive is found by both patterns (unlikely with KITTI structure)
        drive_dirs = sorted(list(set(all_drive_dirs)))

        if not drive_dirs:
            print(
                f"No drive directories found in {self.kitti_root_dir} (or its subdirectories). Please check the path and structure."
            )
            return []

        for drive_dir in drive_dirs:
            date_drive_name = os.path.basename(
                drive_dir
            )  # e.g., 2011_09_26_drive_0001_sync

            # Extract date_name (e.g., "2011_09_26" from "2011_09_26_drive_0001_sync")
            date_name = date_drive_name.split("_drive")[0]

            # Determine the base directory for calibration files (usually the date folder)
            calib_base_dir = os.path.join(self.kitti_root_dir, date_name)

            # Paths for RGB images, depth maps, and calibration
            # Assuming RGB images are within the drive_dir itself
            rgb_image_dir = os.path.join(drive_dir, "image_02", "data")

            # Assuming depth maps are in the processed_depth_dir, mirroring the structure
            processed_date_dir = os.path.join(self.processed_depth_dir, date_name)
            processed_drive_dir = os.path.join(processed_date_dir, date_drive_name)
            depth_map_dir = os.path.join(processed_drive_dir, "depth_maps")

            calib_filepath = os.path.join(calib_base_dir, "calib_cam_to_cam.txt")

            if not (
                os.path.exists(rgb_image_dir)
                and os.path.exists(depth_map_dir)
                and os.path.exists(calib_filepath)
            ):
                print(
                    f"Skipping {drive_dir}: Missing RGB, depth, or calibration files."
                )
                continue

            # Load camera intrinsics once per sequence
            calib_data = read_calib_file(calib_filepath)

            # Validate P_rect_02 matrix has correct number of elements
            # KITTI uses "P_rect_02" (not "P2") for the left color camera projection matrix
            if "P_rect_02" not in calib_data or len(calib_data["P_rect_02"]) != 12:
                print(
                    f"Skipping {drive_dir}: Invalid P_rect_02 calibration matrix (expected 12 elements)."
                )
                continue

            # Extract 3x4 projection matrix
            P_rect_02 = calib_data["P_rect_02"].reshape(3, 4)
            K = torch.from_numpy(P_rect_02).float()

            rgb_files = sorted(glob.glob(os.path.join(rgb_image_dir, "*.png")))
            depth_files = sorted(glob.glob(os.path.join(depth_map_dir, "*.npy")))

            if len(rgb_files) != len(depth_files) or len(rgb_files) == 0:
                print(
                    f"Skipping {drive_dir}: Mismatch in number of RGB images and depth maps or no files found."
                )
                continue

            # Create temporal sequences from this drive
            # We need at least `sequence_length` frames to create one sequence
            num_frames = len(rgb_files)
            if num_frames < self.sequence_length:
                print(
                    f"Skipping {drive_dir}: Only {num_frames} frames, need at least {self.sequence_length}."
                )
                continue

            # Create sequences with sliding window
            for start_idx in range(
                0, num_frames - self.sequence_length + 1, self.sequence_stride
            ):
                end_idx = start_idx + self.sequence_length

                sequence_sample = {
                    "rgb_paths": rgb_files[start_idx:end_idx],
                    "depth_path": depth_files[
                        end_idx - 1
                    ],  # Use last frame's depth as target
                    "intrinsics": K,
                }
                sequences.append(sequence_sample)

        # Split data into train/val (80/20 split)
        num_sequences = len(sequences)
        split_idx = int(0.8 * num_sequences)

        if self.split == "train":
            return sequences[:split_idx]
        elif self.split == "val":
            return sequences[split_idx:]
        else:
            return sequences

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load temporal sequence of RGB images
        rgb_sequence = []
        for rgb_path in sample["rgb_paths"]:
            rgb_image = Image.open(rgb_path).convert("RGB")
            rgb_image = rgb_image.resize(
                (self.img_width, self.img_height), Image.LANCZOS
            )
            rgb_sequence.append(rgb_image)

        # Load depth map (target is the last frame's depth)
        depth_map = np.load(sample["depth_path"])
        depth_map = Image.fromarray(depth_map).resize(
            (self.img_width, self.img_height), Image.NEAREST
        )
        depth_map = np.array(depth_map, dtype=np.float32)

        # Apply transformations (augmentation for RGB, sync augmentation for depth)
        # Important: Apply same random augmentation to all frames in sequence
        if self.split == "train" and self.augmentation_params:
            # Decide on augmentation parameters once for the entire sequence
            do_hflip = np.random.rand() < self.augmentation_params.get("hflip_p", 0.5)

            color_jitter = transforms.ColorJitter(
                brightness=self.augmentation_params.get("brightness", (0.8, 1.2)),
                contrast=self.augmentation_params.get("contrast", (0.8, 1.2)),
                saturation=self.augmentation_params.get("saturation", (0.8, 1.2)),
                hue=self.augmentation_params.get("hue", (-0.1, 0.1)),
            )

            grayscale = transforms.RandomGrayscale(
                p=self.augmentation_params.get("grayscale_p", 0.1)
            )

            # Apply same augmentation to all frames
            augmented_sequence = []
            for frame in rgb_sequence:
                frame = color_jitter(frame)
                frame = grayscale(frame)
                if do_hflip:
                    frame = transforms.functional.hflip(frame)
                augmented_sequence.append(frame)
            rgb_sequence = augmented_sequence

            # Also flip depth if we flipped images
            if do_hflip:
                depth_map = np.fliplr(depth_map).copy()

        # Convert RGB sequence to tensors and normalize
        rgb_tensors = []
        for frame in rgb_sequence:
            frame_tensor = self.transform_no_aug_rgb(frame)
            rgb_tensors.append(frame_tensor)

        # Stack into (T, C, H, W) tensor
        sequence_tensor = torch.stack(rgb_tensors, dim=0)

        # Convert depth to tensor
        depth_map = torch.from_numpy(depth_map).unsqueeze(0)  # (1, H, W)

        # Adjust intrinsics for resizing
        # Get original image dimensions from first frame
        original_width, original_height = Image.open(sample["rgb_paths"][0]).size
        fx_scale = self.img_width / original_width
        fy_scale = self.img_height / original_height

        # Create a new intrinsic matrix for the resized image
        K_resized = sample["intrinsics"].clone()
        K_resized[0, 0] *= fx_scale  # fx
        K_resized[1, 1] *= fy_scale  # fy
        K_resized[0, 2] *= fx_scale  # cx
        K_resized[1, 2] *= fy_scale  # cy

        return {
            "image": sequence_tensor,  # (T, 3, H, W)
            "depth": depth_map,  # (1, H, W)
            "intrinsics": K_resized,  # (3, 4)
            "filename": os.path.basename(sample["rgb_paths"][-1]),
        }
