"""
KITTI Depth Prediction Benchmark Dataset Loader

Loads the official KITTI depth prediction benchmark with:
- 93K+ dense depth annotations (groundtruth/)
- Official train/val/test splits
- Temporal sequence support for SNN models

Directory structure expected:
    data_benchmark/
        ├── train/
        │   └── 2011_XX_YY/
        │       └── 2011_XX_YY_drive_ZZZZ_sync/
        │           └── proj_depth/
        │               └── groundtruth/image_02/*.png  # Dense GT
        ├── val/
        └── raw_kitti/
            └── 2011_XX_YY/
                └── 2011_XX_YY_drive_ZZZZ_sync/
                    └── image_02/data/*.png  # RGB images
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms


class KITTIBenchmarkDataset(Dataset):
    """
    KITTI Depth Prediction Benchmark Dataset for temporal depth estimation.

    Args:
        benchmark_root: Root directory of benchmark (contains train/val/raw_kitti/)
        split: 'train', 'val', or 'test'
        img_height: Target image height
        img_width: Target image width
        sequence_length: Number of frames in temporal sequence
        sequence_stride: Stride between consecutive frames
        augmentation_params: Dict of augmentation parameters (only for train)
    """

    def __init__(
        self,
        benchmark_root,
        split="train",
        img_height=384,
        img_width=1280,
        sequence_length=4,
        sequence_stride=1,
        augmentation_params=None,
    ):
        self.benchmark_root = Path(benchmark_root)
        self.split = split
        self.img_height = img_height
        self.img_width = img_width
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.augmentation_params = augmentation_params if split == "train" else None

        # Paths
        self.depth_root = self.benchmark_root / split
        self.rgb_root = self.benchmark_root / "raw_kitti"

        # Load samples
        self.samples = self._load_samples()

        print(f"[KITTI Benchmark] Loaded {len(self.samples)} {split} sequences")
        print(f"  Sequence length: {sequence_length}, stride: {sequence_stride}")

    def _load_samples(self):
        """Load all depth maps and create temporal sequences."""
        samples = []

        # Find all depth maps - try both possible structures
        depth_pattern1 = (
            f"*/*/proj_depth/groundtruth/image_02/*.png"  # train/date/drive/...
        )
        depth_pattern2 = f"*/proj_depth/groundtruth/image_02/*.png"  # train/drive/...

        depth_files = sorted(self.depth_root.glob(depth_pattern1))
        if len(depth_files) == 0:
            depth_files = sorted(self.depth_root.glob(depth_pattern2))

        if len(depth_files) == 0:
            raise ValueError(f"No depth maps found in {self.depth_root}")

        # Group by drive
        from collections import defaultdict

        drive_groups = defaultdict(list)

        for depth_path in depth_files:
            # Parse: train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000000.png
            # OR:    train/2011_09_26/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000000.png
            parts = depth_path.parts

            # Detect structure by checking if we have date subfolder
            if (
                len(parts) >= 7 and parts[-7].count("_") == 2
            ):  # Has date folder (2011_09_26)
                date = parts[-7]
                drive = parts[-6]
            else:  # No date folder
                drive = parts[-6]
                # Extract date from drive name (first 10 chars)
                date = "_".join(
                    drive.split("_")[:3]
                )  # Get "2011_09_26" from "2011_09_26_drive_0001_sync"

            frame_id = depth_path.stem

            drive_key = f"{date}/{drive}"
            drive_groups[drive_key].append(
                {
                    "depth_path": depth_path,
                    "frame_id": frame_id,
                    "date": date,
                    "drive": drive,
                }
            )

        # Create temporal sequences within each drive
        total_drives = len(drive_groups)
        drives_with_rgb = 0
        total_potential_sequences = 0

        for drive_key, frames in drive_groups.items():
            # Sort by frame_id
            frames = sorted(frames, key=lambda x: x["frame_id"])

            # Skip if not enough frames for a sequence
            if len(frames) < self.sequence_length:
                continue

            # Check if this drive has any RGB images
            test_frame = frames[0]
            test_rgb_path = (
                self.rgb_root
                / test_frame["date"]
                / test_frame["drive"]
                / "image_02"
                / "data"
                / f"{test_frame['frame_id']}.png"
            )

            if not test_rgb_path.parent.exists():
                # Try to find the RGB directory to debug
                if len(samples) == 0:  # Only print debug for first missing drive
                    print(f"[DEBUG] RGB path doesn't exist for drive {drive_key}")
                    print(f"[DEBUG] Looking for: {test_rgb_path}")
                    print(f"[DEBUG] RGB root: {self.rgb_root}")
                continue

            drives_with_rgb += 1
            potential_seqs = len(frames) - self.sequence_length + 1
            total_potential_sequences += potential_seqs

            # Create sliding window sequences
            for i in range(
                0, len(frames) - self.sequence_length + 1, self.sequence_stride
            ):
                sequence_frames = frames[i : i + self.sequence_length]

                # Verify RGB images exist for all frames
                rgb_paths = []
                all_exist = True

                for frame in sequence_frames:
                    rgb_path = (
                        self.rgb_root
                        / frame["date"]
                        / frame["drive"]
                        / "image_02"
                        / "data"
                        / f"{frame['frame_id']}.png"
                    )
                    if not rgb_path.exists():
                        all_exist = False
                        break
                    rgb_paths.append(rgb_path)

                if all_exist:
                    samples.append(
                        {
                            "rgb_paths": rgb_paths,
                            "depth_path": sequence_frames[-1][
                                "depth_path"
                            ],  # Use last frame's depth
                            "drive": drive_key,
                        }
                    )

        print(f"[DEBUG] Total drives with depth: {total_drives}")
        print(f"[DEBUG] Drives with RGB images: {drives_with_rgb}")
        print(f"[DEBUG] Total potential sequences: {total_potential_sequences}")
        print(f"[DEBUG] Valid sequences (with all RGB): {len(samples)}")

        return samples

    def __len__(self):
        return len(self.samples)

    def _load_depth_png(self, path):
        """Load depth from PNG (KITTI format: depth = float(value) / 256.0)."""
        depth_png = np.array(Image.open(path), dtype=np.float32)
        # KITTI depth is stored as uint16 where depth_meters = float(depth_png) / 256.0
        depth = depth_png / 256.0
        return depth

    def _load_calibration(self, date):
        """Load camera calibration."""
        calib_file = self.rgb_root / date / "calib_cam_to_cam.txt"

        if not calib_file.exists():
            # Return default intrinsics for KITTI camera 2
            # Approximate values
            fx = 721.5377
            fy = 721.5377
            cx = 609.5593
            cy = 172.854

            K = np.array(
                [
                    [fx, 0, cx, 0],
                    [0, fy, cy, 0],
                    [0, 0, 1, 0],
                ],
                dtype=np.float32,
            )
            return K

        # Parse calibration file
        with open(calib_file, "r") as f:
            lines = f.readlines()

        # Find P_rect_02 line
        for line in lines:
            if line.startswith("P_rect_02:"):
                values = line.split()[1:]
                P = np.array([float(v) for v in values], dtype=np.float32).reshape(3, 4)
                return P

        raise ValueError(f"P_rect_02 not found in {calib_file}")

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load RGB sequence
        rgb_sequence = []
        for rgb_path in sample["rgb_paths"]:
            rgb = Image.open(rgb_path).convert("RGB")
            rgb = rgb.resize((self.img_width, self.img_height), Image.LANCZOS)
            rgb_sequence.append(rgb)

        # Load depth (last frame)
        depth = self._load_depth_png(sample["depth_path"])

        # Resize depth to match resized RGB
        depth_pil = Image.fromarray(depth.astype(np.float32))
        depth_pil = depth_pil.resize((self.img_width, self.img_height), Image.NEAREST)
        depth = np.array(depth_pil, dtype=np.float32)

        # Load calibration
        date = sample["drive"].split("/")[0]
        K = self._load_calibration(date)

        # Adjust intrinsics for resized image
        original_width, original_height = 1242, 375  # KITTI image_02 original size
        scale_x = self.img_width / original_width
        scale_y = self.img_height / original_height

        K_resized = K.copy()
        K_resized[0, :] *= scale_x  # fx, cx
        K_resized[1, :] *= scale_y  # fy, cy

        # Apply augmentation (synchronized across sequence)
        if self.augmentation_params:
            # Random horizontal flip
            if np.random.rand() < self.augmentation_params.get("hflip_p", 0.0):
                rgb_sequence = [
                    rgb.transpose(Image.FLIP_LEFT_RIGHT) for rgb in rgb_sequence
                ]
                depth = np.fliplr(depth)
                # Adjust cx for flip
                K_resized[0, 2] = self.img_width - K_resized[0, 2]

            # Color jitter (same for all frames)
            brightness = self.augmentation_params.get("brightness", (1.0, 1.0))
            contrast = self.augmentation_params.get("contrast", (1.0, 1.0))
            saturation = self.augmentation_params.get("saturation", (1.0, 1.0))
            hue = self.augmentation_params.get("hue", (0.0, 0.0))

            color_jitter = transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )

            # Apply same jitter to all frames
            rgb_sequence = [color_jitter(rgb) for rgb in rgb_sequence]

        # Convert to tensors
        to_tensor = transforms.ToTensor()
        rgb_tensors = [to_tensor(rgb) for rgb in rgb_sequence]

        # Stack sequence: (T, 3, H, W)
        sequence_tensor = torch.stack(rgb_tensors, dim=0)

        # Depth tensor: (1, H, W)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0)

        # Intrinsics tensor: (3, 4)
        K_tensor = torch.from_numpy(K_resized)

        return {
            "image": sequence_tensor,  # (T, 3, H, W)
            "depth": depth_tensor,  # (1, H, W)
            "intrinsics": K_tensor,  # (3, 4)
        }


if __name__ == "__main__":
    # Test the dataset
    import sys

    if len(sys.argv) < 2:
        print("Usage: python kitti_benchmark_dataset.py /path/to/data_benchmark")
        sys.exit(1)

    benchmark_root = sys.argv[1]

    print("Testing KITTI Benchmark Dataset...")
    print(f"Benchmark root: {benchmark_root}")
    print()

    # Test train split
    dataset = KITTIBenchmarkDataset(
        benchmark_root=benchmark_root,
        split="train",
        img_height=384,
        img_width=1280,
        sequence_length=4,
        sequence_stride=1,
    )

    print(f"\nDataset size: {len(dataset)}")

    if len(dataset) > 0:
        print("\nTesting first sample...")
        sample = dataset[0]

        print(f"  Image shape: {sample['image'].shape}")  # Should be (4, 3, 384, 1280)
        print(f"  Depth shape: {sample['depth'].shape}")  # Should be (1, 384, 1280)
        print(f"  Intrinsics shape: {sample['intrinsics'].shape}")  # Should be (3, 4)
        print(
            f"  Depth range: {sample['depth'].min():.2f} - {sample['depth'].max():.2f}m"
        )
        print(
            f"  Valid depth pixels: {(sample['depth'] > 0).sum().item()} / {sample['depth'].numel()}"
        )

        print("\n✓ Dataset test passed!")
    else:
        print("\n✗ No samples found!")
