"""
CARLA Diverse Dataset Loader for Dense Temporal SNN Training.

This module provides efficient loading of the diverse CARLA dataset collected
with multiple weather conditions and camera configurations.

Dataset Structure:
    - data/diverse_20k/batch_001.npz to batch_040.npz
    - Each batch: 500 frames of 640x480 RGB + depth
    - Total: ~20,000 frames (40 batches × 500 frames)
    - Collected at 20 FPS with diverse conditions

Usage:
    from utils.carla_diverse_dataset import CARLADiverseDataset

    train_dataset = CARLADiverseDataset(
        data_dir='data/diverse_20k',
        split='train',
        num_frames=4,
        resize=(224, 224)
    )
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional
import json


class CARLADiverseDataset(Dataset):
    """
    PyTorch Dataset for CARLA diverse dataset with temporal sequences.

    Loads data from multiple .npz batch files and creates temporal sequences
    of consecutive frames for training the Dense Temporal SNN model.

    Args:
        data_dir (str): Path to directory containing batch_*.npz files
        split (str): 'train' or 'val' or 'all'
        num_frames (int): Number of consecutive frames in each sequence (default: 4)
        resize (tuple): Target size (H, W) for images (default: (224, 224))
        stride (int): Stride for creating sequences (default: 1)
        min_depth (float): Minimum depth value in meters (default: 0.1)
        max_depth (float): Maximum depth value in meters (default: 80.0)
        normalize_rgb (bool): Whether to apply ImageNet normalization (default: True)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        num_frames: int = 4,
        resize: Tuple[int, int] = (224, 224),
        stride: int = 1,
        min_depth: float = 0.1,
        max_depth: float = 80.0,
        normalize_rgb: bool = True,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.split = split
        self.num_frames = num_frames
        self.resize = resize
        self.stride = stride
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.normalize_rgb = normalize_rgb

        # ImageNet normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # Load metadata
        metadata_file = self.data_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None
            print(f"Warning: metadata.json not found in {self.data_dir}")

        # Find all batch files
        batch_files = sorted(self.data_dir.glob("batch_*.npz"))
        if len(batch_files) == 0:
            raise FileNotFoundError(f"No batch_*.npz files found in {self.data_dir}")

        print(f"\n{'=' * 60}")
        print(f"Loading CARLA Diverse Dataset: {split.upper()} split")
        print(f"{'=' * 60}")
        print(f"Data directory: {self.data_dir}")
        print(f"Total batch files found: {len(batch_files)}")

        # Split batches into train/val
        # Strategy: First 80% of batches for training, last 20% for validation
        # This maintains temporal coherence within scenes
        num_batches = len(batch_files)
        train_split_idx = int(num_batches * 0.8)

        if split == "train":
            self.batch_files = batch_files[:train_split_idx]
        elif split == "val":
            self.batch_files = batch_files[train_split_idx:]
        elif split == "all":
            self.batch_files = batch_files
        else:
            raise ValueError(
                f"Invalid split '{split}'. Must be 'train', 'val', or 'all'"
            )

        print(f"Batches for {split}: {len(self.batch_files)}")

        # Build index: (batch_idx, frame_start_idx) for each valid sequence
        self.sequence_index = []
        self._build_sequence_index()

        print(f"Total sequences created: {len(self.sequence_index)}")
        print(f"Sequence length: {num_frames} frames")
        print(f"Stride: {stride}")
        print(f"Target resolution: {resize}")
        print(f"Depth range: [{min_depth}, {max_depth}] meters")
        print(f"{'=' * 60}\n")

        # Cache for loaded batches (memory vs speed tradeoff)
        self._batch_cache = {}
        self._cache_size = 4  # Keep 4 batches in memory at a time

    def _build_sequence_index(self):
        """
        Build index of all valid sequences.

        Each sequence is identified by (batch_idx, start_frame_idx).
        We create sequences within each batch to maintain temporal coherence.
        """
        frames_per_batch = 500  # All batches have 500 frames

        for batch_idx, batch_file in enumerate(self.batch_files):
            # Calculate number of valid sequences in this batch
            # A sequence needs num_frames consecutive frames
            num_sequences_in_batch = (
                frames_per_batch - self.num_frames
            ) // self.stride + 1

            for seq_idx in range(num_sequences_in_batch):
                start_frame = seq_idx * self.stride
                self.sequence_index.append((batch_idx, start_frame))

    def _load_batch(self, batch_idx: int):
        """
        Load a batch file into memory with caching.

        Args:
            batch_idx: Index into self.batch_files

        Returns:
            dict with 'rgb' and 'depth' numpy arrays
        """
        # Check cache first
        if batch_idx in self._batch_cache:
            return self._batch_cache[batch_idx]

        # Load from disk
        batch_file = self.batch_files[batch_idx]
        data = np.load(batch_file)
        batch_data = {
            "rgb": data["rgb"],  # (500, 480, 640, 3) uint8
            "depth": data["depth"],  # (500, 480, 640) float32
        }

        # Update cache (simple FIFO)
        if len(self._batch_cache) >= self._cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._batch_cache))
            del self._batch_cache[oldest_key]

        self._batch_cache[batch_idx] = batch_data

        return batch_data

    def _resize_and_preprocess(
        self, rgb_sequence: np.ndarray, depth_frame: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Resize and preprocess RGB sequence and depth frame.

        Args:
            rgb_sequence: (T, H, W, 3) numpy array, uint8
            depth_frame: (H, W) numpy array, float32

        Returns:
            rgb_tensor: (T, 3, H_new, W_new) normalized float32 tensor
            depth_tensor: (1, H_new, W_new) float32 tensor
        """
        import torch.nn.functional as F

        T, H, W, C = rgb_sequence.shape
        H_new, W_new = self.resize

        # Process RGB sequence
        rgb_list = []
        for t in range(T):
            # Convert to tensor and normalize to [0, 1]
            rgb = torch.from_numpy(rgb_sequence[t]).float() / 255.0
            # HWC -> CHW
            rgb = rgb.permute(2, 0, 1)
            # Resize
            rgb = F.interpolate(
                rgb.unsqueeze(0),
                size=(H_new, W_new),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            # Apply ImageNet normalization
            if self.normalize_rgb:
                rgb = (rgb - self.mean) / self.std
            rgb_list.append(rgb)

        rgb_tensor = torch.stack(rgb_list, dim=0)  # (T, 3, H_new, W_new)

        # Process depth frame
        depth = torch.from_numpy(depth_frame).float()
        # Clip to valid depth range
        depth = torch.clamp(depth, self.min_depth, self.max_depth)
        # Add channel dimension: (H, W) -> (1, H, W)
        depth = depth.unsqueeze(0)
        # Resize
        depth = F.interpolate(
            depth.unsqueeze(0),
            size=(H_new, W_new),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # (1, H_new, W_new)

        return rgb_tensor, depth

    def __len__(self) -> int:
        """Return total number of sequences."""
        return len(self.sequence_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a temporal sequence and corresponding depth map.

        Args:
            idx: Sequence index

        Returns:
            rgb_sequence: (T, 3, H, W) tensor - T consecutive RGB frames
            depth_target: (1, H, W) tensor - Depth for the last frame (frame t)
        """
        batch_idx, start_frame = self.sequence_index[idx]

        # Load batch data
        batch_data = self._load_batch(batch_idx)

        # Extract sequence
        end_frame = start_frame + self.num_frames
        rgb_sequence = batch_data["rgb"][start_frame:end_frame]  # (T, 480, 640, 3)

        # Target depth is for the last frame (current frame 't')
        depth_target = batch_data["depth"][end_frame - 1]  # (480, 640)

        # Resize and preprocess
        rgb_tensor, depth_tensor = self._resize_and_preprocess(
            rgb_sequence, depth_target
        )

        return rgb_tensor, depth_tensor

    def get_raw_sequence(self, idx: int) -> dict:
        """
        Get raw (unprocessed) sequence for visualization.

        Args:
            idx: Sequence index

        Returns:
            dict with 'rgb' (T, H, W, 3) and 'depth' (H, W)
        """
        batch_idx, start_frame = self.sequence_index[idx]
        batch_data = self._load_batch(batch_idx)

        end_frame = start_frame + self.num_frames
        rgb_sequence = batch_data["rgb"][start_frame:end_frame]
        depth_target = batch_data["depth"][end_frame - 1]

        return {
            "rgb": rgb_sequence,
            "depth": depth_target,
            "batch_idx": batch_idx,
            "start_frame": start_frame,
        }


def test_dataset():
    """Test the dataset loader."""
    import time

    print("\n" + "=" * 60)
    print("TESTING CARLA DIVERSE DATASET LOADER")
    print("=" * 60)

    # Test train split
    train_dataset = CARLADiverseDataset(
        data_dir="data/diverse_20k", split="train", num_frames=4, resize=(224, 224)
    )

    # Test val split
    val_dataset = CARLADiverseDataset(
        data_dir="data/diverse_20k", split="val", num_frames=4, resize=(224, 224)
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} sequences")
    print(f"  Val:   {len(val_dataset)} sequences")

    # Test loading a sample
    print(f"\nLoading sample from train dataset...")
    start_time = time.time()
    rgb_seq, depth = train_dataset[0]
    load_time = time.time() - start_time

    print(f"  Load time: {load_time * 1000:.1f} ms")
    print(f"  RGB sequence shape: {rgb_seq.shape}")
    print(f"  Depth shape: {depth.shape}")
    print(f"  RGB range: [{rgb_seq.min():.3f}, {rgb_seq.max():.3f}]")
    print(f"  Depth range: [{depth.min():.3f}, {depth.max():.3f}] meters")

    # Test loading speed with cache
    print(f"\nTesting cache performance (loading same sample again)...")
    start_time = time.time()
    rgb_seq2, depth2 = train_dataset[0]
    cache_time = time.time() - start_time
    print(f"  Cached load time: {cache_time * 1000:.1f} ms")
    print(f"  Speedup: {load_time / cache_time:.1f}x")

    # Test batch loading
    print(f"\nTesting batch loading...")
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True
    )

    start_time = time.time()
    batch_rgb, batch_depth = next(iter(train_loader))
    batch_time = time.time() - start_time

    print(f"  Batch load time: {batch_time * 1000:.1f} ms")
    print(f"  Batch RGB shape: {batch_rgb.shape}")  # (B, T, 3, H, W)
    print(f"  Batch depth shape: {batch_depth.shape}")  # (B, 1, H, W)

    print(f"\n{'=' * 60}")
    print("DATASET LOADER TEST PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_dataset()
