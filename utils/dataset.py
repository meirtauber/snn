"""
Dataset loading and preprocessing utilities.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


def load_carla_data(data_path):
    """
    Load CARLA dataset from .npz file.

    Args:
        data_path (str): Path to carla_data.npz file

    Returns:
        dict: Dictionary containing 'rgb', 'depth', and 'velocity' arrays
    """
    data = np.load(data_path)
    return {"rgb": data["rgb"], "depth": data["depth"], "velocity": data["velocity"]}


class DepthDataset(Dataset):
    """
    PyTorch Dataset for depth estimation.

    Handles loading, normalization, and preprocessing of RGB-Depth pairs.

    Args:
        data_path (str): Path to carla_data.npz file
        max_depth (float): Maximum depth value for normalization (default: 100m)
        transform (callable, optional): Optional transform to apply to RGB images
    """

    def __init__(self, data_path, max_depth=100.0, transform=None):
        super().__init__()

        self.data_path = Path(data_path)
        self.max_depth = max_depth
        self.transform = transform

        # Load data
        print(f"Loading dataset from {self.data_path}...")
        data = load_carla_data(self.data_path)

        self.rgb_images = data["rgb"]  # (N, H, W, 3), uint8
        self.depth_images = data["depth"]  # (N, H, W), float32

        # Validate data
        assert self.rgb_images.shape[0] == self.depth_images.shape[0], (
            "RGB and depth arrays must have same number of samples"
        )

        self.num_samples = self.rgb_images.shape[0]
        print(f"Loaded {self.num_samples} samples")
        print(f"  RGB shape: {self.rgb_images.shape}")
        print(f"  Depth shape: {self.depth_images.shape}")

        # Compute depth statistics
        valid_mask = self.depth_images > 0
        self.depth_mean = self.depth_images[valid_mask].mean()
        self.depth_std = self.depth_images[valid_mask].std()
        self.depth_min = self.depth_images[valid_mask].min()
        self.depth_max = self.depth_images[valid_mask].max()

        print(
            f"  Depth stats: mean={self.depth_mean:.2f}m, std={self.depth_std:.2f}m, "
            f"range=[{self.depth_min:.2f}, {self.depth_max:.2f}]m"
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Get a single RGB-Depth pair.

        Args:
            idx (int): Sample index

        Returns:
            tuple: (rgb_tensor, depth_tensor)
                - rgb_tensor: (3, H, W) normalized to [0, 1]
                - depth_tensor: (1, H, W) normalized to [0, 1]
        """
        # Load RGB image
        rgb = self.rgb_images[idx]  # (H, W, 3), uint8

        # Load depth image
        depth = self.depth_images[idx]  # (H, W), float32

        # Normalize RGB to [0, 1] and convert to tensor
        rgb = torch.from_numpy(rgb).float() / 255.0
        rgb = rgb.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)

        # Normalize depth to [0, 1]
        depth = np.clip(depth, 0, self.max_depth)
        depth = depth / self.max_depth
        depth = torch.from_numpy(depth).float().unsqueeze(0)  # (H, W) -> (1, H, W)

        # Apply transforms if specified
        if self.transform is not None:
            rgb = self.transform(rgb)

        return rgb, depth

    def get_raw_sample(self, idx):
        """
        Get raw (unnormalized) sample for visualization.

        Args:
            idx (int): Sample index

        Returns:
            tuple: (rgb_array, depth_array)
        """
        return self.rgb_images[idx], self.depth_images[idx]


def split_dataset(dataset, train_ratio=0.8, seed=42):
    """
    Split dataset into training and validation sets.

    Args:
        dataset (Dataset): PyTorch dataset to split
        train_ratio (float): Ratio of training samples (default: 0.8)
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    from torch.utils.data import random_split

    # Set random seed
    generator = torch.Generator().manual_seed(seed)

    # Calculate sizes
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    # Split
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    print(f"\nDataset split: {train_size} train, {val_size} val")

    return train_dataset, val_dataset


if __name__ == "__main__":
    # Test dataset loading
    import sys
    from pathlib import Path

    # Look for sample data
    data_path = Path("data/run1/carla_data.npz")

    if not data_path.exists():
        print(f"No data found at {data_path}")
        print("Run collect_data.py first to generate training data")
        sys.exit(1)

    # Test dataset
    print("Testing DepthDataset...")
    dataset = DepthDataset(data_path)

    # Get sample
    rgb, depth = dataset[0]
    print(f"\nSample 0:")
    print(f"  RGB tensor: {rgb.shape}, range=[{rgb.min():.3f}, {rgb.max():.3f}]")
    print(
        f"  Depth tensor: {depth.shape}, range=[{depth.min():.3f}, {depth.max():.3f}]"
    )

    # Test split
    train_ds, val_ds = split_dataset(dataset, train_ratio=0.8)
    print(f"\nSplit: {len(train_ds)} train, {len(val_ds)} val")

    print("\nDataset test passed!")
