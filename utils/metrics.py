"""
Depth estimation evaluation metrics.

Implements standard metrics from the computer vision literature:
- Absolute Relative Error (abs_rel)
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Threshold Accuracy (delta < 1.25, 1.25^2, 1.25^3)
"""

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class DepthMetrics:
    """Container for depth estimation metrics."""

    abs_rel: float
    sq_rel: float
    mae: float
    rmse: float
    delta_1: float
    delta_2: float
    delta_3: float

    def __str__(self):
        return (
            f"abs_rel: {self.abs_rel:.4f}, "
            f"sq_rel: {self.sq_rel:.4f}, "
            f"MAE: {self.mae:.4f}, "
            f"RMSE: {self.rmse:.4f}, "
            f"δ<1.25: {self.delta_1:.4f}, "
            f"δ<1.25²: {self.delta_2:.4f}, "
            f"δ<1.25³: {self.delta_3:.4f}"
        )

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "abs_rel": self.abs_rel,
            "sq_rel": self.sq_rel,
            "mae": self.mae,
            "rmse": self.rmse,
            "delta_1": self.delta_1,
            "delta_2": self.delta_2,
            "delta_3": self.delta_3,
        }


def compute_depth_metrics(
    pred, target, max_depth=100.0, eps=1e-6, already_in_meters=False
):
    """
    Compute depth estimation metrics.

    Args:
        pred (torch.Tensor): Predicted depth (B, 1, H, W) or (B, H, W)
        target (torch.Tensor): Ground truth depth (B, 1, H, W) or (B, H, W)
        max_depth (float): Maximum depth value for denormalization (default: 100m)
        eps (float): Small value to avoid division by zero
        already_in_meters (bool): If True, pred and target are already in meters (no denormalization)

    Returns:
        DepthMetrics: Object containing all metrics
    """
    # Ensure same shape
    if pred.dim() == 4 and pred.size(1) == 1:
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)

    # Denormalize to meters if needed
    if not already_in_meters:
        pred = pred * max_depth
        target = target * max_depth

    # Create valid mask (depth > 0)
    valid_mask = target > eps

    if valid_mask.sum() == 0:
        # No valid pixels
        return DepthMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Extract valid pixels
    pred_valid = pred[valid_mask]
    target_valid = target[valid_mask]

    # Absolute Relative Error
    abs_rel = torch.mean(torch.abs(pred_valid - target_valid) / (target_valid + eps))

    # Squared Relative Error
    sq_rel = torch.mean(((pred_valid - target_valid) ** 2) / (target_valid + eps))

    # Mean Absolute Error (meters)
    mae = torch.mean(torch.abs(pred_valid - target_valid))

    # Root Mean Square Error (meters)
    rmse = torch.sqrt(torch.mean((pred_valid - target_valid) ** 2))

    # Threshold Accuracy (δ < threshold)
    # max(pred/target, target/pred) < threshold
    ratio = torch.max(
        pred_valid / (target_valid + eps), target_valid / (pred_valid + eps)
    )

    delta_1 = torch.mean((ratio < 1.25).float())
    delta_2 = torch.mean((ratio < 1.25**2).float())
    delta_3 = torch.mean((ratio < 1.25**3).float())

    return DepthMetrics(
        abs_rel=abs_rel.item(),
        sq_rel=sq_rel.item(),
        mae=mae.item(),
        rmse=rmse.item(),
        delta_1=delta_1.item(),
        delta_2=delta_2.item(),
        delta_3=delta_3.item(),
    )


def compute_batch_metrics(pred, target, max_depth=100.0):
    """
    Compute metrics for a batch and return averaged values.

    Args:
        pred (torch.Tensor): Predicted depth (B, 1, H, W)
        target (torch.Tensor): Ground truth depth (B, 1, H, W)
        max_depth (float): Maximum depth value

    Returns:
        DepthMetrics: Averaged metrics across the batch
    """
    batch_size = pred.size(0)

    # Compute metrics for each sample
    metrics_list = []
    for i in range(batch_size):
        metrics = compute_depth_metrics(
            pred[i : i + 1], target[i : i + 1], max_depth=max_depth
        )
        metrics_list.append(metrics)

    # Average across batch
    avg_metrics = DepthMetrics(
        abs_rel=np.mean([m.abs_rel for m in metrics_list]),
        sq_rel=np.mean([m.sq_rel for m in metrics_list]),
        mae=np.mean([m.mae for m in metrics_list]),
        rmse=np.mean([m.rmse for m in metrics_list]),
        delta_1=np.mean([m.delta_1 for m in metrics_list]),
        delta_2=np.mean([m.delta_2 for m in metrics_list]),
        delta_3=np.mean([m.delta_3 for m in metrics_list]),
    )

    return avg_metrics


class MetricsTracker:
    """
    Tracks metrics across multiple batches.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all accumulated metrics."""
        self.abs_rel_sum = 0.0
        self.sq_rel_sum = 0.0
        self.mae_sum = 0.0
        self.rmse_sum = 0.0
        self.delta_1_sum = 0.0
        self.delta_2_sum = 0.0
        self.delta_3_sum = 0.0
        self.count = 0

    def update(self, metrics):
        """
        Update tracker with new metrics.

        Args:
            metrics (DepthMetrics): Metrics to add
        """
        self.abs_rel_sum += metrics.abs_rel
        self.sq_rel_sum += metrics.sq_rel
        self.mae_sum += metrics.mae
        self.rmse_sum += metrics.rmse
        self.delta_1_sum += metrics.delta_1
        self.delta_2_sum += metrics.delta_2
        self.delta_3_sum += metrics.delta_3
        self.count += 1

    def get_average(self):
        """
        Get average metrics.

        Returns:
            DepthMetrics: Averaged metrics
        """
        if self.count == 0:
            return DepthMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        return DepthMetrics(
            abs_rel=self.abs_rel_sum / self.count,
            sq_rel=self.sq_rel_sum / self.count,
            mae=self.mae_sum / self.count,
            rmse=self.rmse_sum / self.count,
            delta_1=self.delta_1_sum / self.count,
            delta_2=self.delta_2_sum / self.count,
            delta_3=self.delta_3_sum / self.count,
        )


if __name__ == "__main__":
    # Test metrics computation
    print("Testing depth metrics...")

    # Create dummy data
    batch_size = 4
    height, width = 480, 640

    # Simulate predictions and targets
    pred = torch.rand(batch_size, 1, height, width)  # [0, 1]
    target = torch.rand(batch_size, 1, height, width)  # [0, 1]

    # Compute metrics
    metrics = compute_depth_metrics(pred, target, max_depth=100.0)
    print(f"\nMetrics: {metrics}")

    # Test tracker
    print("\nTesting MetricsTracker...")
    tracker = MetricsTracker()

    for i in range(5):
        pred = torch.rand(batch_size, 1, height, width)
        target = torch.rand(batch_size, 1, height, width)
        metrics = compute_depth_metrics(pred, target)
        tracker.update(metrics)

    avg_metrics = tracker.get_average()
    print(f"Average over 5 batches: {avg_metrics}")

    print("\nMetrics test passed!")
