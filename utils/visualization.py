"""
Visualization utilities for depth estimation.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from pathlib import Path


def visualize_predictions(rgb, depth_true, depth_pred, max_depth=100.0, save_path=None):
    """
    Create a visualization of RGB, ground truth depth, and predicted depth.

    Args:
        rgb (np.ndarray or torch.Tensor): RGB image (H, W, 3) or (3, H, W)
        depth_true (np.ndarray or torch.Tensor): Ground truth depth (H, W) or (1, H, W)
        depth_pred (np.ndarray or torch.Tensor): Predicted depth (H, W) or (1, H, W)
        max_depth (float): Maximum depth value for visualization
        save_path (str, optional): Path to save the visualization

    Returns:
        np.ndarray: Combined visualization image
    """
    # Convert tensors to numpy
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()
    if isinstance(depth_true, torch.Tensor):
        depth_true = depth_true.cpu().numpy()
    if isinstance(depth_pred, torch.Tensor):
        depth_pred = depth_pred.cpu().numpy()

    # Reshape if needed
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = rgb.transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
    if depth_true.ndim == 3 and depth_true.shape[0] == 1:
        depth_true = depth_true.squeeze(0)  # (1, H, W) -> (H, W)
    if depth_pred.ndim == 3 and depth_pred.shape[0] == 1:
        depth_pred = depth_pred.squeeze(0)  # (1, H, W) -> (H, W)

    # Denormalize if normalized
    if rgb.max() <= 1.0:
        rgb = (rgb * 255).astype(np.uint8)
    if depth_true.max() <= 1.0:
        depth_true = depth_true * max_depth
    if depth_pred.max() <= 1.0:
        depth_pred = depth_pred * max_depth

    # Create depth colormaps
    depth_true_viz = create_depth_colormap(depth_true, max_depth)
    depth_pred_viz = create_depth_colormap(depth_pred, max_depth)

    # Compute error map
    error = np.abs(depth_pred - depth_true)
    error_viz = create_error_colormap(error, max_error=10.0)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB Input")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(depth_true_viz)
    axes[0, 1].set_title("Ground Truth Depth")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(depth_pred_viz)
    axes[1, 0].set_title("Predicted Depth")
    axes[1, 0].axis("off")

    im = axes[1, 1].imshow(error_viz)
    axes[1, 1].set_title("Absolute Error")
    axes[1, 1].axis("off")

    # Add colorbar for error
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Error (m)")

    plt.tight_layout()

    # Save or show
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    # Return combined image for programmatic use
    return combine_images([rgb, depth_true_viz, depth_pred_viz, error_viz])


def create_depth_colormap(depth, max_depth, colormap=cv2.COLORMAP_JET):
    """
    Create a color-mapped depth visualization.

    Args:
        depth (np.ndarray): Depth map (H, W)
        max_depth (float): Maximum depth value
        colormap (int): OpenCV colormap

    Returns:
        np.ndarray: RGB colormap image (H, W, 3)
    """
    # Clip and normalize to [0, 255]
    depth_clipped = np.clip(depth, 0, max_depth)
    depth_normalized = (depth_clipped / max_depth * 255).astype(np.uint8)

    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_normalized, colormap)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

    return depth_colored


def create_error_colormap(error, max_error=10.0):
    """
    Create a color-mapped error visualization.

    Args:
        error (np.ndarray): Error map (H, W)
        max_error (float): Maximum error value for visualization

    Returns:
        np.ndarray: RGB colormap image (H, W, 3)
    """
    # Clip and normalize
    error_clipped = np.clip(error, 0, max_error)
    error_normalized = (error_clipped / max_error * 255).astype(np.uint8)

    # Apply hot colormap
    error_colored = cv2.applyColorMap(error_normalized, cv2.COLORMAP_HOT)
    error_colored = cv2.cvtColor(error_colored, cv2.COLOR_BGR2RGB)

    return error_colored


def combine_images(images, layout="horizontal"):
    """
    Combine multiple images into a single image.

    Args:
        images (list): List of images to combine
        layout (str): 'horizontal' or 'vertical'

    Returns:
        np.ndarray: Combined image
    """
    if layout == "horizontal":
        return np.hstack(images)
    elif layout == "vertical":
        return np.vstack(images)
    else:
        raise ValueError(f"Unknown layout: {layout}")


def create_comparison_plot(snn_metrics, cnn_metrics, save_path=None):
    """
    Create a bar chart comparing SNN and CNN metrics.

    Args:
        snn_metrics (DepthMetrics): SNN metrics
        cnn_metrics (DepthMetrics): CNN metrics
        save_path (str, optional): Path to save the plot
    """
    metrics_names = ["abs_rel", "MAE (m)", "RMSE (m)", "δ<1.25", "δ<1.25²", "δ<1.25³"]

    snn_values = [
        snn_metrics.abs_rel,
        snn_metrics.mae,
        snn_metrics.rmse,
        snn_metrics.delta_1,
        snn_metrics.delta_2,
        snn_metrics.delta_3,
    ]

    cnn_values = [
        cnn_metrics.abs_rel,
        cnn_metrics.mae,
        cnn_metrics.rmse,
        cnn_metrics.delta_1,
        cnn_metrics.delta_2,
        cnn_metrics.delta_3,
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width / 2, snn_values, width, label="SNN", color="steelblue")
    ax.bar(x + width / 2, cnn_values, width, label="CNN", color="coral")

    ax.set_xlabel("Metrics")
    ax.set_ylabel("Value")
    ax.set_title("SNN vs CNN Depth Estimation Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def save_depth_predictions(rgb, depth_true, depth_pred, output_dir, prefix="sample"):
    """
    Save individual depth prediction images.

    Args:
        rgb (np.ndarray): RGB image
        depth_true (np.ndarray): Ground truth depth
        depth_pred (np.ndarray): Predicted depth
        output_dir (str): Output directory
        prefix (str): Filename prefix
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert and save RGB
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()
    if rgb.shape[0] == 3:
        rgb = rgb.transpose(1, 2, 0)
    if rgb.max() <= 1.0:
        rgb = (rgb * 255).astype(np.uint8)

    cv2.imwrite(
        str(output_path / f"{prefix}_rgb.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    )

    # Convert and save depth maps
    for name, depth in [("true", depth_true), ("pred", depth_pred)]:
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()
        if depth.ndim == 3:
            depth = depth.squeeze(0)

        # Denormalize if needed
        if depth.max() <= 1.0:
            depth = depth * 100.0

        # Create colormap
        depth_viz = create_depth_colormap(depth, max_depth=100.0)

        cv2.imwrite(
            str(output_path / f"{prefix}_depth_{name}.jpg"),
            cv2.cvtColor(depth_viz, cv2.COLOR_RGB2BGR),
        )


def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics over time.

    Args:
        history (dict): Dictionary containing training history
            Keys: 'train_loss', 'val_loss', 'train_mae', 'val_mae', etc.
        save_path (str, optional): Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Loss
    axes[0, 0].plot(history["train_loss"], label="Train")
    axes[0, 0].plot(history["val_loss"], label="Val")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training and Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # MAE
    axes[0, 1].plot(history["train_mae"], label="Train")
    axes[0, 1].plot(history["val_mae"], label="Val")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("MAE (m)")
    axes[0, 1].set_title("Mean Absolute Error")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Absolute Relative Error
    axes[1, 0].plot(history["train_abs_rel"], label="Train")
    axes[1, 0].plot(history["val_abs_rel"], label="Val")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Abs Rel")
    axes[1, 0].set_title("Absolute Relative Error")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Delta < 1.25
    axes[1, 1].plot(history["train_delta_1"], label="Train")
    axes[1, 1].plot(history["val_delta_1"], label="Val")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("δ<1.25")
    axes[1, 1].set_title("Threshold Accuracy")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization utilities...")

    # Create dummy data
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth_true = np.random.rand(480, 640) * 50  # 0-50m
    depth_pred = depth_true + np.random.randn(480, 640) * 5  # Add noise

    # Test depth colormap
    depth_viz = create_depth_colormap(depth_true, max_depth=100.0)
    print(f"Depth colormap shape: {depth_viz.shape}")

    # Test error colormap
    error = np.abs(depth_pred - depth_true)
    error_viz = create_error_colormap(error, max_error=10.0)
    print(f"Error colormap shape: {error_viz.shape}")

    print("\nVisualization test passed!")
