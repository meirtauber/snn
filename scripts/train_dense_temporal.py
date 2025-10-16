"""
Training script for Dense Temporal SNN Depth Estimation.

This script trains the DenseTemporalSNNDepth model on the CARLA diverse dataset
with 4-frame temporal sequences.

Architecture:
    - Swin Transformer spatial encoder (per-frame feature extraction)
    - SNN temporal fusion (sequential frame processing)
    - Multi-scale U-Net decoder with skip connections
    - Composite loss (SILog + SSIM)

Usage:
    python scripts/train_dense_temporal.py --epochs 50 --batch-size 4
    python scripts/train_dense_temporal.py --epochs 100 --batch-size 8 --lr 5e-5
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

import os
import argparse
import time
import json
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.dense_temporal_snn_depth import DenseTemporalSNNDepth
from utils.losses import CompositeLoss
from utils.kitti_dataset import KittiDataset
from utils.metrics import compute_depth_metrics, MetricsTracker


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(
    model, loader, optimizer, loss_fn, scaler, device, epoch, clip_grad=1.0
):
    """
    Train for one epoch.

    Args:
        model: The neural network model
        loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        scaler: GradScaler for mixed precision
        device: Device to train on
        epoch: Current epoch number
        clip_grad: Gradient clipping max norm

    Returns:
        dict: Training metrics (loss, mae, abs_rel, etc.)
    """
    model.train()

    total_loss = 0
    metrics_tracker = MetricsTracker()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, batch_data in enumerate(pbar):
        # Move to device
        sequence = batch_data["image"].to(device)  # (B, T, 3, H, W)
        target_depth = batch_data["depth"].to(device)  # (B, 1, H, W)
        intrinsics = batch_data["intrinsics"].to(device)  # (B, 4, 4)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast():
            prediction = model(sequence, intrinsics)  # (B, 1, H, W)
            loss = loss_fn(prediction, target_depth)

        # Skip batch if loss is not finite
        if not torch.isfinite(loss):
            print(f"\nWarning: Skipping batch {batch_idx} - loss is {loss.item()}")
            continue

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Gradient clipping (important for SNN training stability)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        scaler.step(optimizer)
        scaler.update()

        # Track metrics
        total_loss += loss.item()

        with torch.no_grad():
            metrics = compute_depth_metrics(
                prediction, target_depth, already_in_meters=True
            )
            metrics_tracker.update(metrics)

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "mae": f"{metrics.mae:.3f}",
                "abs_rel": f"{metrics.abs_rel:.3f}",
            }
        )

    # Compute averages
    avg_loss = total_loss / len(loader)
    avg_metrics = metrics_tracker.get_average()

    return {
        "loss": avg_loss,
        "mae": avg_metrics.mae,
        "rmse": avg_metrics.rmse,
        "abs_rel": avg_metrics.abs_rel,
        "sq_rel": avg_metrics.sq_rel,
        "delta_1": avg_metrics.delta_1,
        "delta_2": avg_metrics.delta_2,
        "delta_3": avg_metrics.delta_3,
    }


def validate(model, loader, loss_fn, device, epoch):
    """
    Validate on validation set.

    Args:
        model: The neural network model
        loader: Validation data loader
        loss_fn: Loss function
        device: Device to validate on
        epoch: Current epoch number

    Returns:
        dict: Validation metrics
    """
    model.eval()

    total_loss = 0
    metrics_tracker = MetricsTracker()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(pbar):
            sequence = batch_data["image"].to(device)
            target_depth = batch_data["depth"].to(device)
            intrinsics = batch_data["intrinsics"].to(device)

            # Forward pass with mixed precision
            with autocast():
                prediction = model(sequence, intrinsics)
                loss = loss_fn(prediction, target_depth)

            # Track metrics
            total_loss += loss.item()
            metrics = compute_depth_metrics(
                prediction, target_depth, already_in_meters=True
            )
            metrics_tracker.update(metrics)

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "mae": f"{metrics.mae:.3f}",
                    "abs_rel": f"{metrics.abs_rel:.3f}",
                }
            )

    # Compute averages
    avg_loss = total_loss / len(loader)
    avg_metrics = metrics_tracker.get_average()

    return {
        "loss": avg_loss,
        "mae": avg_metrics.mae,
        "rmse": avg_metrics.rmse,
        "abs_rel": avg_metrics.abs_rel,
        "sq_rel": avg_metrics.sq_rel,
        "delta_1": avg_metrics.delta_1,
        "delta_2": avg_metrics.delta_2,
        "delta_3": avg_metrics.delta_3,
    }


def save_checkpoint(
    model, optimizer, epoch, val_loss, output_dir, is_best=False, history=None
):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "history": history,
    }

    if is_best:
        checkpoint_path = output_dir / "best_model.pth"
        print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
    else:
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
        print(f"  ✓ Saved checkpoint (epoch {epoch})")

    torch.save(checkpoint, checkpoint_path)


def save_training_history(history, output_dir):
    """Save training history to JSON."""
    history_file = output_dir / "training_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  ✓ Saved training history to {history_file}")


def main(args):
    """Main training function."""

    print("\n" + "=" * 70)
    print("DENSE TEMPORAL SNN DEPTH ESTIMATION - TRAINING")
    print("=" * 70)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load datasets
    print(f"\n{'=' * 70}")
    print("LOADING DATASETS")
    print(f"{'=' * 70}")

    # Define augmentation parameters
    augmentation_params = {
        "brightness": (args.brightness_min, args.brightness_max),
        "contrast": (args.contrast_min, args.contrast_max),
        "saturation": (args.saturation_min, args.saturation_max),
        "hue": (args.hue_min, args.hue_max),
        "grayscale_p": args.grayscale_p,
        "hflip_p": args.hflip_p,
    }

    train_dataset = KittiDataset(
        kitti_root_dir=args.kitti_root_dir,
        processed_depth_dir=args.processed_depth_dir,
        split="train",
        img_height=args.img_height,
        img_width=args.img_width,
        augmentation_params=augmentation_params,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
    )

    val_dataset = KittiDataset(
        kitti_root_dir=args.kitti_root_dir,
        processed_depth_dir=args.processed_depth_dir,
        split="val",
        img_height=args.img_height,
        img_width=args.img_width,
        augmentation_params=None,  # No augmentation for validation
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Create model
    print(f"\n{'=' * 70}")
    print("MODEL ARCHITECTURE")
    print(f"{'=' * 70}")

    model = DenseTemporalSNNDepth(
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        img_width=args.img_width,
        img_height=args.img_height,
        # sequence_length=4, # Assuming T=4 frames for temporal SNN
    ).to(device)

    num_params = count_parameters(model)
    print(f"Total parameters: {num_params:,}")
    print(f"Model size: ~{num_params * 4 / 1024 / 1024:.1f} MB (FP32)")

    # Create loss function
    loss_fn = CompositeLoss(
        alpha_ssim=args.ssim_weight, beta_silog=args.silog_weight, variance_focus=0.85
    )

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    # Mixed precision scaler
    scaler = GradScaler()

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": [],
        "train_abs_rel": [],
        "val_abs_rel": [],
        "train_delta_1": [],
        "val_delta_1": [],
        "learning_rate": [],
    }

    # Print training configuration
    print(f"\n{'=' * 70}")
    print("TRAINING CONFIGURATION")
    print(f"{'=' * 70}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Min LR: {args.min_lr}")
    print(f"Gradient clipping: {args.clip_grad}")
    print(f"SSIM weight: {args.ssim_weight}")
    print(f"SILog weight: {args.silog_weight}")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Val batches per epoch: {len(val_loader)}")

    # Start training
    print(f"\n{'=' * 70}")
    print("TRAINING START")
    print(f"{'=' * 70}\n")

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Train
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            scaler,
            device,
            epoch,
            args.clip_grad,
        )

        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device, epoch)

        # Update scheduler
        scheduler.step()

        # Update history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_mae"].append(train_metrics["mae"])
        history["val_mae"].append(val_metrics["mae"])
        history["train_abs_rel"].append(train_metrics["abs_rel"])
        history["val_abs_rel"].append(val_metrics["abs_rel"])
        history["train_delta_1"].append(train_metrics["delta_1"])
        history["val_delta_1"].append(val_metrics["delta_1"])
        history["learning_rate"].append(current_lr)

        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(
            f"\nEpoch {epoch}/{args.epochs} Summary (LR: {current_lr:.6f}, Time: {epoch_time:.1f}s):"
        )
        print(
            f"  Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.3f}, "
            f"Abs Rel: {train_metrics['abs_rel']:.3f}, δ<1.25: {train_metrics['delta_1']:.3f}"
        )
        print(
            f"  Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.3f}, "
            f"Abs Rel: {val_metrics['abs_rel']:.3f}, δ<1.25: {val_metrics['delta_1']:.3f}"
        )

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics["loss"],
                output_dir,
                is_best=True,
                history=history,
            )

        # Save periodic checkpoint
        if epoch % args.save_freq == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics["loss"],
                output_dir,
                is_best=False,
                history=history,
            )

        print("-" * 70)

    # Training complete
    total_time = time.time() - start_time

    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total time: {total_time / 3600:.2f} hours")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")

    # Save final history
    save_training_history(history, output_dir)

    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Dense Temporal SNN for Depth Estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--kitti-root-dir",
        type=str,
        required=True,
        help="Root directory of the downloaded KITTI raw dataset",
    )
    parser.add_argument(
        "--processed-depth-dir",
        type=str,
        required=True,
        help="Directory containing preprocessed KITTI depth maps",
    )
    parser.add_argument(
        "--img-height", type=int, default=384, help="Image height for model input"
    )
    parser.add_argument(
        "--img-width", type=int, default=1280, help="Image width for model input"
    )
    parser.add_argument(
        "--min-depth", type=float, default=0.1, help="Minimum depth value for the model"
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=80.0,
        help="Maximum depth value for the model",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=4,
        help="Number of frames in each temporal sequence",
    )
    parser.add_argument(
        "--sequence-stride",
        type=int,
        default=1,
        help="Stride between consecutive sequences (1 = overlapping, >1 = skip frames)",
    )

    # Augmentation arguments
    parser.add_argument(
        "--brightness-min",
        type=float,
        default=0.8,
        help="Min brightness for color jitter",
    )
    parser.add_argument(
        "--brightness-max",
        type=float,
        default=1.2,
        help="Max brightness for color jitter",
    )
    parser.add_argument(
        "--contrast-min", type=float, default=0.8, help="Min contrast for color jitter"
    )
    parser.add_argument(
        "--contrast-max", type=float, default=1.2, help="Max contrast for color jitter"
    )
    parser.add_argument(
        "--saturation-min",
        type=float,
        default=0.8,
        help="Min saturation for color jitter",
    )
    parser.add_argument(
        "--saturation-max",
        type=float,
        default=1.2,
        help="Max saturation for color jitter",
    )
    parser.add_argument(
        "--hue-min", type=float, default=-0.1, help="Min hue for color jitter"
    )
    parser.add_argument(
        "--hue-max", type=float, default=0.1, help="Max hue for color jitter"
    )
    parser.add_argument(
        "--grayscale-p",
        type=float,
        default=0.1,
        help="Probability of grayscale augmentation",
    )
    parser.add_argument(
        "--hflip-p",
        type=float,
        default=0.5,
        help="Probability of horizontal flip augmentation",
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size (reduce if OOM)"
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--min-lr", type=float, default=1e-6, help="Minimum learning rate for scheduler"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-2, help="Weight decay for AdamW"
    )
    parser.add_argument(
        "--clip-grad", type=float, default=1.0, help="Gradient clipping max norm"
    )

    # Loss weights
    parser.add_argument(
        "--ssim-weight", type=float, default=0.85, help="Weight for SSIM loss component"
    )
    parser.add_argument(
        "--silog-weight",
        type=float,
        default=1.0,
        help="Weight for SILog loss component",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/dense_temporal_snn",
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument(
        "--save-freq", type=int, default=10, help="Save checkpoint every N epochs"
    )

    # System arguments
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Run training
    main(args)
