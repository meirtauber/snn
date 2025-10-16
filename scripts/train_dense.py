import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

import os
import argparse
import time
from pathlib import Path

# --- Make the model and losses visible to this script ---
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.dense_temporal_snn_depth import DenseTemporalSNNDepth
from utils.losses import CompositeLoss


# --- 1. Placeholder CARLA Dataset ---
# This class needs to be implemented to load your specific CARLA data.
# It should return a sequence of 4 frames and a dense depth map for the last frame.
class CARLADataset(Dataset):
    def __init__(self, root_dir, split="train", num_frames=4, resize=(224, 224)):
        """
        Args:
            root_dir (str): Directory with all the data.
            split (str): 'train' or 'val'.
            num_frames (int): Number of frames in a sequence.
            resize (tuple): The target size for images and depth maps, must match the model's backbone.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.resize = resize

        # --- This is placeholder logic. Replace with your actual data loading. ---
        # For example, you might have a list of directories, where each directory
        # contains a sequence of images and their corresponding depth maps.
        self.sequences = [
            f"sequence_{i:04d}" for i in range(100 if split == "train" else 20)
        ]
        print(f"Loaded {len(self.sequences)} placeholder sequences for {split} split.")
        # --------------------------------------------------------------------------

    def __len__(self):
        # This should return the total number of sequences in the dataset.
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Should return:
            - A tensor of shape (num_frames, 3, H, W) for the image sequence.
            - A tensor of shape (1, H, W) for the final depth map.
        """
        sequence_id = self.sequences[idx]

        # --- Placeholder data generation. Replace with your data loading. ---
        # Simulate loading a 4-frame sequence of 224x224 images
        image_sequence = torch.randn(self.num_frames, 3, self.resize[0], self.resize[1])

        # Simulate loading a 224x224 depth map for the last frame
        # Depth values are typically in meters.
        depth_map = torch.rand(1, self.resize[0], self.resize[1]) * 80.0
        # ---------------------------------------------------------------------

        return image_sequence, depth_map


# --- 2. Training and Validation Functions ---


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device):
    """Performs one training epoch."""
    model.train()
    total_loss = 0
    start_time = time.time()

    for i, (sequence, target_depth) in enumerate(loader):
        sequence, target_depth = sequence.to(device), target_depth.to(device)

        optimizer.zero_grad()

        # Use autocast for mixed precision
        with autocast():
            prediction = model(sequence)
            loss = loss_fn(prediction, target_depth)

        # Scale loss and backpropagate
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f"  Batch {i + 1}/{len(loader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    epoch_time = time.time() - start_time
    print(f"--> Train Epoch Summary: Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
    return avg_loss


def validate(model, loader, loss_fn, device):
    """Performs validation on the validation set."""
    model.eval()
    total_loss = 0
    start_time = time.time()

    with torch.no_grad():
        for i, (sequence, target_depth) in enumerate(loader):
            sequence, target_depth = sequence.to(device), target_depth.to(device)

            # Autocast is still recommended for validation for consistency
            with autocast():
                prediction = model(sequence)
                loss = loss_fn(prediction, target_depth)

            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    epoch_time = time.time() - start_time
    print(f"--> Validation Summary: Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
    return avg_loss


# --- 3. Main Training Orchestrator ---


def main(args):
    """Main function to run the training."""

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Datasets and Dataloaders ---
    # NOTE: Replace 'path/to/your/carla/data' with the actual path
    train_dataset = CARLADataset(root_dir="path/to/your/carla/data", split="train")
    val_dataset = CARLADataset(root_dir="path/to/your/carla/data", split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- Model, Loss, and Optimizer ---
    print("Instantiating model...")
    model = DenseTemporalSNNDepth().to(device)

    loss_fn = CompositeLoss(alpha_ssim=args.ssim_weight, beta_silog=1.0)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-2
    )

    # Smart learning rate scheduling
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # GradScaler for Automatic Mixed Precision (AMP)
    scaler = GradScaler()

    print(
        f"Model, Loss, and Optimizer initialized. Starting training for {args.epochs} epochs."
    )

    # --- Training Loop ---
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        print(f"Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device
        )
        val_loss = validate(model, val_loader, loss_fn, device)

        # Update learning rate
        scheduler.step()

        # Save checkpoint if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint_path = output_dir / "best_model.pth"
            print(
                f"Validation loss improved to {val_loss:.4f}. Saving model to {checkpoint_path}"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss,
                },
                checkpoint_path,
            )

    print("\n--- Training Complete ---")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to {output_dir / 'best_model.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dense Temporal SNN Model")

    # --- Key arguments ---
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (adjust based on your VRAM)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Initial learning rate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--ssim-weight",
        type=float,
        default=0.85,
        help="Weight for the SSIM component of the loss",
    )

    args = parser.parse_args()

    main(args)
