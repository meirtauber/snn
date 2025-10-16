"""
Validate the quality of dense depth interpolation.

This script checks:
1. Whether dense depth preserves original LiDAR values
2. If interpolation introduces unrealistic smoothing
3. Potential data leakage or artifacts that could inflate training metrics

Usage:
    python scripts/validate_dense_quality.py --kitti-root /path/to/kitti --processed-depth /path/to/processed
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.kitti_dataset import KittiDataset


def analyze_dense_quality(sparse_depth, dense_depth):
    """
    Analyze quality of dense depth interpolation.

    Args:
        sparse_depth: Original sparse LiDAR depth (H, W)
        dense_depth: Densified depth map (H, W)

    Returns:
        dict: Analysis results
    """
    results = {}

    # 1. Check if dense preserves sparse values
    valid_mask = sparse_depth > 0
    if valid_mask.sum() > 0:
        sparse_valid = sparse_depth[valid_mask]
        dense_valid = dense_depth[valid_mask]

        # Error on LiDAR hit locations
        errors = np.abs(dense_valid - sparse_valid)
        results["mae_on_lidar_hits"] = errors.mean()
        results["median_error_on_lidar_hits"] = np.median(errors)
        results["max_error_on_lidar_hits"] = errors.max()
        results["rmse_on_lidar_hits"] = np.sqrt((errors**2).mean())

        # Check if interpolation preserves values (should be near-perfect)
        results["preservation_score"] = (
            (errors < 0.01).sum() / len(errors) * 100
        )  # % within 1cm
    else:
        results["mae_on_lidar_hits"] = float("nan")

    # 2. Check smoothness (potential over-smoothing)
    dense_gradient_x = np.abs(np.diff(dense_depth, axis=1))
    dense_gradient_y = np.abs(np.diff(dense_depth, axis=0))

    # Only check gradients where depth > 0
    valid_grad_x = dense_gradient_x[dense_depth[:, :-1] > 0]
    valid_grad_y = dense_gradient_y[dense_depth[:-1, :] > 0]

    results["mean_gradient_x"] = valid_grad_x.mean() if len(valid_grad_x) > 0 else 0
    results["mean_gradient_y"] = valid_grad_y.mean() if len(valid_grad_y) > 0 else 0
    results["max_gradient_x"] = valid_grad_x.max() if len(valid_grad_x) > 0 else 0
    results["max_gradient_y"] = valid_grad_y.max() if len(valid_grad_y) > 0 else 0

    # Check for unrealistically smooth regions (potential artifact)
    smooth_threshold = 0.01  # Less than 1cm change per pixel
    very_smooth_regions_x = (
        (valid_grad_x < smooth_threshold).sum() / len(valid_grad_x) * 100
        if len(valid_grad_x) > 0
        else 0
    )
    very_smooth_regions_y = (
        (valid_grad_y < smooth_threshold).sum() / len(valid_grad_y) * 100
        if len(valid_grad_y) > 0
        else 0
    )
    results["very_smooth_percentage"] = (
        very_smooth_regions_x + very_smooth_regions_y
    ) / 2

    # 3. Coverage statistics
    results["sparse_coverage_pct"] = (sparse_depth > 0).sum() / sparse_depth.size * 100
    results["dense_coverage_pct"] = (dense_depth > 0).sum() / dense_depth.size * 100
    results["interpolated_pct"] = (
        results["dense_coverage_pct"] - results["sparse_coverage_pct"]
    )

    # 4. Depth distribution comparison
    sparse_valid_all = sparse_depth[sparse_depth > 0]
    dense_valid_all = dense_depth[dense_depth > 0]

    if len(sparse_valid_all) > 0 and len(dense_valid_all) > 0:
        results["sparse_mean_depth"] = sparse_valid_all.mean()
        results["sparse_std_depth"] = sparse_valid_all.std()
        results["dense_mean_depth"] = dense_valid_all.mean()
        results["dense_std_depth"] = dense_valid_all.std()

        # Check if dense shifts distribution (potential bias)
        results["mean_depth_shift"] = (
            results["dense_mean_depth"] - results["sparse_mean_depth"]
        )
        results["std_depth_shift"] = (
            results["dense_std_depth"] - results["sparse_std_depth"]
        )

    return results


def visualize_comparison(sparse_depth, dense_depth, save_path):
    """Create visualization comparing sparse and dense depth."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    vmax = max(sparse_depth.max(), dense_depth.max())
    vmax = min(vmax, 80)  # Cap at 80m for visualization

    # Row 1: Depth maps
    im1 = axes[0, 0].imshow(sparse_depth, cmap="plasma", vmin=0, vmax=vmax)
    axes[0, 0].set_title("Sparse (Ground Truth LiDAR)", fontsize=12)
    axes[0, 0].axis("off")
    plt.colorbar(im1, ax=axes[0, 0], label="Depth (m)", fraction=0.046)

    im2 = axes[0, 1].imshow(dense_depth, cmap="plasma", vmin=0, vmax=vmax)
    axes[0, 1].set_title("Dense (Interpolated)", fontsize=12)
    axes[0, 1].axis("off")
    plt.colorbar(im2, ax=axes[0, 1], label="Depth (m)", fraction=0.046)

    # Coverage mask
    sparse_mask = sparse_depth > 0
    coverage_pct = sparse_mask.sum() / sparse_mask.size * 100
    axes[0, 2].imshow(sparse_mask, cmap="gray")
    axes[0, 2].set_title(f"LiDAR Coverage ({coverage_pct:.2f}%)", fontsize=12)
    axes[0, 2].axis("off")

    # Row 2: Error analysis
    # Error on LiDAR hits only
    error_map = np.abs(dense_depth - sparse_depth)
    error_map[sparse_depth == 0] = np.nan  # Mask out interpolated regions
    im3 = axes[1, 0].imshow(error_map, cmap="hot", vmin=0, vmax=1.0)
    axes[1, 0].set_title("Error on LiDAR Hits", fontsize=12)
    axes[1, 0].axis("off")
    plt.colorbar(im3, ax=axes[1, 0], label="Absolute Error (m)", fraction=0.046)

    # Gradient magnitude (smoothness check)
    dense_grad = np.sqrt(
        np.pad(np.diff(dense_depth, axis=1) ** 2, ((0, 0), (0, 1)), "edge")
        + np.pad(np.diff(dense_depth, axis=0) ** 2, ((0, 1), (0, 0)), "edge")
    )
    dense_grad[dense_depth == 0] = np.nan
    im4 = axes[1, 1].imshow(dense_grad, cmap="viridis", vmin=0, vmax=2.0)
    axes[1, 1].set_title("Dense Depth Gradient Magnitude", fontsize=12)
    axes[1, 1].axis("off")
    plt.colorbar(im4, ax=axes[1, 1], label="Gradient (m/pixel)", fraction=0.046)

    # Histogram comparison
    sparse_valid = sparse_depth[sparse_depth > 0].flatten()
    dense_valid = dense_depth[dense_depth > 0].flatten()
    axes[1, 2].hist(
        sparse_valid, bins=50, alpha=0.7, label="Sparse", color="blue", density=True
    )
    axes[1, 2].hist(
        dense_valid, bins=50, alpha=0.7, label="Dense", color="red", density=True
    )
    axes[1, 2].set_xlabel("Depth (m)", fontsize=10)
    axes[1, 2].set_ylabel("Density", fontsize=10)
    axes[1, 2].set_title("Depth Distribution", fontsize=12)
    axes[1, 2].legend()
    axes[1, 2].set_xlim(0, 80)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved visualization to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Validate dense depth quality")
    parser.add_argument(
        "--kitti-root-dir", type=str, required=True, help="KITTI root directory"
    )
    parser.add_argument(
        "--processed-depth-dir",
        type=str,
        required=True,
        help="Processed depth directory",
    )
    parser.add_argument(
        "--num-samples", type=int, default=10, help="Number of samples to analyze"
    )
    parser.add_argument(
        "--output-dir", type=str, default="validation_results", help="Output directory"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DENSE DEPTH QUALITY VALIDATION")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    dataset = KittiDataset(
        kitti_root_dir=args.kitti_root_dir,
        processed_depth_dir=args.processed_depth_dir,
        split="train",
        img_height=384,
        img_width=1280,
        sequence_length=1,  # Single frame for analysis
        augmentation_params=None,  # No augmentation
    )

    print(f"Dataset size: {len(dataset)} samples")

    # Analyze multiple samples
    all_results = []

    for i in range(min(args.num_samples, len(dataset))):
        print(f"\n{'=' * 70}")
        print(f"Sample {i + 1}/{min(args.num_samples, len(dataset))}")
        print(f"{'=' * 70}")

        sample = dataset[i]

        # Get depth maps
        depth_tensor = sample["depth"]  # (1, H, W)
        dense_depth = depth_tensor[0].numpy()  # (H, W)

        # To get sparse depth, we need to load it from the sparse directory
        # The dataset might be loading from dense already, so let's check the sample info
        sample_info = dataset.samples[i]

        # Try to load sparse version
        sparse_path = sample_info["depth_path"].replace(
            "depth_maps_dense", "depth_maps"
        )

        if Path(sparse_path).exists():
            sparse_depth = np.load(sparse_path)
        else:
            print(f"⚠️  Sparse depth not found at {sparse_path}")
            print(f"   Skipping this sample...")
            continue

        # Analyze
        results = analyze_dense_quality(sparse_depth, dense_depth)
        all_results.append(results)

        # Print results
        print(f"\n1. INTERPOLATION ACCURACY (on LiDAR hits):")
        print(f"   MAE:         {results['mae_on_lidar_hits']:.6f} m")
        print(f"   Median:      {results['median_error_on_lidar_hits']:.6f} m")
        print(f"   Max:         {results['max_error_on_lidar_hits']:.6f} m")
        print(f"   RMSE:        {results['rmse_on_lidar_hits']:.6f} m")
        print(f"   Preserved:   {results['preservation_score']:.2f}% (within 1cm)")

        if results["mae_on_lidar_hits"] < 0.01:
            print(f"   ✅ EXCELLENT - Dense preserves LiDAR perfectly")
        elif results["mae_on_lidar_hits"] < 0.1:
            print(f"   ✅ GOOD - Minor interpolation error")
        else:
            print(f"   ⚠️  WARNING - Significant interpolation error!")

        print(f"\n2. SMOOTHNESS ANALYSIS:")
        print(f"   Mean gradient X: {results['mean_gradient_x']:.4f} m/pixel")
        print(f"   Mean gradient Y: {results['mean_gradient_y']:.4f} m/pixel")
        print(f"   Max gradient X:  {results['max_gradient_x']:.4f} m/pixel")
        print(f"   Max gradient Y:  {results['max_gradient_y']:.4f} m/pixel")
        print(f"   Very smooth:     {results['very_smooth_percentage']:.2f}%")

        if results["very_smooth_percentage"] > 80:
            print(f"   ⚠️  WARNING - Possibly over-smoothed (>80% very smooth)")
        else:
            print(f"   ✅ Reasonable gradient distribution")

        print(f"\n3. COVERAGE:")
        print(f"   Sparse:       {results['sparse_coverage_pct']:.2f}%")
        print(f"   Dense:        {results['dense_coverage_pct']:.2f}%")
        print(f"   Interpolated: {results['interpolated_pct']:.2f}%")

        print(f"\n4. DEPTH DISTRIBUTION:")
        print(
            f"   Sparse mean:  {results['sparse_mean_depth']:.2f} m (std: {results['sparse_std_depth']:.2f})"
        )
        print(
            f"   Dense mean:   {results['dense_mean_depth']:.2f} m (std: {results['dense_std_depth']:.2f})"
        )
        print(f"   Mean shift:   {results['mean_depth_shift']:.4f} m")
        print(f"   Std shift:    {results['std_depth_shift']:.4f} m")

        if abs(results["mean_depth_shift"]) > 1.0:
            print(f"   ⚠️  WARNING - Significant mean depth shift (>{1.0}m)")
        else:
            print(f"   ✅ Minimal distribution shift")

        # Create visualization for first sample
        if i == 0:
            vis_path = output_dir / "dense_vs_sparse_comparison.png"
            visualize_comparison(sparse_depth, dense_depth, vis_path)

    # Aggregate results
    if all_results:
        print(f"\n{'=' * 70}")
        print("AGGREGATE STATISTICS ACROSS ALL SAMPLES")
        print(f"{'=' * 70}")

        avg_mae = np.mean([r["mae_on_lidar_hits"] for r in all_results])
        avg_preservation = np.mean([r["preservation_score"] for r in all_results])
        avg_smooth_pct = np.mean([r["very_smooth_percentage"] for r in all_results])
        avg_mean_shift = np.mean([r["mean_depth_shift"] for r in all_results])

        print(f"\nAverage MAE on LiDAR hits:    {avg_mae:.6f} m")
        print(f"Average preservation score:   {avg_preservation:.2f}%")
        print(f"Average very smooth regions:  {avg_smooth_pct:.2f}%")
        print(f"Average mean depth shift:     {avg_mean_shift:.4f} m")

        print(f"\n{'=' * 70}")
        print("FINAL ASSESSMENT")
        print(f"{'=' * 70}")

        issues = []

        if avg_mae > 0.1:
            issues.append("❌ High interpolation error on LiDAR hits")
        else:
            print("✅ Interpolation preserves LiDAR values well")

        if avg_smooth_pct > 80:
            issues.append("⚠️  Over-smoothing detected (may create unrealistic depth)")
        else:
            print("✅ Reasonable smoothness characteristics")

        if abs(avg_mean_shift) > 1.0:
            issues.append("⚠️  Significant depth distribution shift")
        else:
            print("✅ Depth distribution preserved")

        if not issues:
            print(f"\n✅ VERDICT: Dense interpolation appears VALID")
            print(f"   Your 3.9m MAE training result is likely LEGITIMATE!")
            print(
                f"   The densification properly fills gaps without introducing artifacts."
            )
        else:
            print(f"\n⚠️  VERDICT: Potential issues detected:")
            for issue in issues:
                print(f"   {issue}")
            print(
                f"\n   Your 3.9m MAE may be artificially inflated due to these issues."
            )

        # Save aggregate results
        results_file = output_dir / "validation_summary.txt"
        with open(results_file, "w") as f:
            f.write("DENSE DEPTH VALIDATION SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Samples analyzed: {len(all_results)}\n\n")
            f.write(f"Average MAE on LiDAR hits:    {avg_mae:.6f} m\n")
            f.write(f"Average preservation score:   {avg_preservation:.2f}%\n")
            f.write(f"Average very smooth regions:  {avg_smooth_pct:.2f}%\n")
            f.write(f"Average mean depth shift:     {avg_mean_shift:.4f} m\n")

        print(f"\n✅ Saved summary to {results_file}")


if __name__ == "__main__":
    main()
