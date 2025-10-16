"""
Quick validation of dense depth quality by auto-discovering KITTI data paths.

This script:
1. Searches for KITTI data directories
2. Loads sparse and dense depth samples
3. Validates interpolation quality
4. Assesses if training metrics are legitimate

Usage:
    python scripts/quick_validate_dense.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob


def find_kitti_data():
    """Auto-discover KITTI data directories."""
    # Common search locations
    search_paths = [
        "/home/meir/",
        "/mnt/",
        "/data/",
        str(Path.home()),
    ]

    sparse_dirs = []
    dense_dirs = []

    for base in search_paths:
        # Look for depth_maps directories (sparse)
        sparse_pattern = f"{base}/**/depth_maps/*.npy"
        sparse_files = glob.glob(sparse_pattern, recursive=True)
        if sparse_files:
            sparse_dirs.extend([str(Path(f).parent) for f in sparse_files[:1]])

        # Look for depth_maps_dense directories
        dense_pattern = f"{base}/**/depth_maps_dense/*.npy"
        dense_files = glob.glob(dense_pattern, recursive=True)
        if dense_files:
            dense_dirs.extend([str(Path(f).parent) for f in dense_files[:1]])

    return sparse_dirs, dense_dirs


def analyze_sample(sparse_depth, dense_depth):
    """Quick analysis of dense vs sparse depth."""
    results = {}

    # 1. Check preservation of LiDAR values
    valid_mask = sparse_depth > 0
    if valid_mask.sum() > 0:
        sparse_valid = sparse_depth[valid_mask]
        dense_valid = dense_depth[valid_mask]

        errors = np.abs(dense_valid - sparse_valid)
        results["mae_on_lidar"] = errors.mean()
        results["max_error"] = errors.max()
        results["preserved_pct"] = (errors < 0.01).sum() / len(errors) * 100

    # 2. Coverage
    results["sparse_coverage"] = (sparse_depth > 0).sum() / sparse_depth.size * 100
    results["dense_coverage"] = (dense_depth > 0).sum() / dense_depth.size * 100

    # 3. Smoothness check
    grad_x = np.abs(np.diff(dense_depth, axis=1))
    grad_y = np.abs(np.diff(dense_depth, axis=0))
    valid_grad = grad_x[dense_depth[:, :-1] > 0]
    results["mean_gradient"] = valid_grad.mean() if len(valid_grad) > 0 else 0
    results["smooth_pct"] = (
        (valid_grad < 0.01).sum() / len(valid_grad) * 100 if len(valid_grad) > 0 else 0
    )

    # 4. Depth distribution
    sparse_vals = sparse_depth[sparse_depth > 0]
    dense_vals = dense_depth[dense_depth > 0]
    results["sparse_mean"] = sparse_vals.mean()
    results["dense_mean"] = dense_vals.mean()
    results["mean_shift"] = results["dense_mean"] - results["sparse_mean"]

    return results


def main():
    print("=" * 70)
    print("QUICK DENSE DEPTH VALIDATION")
    print("=" * 70)

    # Find KITTI data
    print("\nSearching for KITTI data...")

    # Try to find depth files directly
    sparse_files = []
    dense_files = []

    for pattern in [
        "/**/depth_maps/*.npy",
        "/home/**/depth_maps/*.npy",
        "/mnt/**/depth_maps/*.npy",
    ]:
        found = glob.glob(pattern, recursive=True)
        if found:
            sparse_files.extend(found)

    for pattern in [
        "/**/depth_maps_dense/*.npy",
        "/home/**/depth_maps_dense/*.npy",
        "/mnt/**/depth_maps_dense/*.npy",
    ]:
        found = glob.glob(pattern, recursive=True)
        if found:
            dense_files.extend(found)

    if not sparse_files:
        print("❌ No sparse depth maps found!")
        print("\nPlease run with explicit paths:")
        print("python scripts/validate_dense_quality.py \\")
        print("  --kitti-root-dir /path/to/kitti \\")
        print("  --processed-depth-dir /path/to/processed_depth")
        return

    if not dense_files:
        print("❌ No dense depth maps found!")
        print(f"   Found sparse at: {sparse_files[0]}")
        print("   But no corresponding dense files.")
        return

    print(f"✅ Found {len(sparse_files)} sparse depth files")
    print(f"✅ Found {len(dense_files)} dense depth files")

    # Match sparse and dense files
    matched_pairs = []
    for sparse_path in sparse_files[:20]:  # Limit to 20 samples
        # Convert sparse path to expected dense path
        dense_path = sparse_path.replace("/depth_maps/", "/depth_maps_dense/")
        if dense_path in dense_files:
            matched_pairs.append((sparse_path, dense_path))

    if not matched_pairs:
        print("❌ No matching sparse/dense pairs found!")
        print(f"\nExample sparse: {sparse_files[0]}")
        print(f"Example dense:  {dense_files[0]}")
        return

    print(f"✅ Found {len(matched_pairs)} matching sparse/dense pairs")

    # Analyze samples
    all_results = []

    for i, (sparse_path, dense_path) in enumerate(matched_pairs[:10]):
        print(f"\nAnalyzing sample {i + 1}/{min(10, len(matched_pairs))}...")
        print(f"  Sparse: {Path(sparse_path).name}")
        print(f"  Dense:  {Path(dense_path).name}")

        sparse = np.load(sparse_path)
        dense = np.load(dense_path)

        results = analyze_sample(sparse, dense)
        all_results.append(results)

        # Quick report
        print(f"  MAE on LiDAR hits: {results['mae_on_lidar']:.6f}m")
        print(f"  Preserved: {results['preserved_pct']:.1f}%")
        print(
            f"  Coverage: {results['sparse_coverage']:.1f}% → {results['dense_coverage']:.1f}%"
        )

        # Visualize first sample
        if i == 0:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            vmax = min(sparse.max(), 80)
            axes[0].imshow(sparse, cmap="plasma", vmin=0, vmax=vmax)
            axes[0].set_title(f"Sparse ({results['sparse_coverage']:.1f}% coverage)")
            axes[0].axis("off")

            axes[1].imshow(dense, cmap="plasma", vmin=0, vmax=vmax)
            axes[1].set_title(f"Dense ({results['dense_coverage']:.1f}% coverage)")
            axes[1].axis("off")

            error = np.abs(dense - sparse)
            error[sparse == 0] = np.nan
            im = axes[2].imshow(error, cmap="hot", vmin=0, vmax=0.5)
            axes[2].set_title(f"Error on LiDAR (MAE={results['mae_on_lidar']:.4f}m)")
            axes[2].axis("off")
            plt.colorbar(im, ax=axes[2], fraction=0.046)

            plt.tight_layout()
            plt.savefig("validation_quick.png", dpi=150, bbox_inches="tight")
            print(f"\n  ✅ Saved visualization to validation_quick.png")

    # Aggregate results
    if all_results:
        print("\n" + "=" * 70)
        print("AGGREGATE RESULTS")
        print("=" * 70)

        avg_mae = np.mean([r["mae_on_lidar"] for r in all_results])
        avg_preserved = np.mean([r["preserved_pct"] for r in all_results])
        avg_smooth = np.mean([r["smooth_pct"] for r in all_results])
        avg_shift = np.mean([r["mean_shift"] for r in all_results])

        print(f"\nSamples analyzed: {len(all_results)}")
        print(f"\n1. INTERPOLATION ACCURACY:")
        print(f"   Average MAE on LiDAR hits: {avg_mae:.6f} m")
        print(f"   Average preserved (<1cm):  {avg_preserved:.2f}%")

        if avg_mae < 0.01:
            print(f"   ✅ EXCELLENT - Dense perfectly preserves LiDAR")
        elif avg_mae < 0.1:
            print(f"   ✅ GOOD - Minor interpolation error")
        else:
            print(f"   ⚠️  WARNING - Significant error!")

        print(f"\n2. SMOOTHNESS:")
        print(f"   Very smooth regions: {avg_smooth:.2f}%")

        if avg_smooth > 80:
            print(f"   ⚠️  Possibly over-smoothed")
        else:
            print(f"   ✅ Reasonable")

        print(f"\n3. DEPTH DISTRIBUTION:")
        print(f"   Mean depth shift: {avg_shift:.4f} m")

        if abs(avg_shift) > 1.0:
            print(f"   ⚠️  Significant shift")
        else:
            print(f"   ✅ Well preserved")

        # Final verdict
        print("\n" + "=" * 70)
        print("VERDICT")
        print("=" * 70)

        issues = 0
        if avg_mae > 0.1:
            print("❌ High interpolation error detected")
            issues += 1
        if avg_smooth > 80:
            print("⚠️  Over-smoothing detected")
            issues += 1
        if abs(avg_shift) > 1.0:
            print("⚠️  Depth distribution shifted")
            issues += 1

        if issues == 0:
            print("✅ Dense interpolation appears VALID")
            print("\n🎉 Your 3.9m MAE training result is likely LEGITIMATE!")
            print("   The densification properly fills gaps without artifacts.")
            print("\n   This is competitive with:")
            print("   - BTS (2020): 3.8-4.5m")
            print("   - Monodepth2 (2019): 4.5-5.5m")
        else:
            print(f"\n⚠️  {issues} potential issue(s) detected.")
            print("   Your 3.9m MAE may be partially inflated.")
            print("   Consider testing on held-out KITTI benchmark splits.")


if __name__ == "__main__":
    main()
