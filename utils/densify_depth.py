"""
Depth Completion for Sparse KITTI Depth Maps

This script densifies sparse LiDAR depth maps using various interpolation methods
to provide dense supervision for training.

Methods:
1. Nearest neighbor interpolation
2. Linear interpolation
3. Guided depth completion (uses RGB image)
"""

import numpy as np
import cv2
from scipy import interpolate
from PIL import Image
import glob
import os
from tqdm import tqdm
import argparse


def nearest_neighbor_fill(depth_map):
    """
    Fill sparse depth using nearest neighbor interpolation.
    Fast but can produce blocky artifacts.
    """
    # Find valid depth pixels
    mask = depth_map > 0

    if not np.any(mask):
        return depth_map

    # Get coordinates of valid pixels
    valid_coords = np.argwhere(mask)
    valid_depths = depth_map[mask]

    # Get coordinates of all pixels
    h, w = depth_map.shape
    all_y, all_x = np.mgrid[0:h, 0:w]

    # Interpolate using nearest neighbor
    from scipy.interpolate import NearestNDInterpolator

    interpolator = NearestNDInterpolator(valid_coords, valid_depths)
    dense_depth = interpolator(all_y, all_x)

    return dense_depth


def linear_interpolation_fill(depth_map):
    """
    Fill sparse depth using linear interpolation.
    Better quality than nearest neighbor but slower.
    """
    mask = depth_map > 0

    if not np.any(mask):
        return depth_map

    valid_coords = np.argwhere(mask)
    valid_depths = depth_map[mask]

    h, w = depth_map.shape
    all_y, all_x = np.mgrid[0:h, 0:w]

    # Use linear interpolation
    from scipy.interpolate import LinearNDInterpolator

    interpolator = LinearNDInterpolator(valid_coords, valid_depths, fill_value=0)
    dense_depth = interpolator(all_y, all_x)

    # Fill remaining zeros with nearest neighbor
    if np.any(dense_depth == 0):
        remaining_mask = dense_depth == 0
        nn_fill = nearest_neighbor_fill(depth_map)
        dense_depth[remaining_mask] = nn_fill[remaining_mask]

    return dense_depth


def guided_depth_completion(depth_map, rgb_image, epsilon=1e-4, radius=7):
    """
    Guided depth completion using RGB image as guidance.
    Uses fast guided filter for edge-aware interpolation.

    Args:
        depth_map: Sparse depth (H, W)
        rgb_image: RGB image (H, W, 3), numpy array 0-255
        epsilon: Regularization parameter
        radius: Filter radius
    """
    # First fill with linear interpolation
    base_depth = linear_interpolation_fill(depth_map)

    # Convert RGB to grayscale for guidance
    if len(rgb_image.shape) == 3:
        guidance = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    else:
        guidance = rgb_image

    # Normalize
    guidance = guidance.astype(np.float32) / 255.0
    base_depth = base_depth.astype(np.float32)

    # Apply guided filter
    dense_depth = cv2.ximgproc.guidedFilter(
        guide=guidance, src=base_depth, radius=radius, eps=epsilon
    )

    # Preserve original sparse values (trust LiDAR more than interpolation)
    mask = depth_map > 0
    dense_depth[mask] = depth_map[mask]

    return dense_depth


def inpaint_depth(depth_map):
    """
    Fill sparse depth using OpenCV's inpainting.
    Fast and produces smooth results.
    """
    # Create mask of missing pixels
    mask = (depth_map == 0).astype(np.uint8)

    # Normalize depth for inpainting
    valid_depth = depth_map[depth_map > 0]
    if len(valid_depth) == 0:
        return depth_map

    depth_normalized = depth_map.copy()
    depth_normalized = (depth_normalized / depth_normalized.max() * 255).astype(
        np.uint8
    )

    # Inpaint
    inpainted = cv2.inpaint(
        depth_normalized, mask, inpaintRadius=5, flags=cv2.INPAINT_NS
    )

    # Denormalize
    dense_depth = inpainted.astype(np.float32) / 255.0 * depth_map.max()

    # Restore original valid pixels
    valid_mask = depth_map > 0
    dense_depth[valid_mask] = depth_map[valid_mask]

    return dense_depth


def densify_kitti_depth(
    sparse_depth_dir, rgb_image_dir, output_dir, method="guided", use_rgb=True
):
    """
    Process all depth maps in a directory.

    Args:
        sparse_depth_dir: Directory with sparse .npy depth maps
        rgb_image_dir: Directory with RGB images (for guided completion)
        output_dir: Where to save dense depth maps
        method: 'nearest', 'linear', 'guided', or 'inpaint'
        use_rgb: Whether to use RGB guidance (for 'guided' method)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all depth files
    depth_files = sorted(glob.glob(os.path.join(sparse_depth_dir, "*.npy")))

    print(f"Found {len(depth_files)} depth files")
    print(f"Using method: {method}")
    print(f"Output directory: {output_dir}")

    for depth_file in tqdm(depth_files, desc="Densifying depth"):
        # Load sparse depth
        sparse_depth = np.load(depth_file)

        # Load corresponding RGB if needed
        rgb_image = None
        if use_rgb and method == "guided" and rgb_image_dir:
            rgb_filename = os.path.basename(depth_file).replace(".npy", ".png")
            rgb_path = os.path.join(rgb_image_dir, rgb_filename)
            if os.path.exists(rgb_path):
                rgb_image = np.array(Image.open(rgb_path))

        # Apply completion method
        if method == "nearest":
            dense_depth = nearest_neighbor_fill(sparse_depth)
        elif method == "linear":
            dense_depth = linear_interpolation_fill(sparse_depth)
        elif method == "guided" and rgb_image is not None:
            dense_depth = guided_depth_completion(sparse_depth, rgb_image)
        elif method == "inpaint":
            dense_depth = inpaint_depth(sparse_depth)
        else:
            # Fallback to linear if RGB not available
            dense_depth = linear_interpolation_fill(sparse_depth)

        # Save dense depth
        output_file = os.path.join(output_dir, os.path.basename(depth_file))
        np.save(output_file, dense_depth)


def main():
    parser = argparse.ArgumentParser(description="Densify sparse KITTI depth maps")
    parser.add_argument(
        "--sparse-depth-root",
        type=str,
        required=True,
        help="Root directory of sparse depth maps",
    )
    parser.add_argument(
        "--rgb-root", type=str, required=True, help="Root directory of RGB images"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Output directory for dense depth maps",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="guided",
        choices=["nearest", "linear", "guided", "inpaint"],
        help="Completion method",
    )

    args = parser.parse_args()

    # Process all drives
    drive_dirs = sorted(
        glob.glob(os.path.join(args.sparse_depth_root, "*", "*_sync", "depth_maps"))
    )

    print(f"Found {len(drive_dirs)} drives to process")

    for drive_depth_dir in drive_dirs:
        # Extract date and drive name
        parts = drive_depth_dir.split(os.sep)
        date_name = parts[-3]  # e.g., 2011_09_26
        drive_name = parts[-2]  # e.g., 2011_09_26_drive_0001_sync

        print(f"\n{'=' * 60}")
        print(f"Processing: {date_name}/{drive_name}")
        print(f"{'=' * 60}")

        # Find corresponding RGB directory
        rgb_dir = os.path.join(args.rgb_root, date_name, drive_name, "image_02", "data")

        if not os.path.exists(rgb_dir):
            print(f"Warning: RGB directory not found: {rgb_dir}")
            print("Skipping RGB guidance")
            rgb_dir = None

        # Output directory
        output_dir = os.path.join(
            args.output_root, date_name, drive_name, "depth_maps_dense"
        )

        # Process this drive
        densify_kitti_depth(
            sparse_depth_dir=drive_depth_dir,
            rgb_image_dir=rgb_dir,
            output_dir=output_dir,
            method=args.method,
            use_rgb=(rgb_dir is not None),
        )

    print("\n" + "=" * 60)
    print("✅ Depth densification complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
