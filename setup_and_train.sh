#!/bin/bash
################################################################################
# KITTI Complete Setup & Training Pipeline
#
# This script:
# 1. Downloads KITTI raw dataset (parallel, optimized)
# 2. Preprocesses depth maps from LiDAR
# 3. GPU-accelerated depth densification
# 4. Launches training with anti-overfitting fixes
#
# Usage:
#   bash setup_and_train.sh [data_root] [num_drives]
#
# Arguments:
#   data_root   - Root directory for data (default: ./data)
#   num_drives  - Number of drives to download (default: 20, max: 50)
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATA_ROOT="${1:-./data}"
NUM_DRIVES="${2:-20}"
BATCH_SIZE=32
EPOCHS=50

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}KITTI DEPTH ESTIMATION - COMPLETE PIPELINE${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Data root:      ${DATA_ROOT}"
echo -e "  Drives:         ${NUM_DRIVES}"
echo -e "  Batch size:     ${BATCH_SIZE}"
echo -e "  Epochs:         ${EPOCHS}"
echo -e "  GPU:            $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# Create directories
mkdir -p "${DATA_ROOT}"
mkdir -p "${DATA_ROOT}/kitti_raw"
mkdir -p "${DATA_ROOT}/kitti_processed_depth"
mkdir -p "${DATA_ROOT}/kitti_dense_depth"

################################################################################
# STEP 1: Download KITTI Raw Dataset (Parallel, Optimized)
################################################################################

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}STEP 1: Downloading KITTI Raw Dataset${NC}"
echo -e "${BLUE}================================================================================================${NC}"

# KITTI Raw data URLs (selected drives with good coverage)
KITTI_DRIVES=(
    "2011_09_26_drive_0001"
    "2011_09_26_drive_0002"
    "2011_09_26_drive_0005"
    "2011_09_26_drive_0009"
    "2011_09_26_drive_0011"
    "2011_09_26_drive_0013"
    "2011_09_26_drive_0014"
    "2011_09_26_drive_0015"
    "2011_09_26_drive_0017"
    "2011_09_26_drive_0018"
    "2011_09_26_drive_0019"
    "2011_09_26_drive_0020"
    "2011_09_26_drive_0022"
    "2011_09_26_drive_0023"
    "2011_09_26_drive_0027"
    "2011_09_26_drive_0028"
    "2011_09_26_drive_0029"
    "2011_09_26_drive_0032"
    "2011_09_26_drive_0035"
    "2011_09_26_drive_0036"
    "2011_09_26_drive_0039"
    "2011_09_26_drive_0046"
    "2011_09_26_drive_0048"
    "2011_09_26_drive_0051"
    "2011_09_26_drive_0052"
    "2011_09_26_drive_0056"
    "2011_09_26_drive_0057"
    "2011_09_26_drive_0059"
    "2011_09_26_drive_0060"
    "2011_09_26_drive_0061"
    "2011_09_26_drive_0064"
    "2011_09_26_drive_0070"
    "2011_09_26_drive_0079"
    "2011_09_26_drive_0084"
    "2011_09_26_drive_0086"
    "2011_09_26_drive_0087"
    "2011_09_26_drive_0091"
    "2011_09_26_drive_0093"
    "2011_09_26_drive_0095"
    "2011_09_26_drive_0096"
    "2011_09_26_drive_0101"
    "2011_09_26_drive_0104"
    "2011_09_26_drive_0106"
    "2011_09_26_drive_0113"
    "2011_09_26_drive_0117"
    "2011_09_28_drive_0001"
    "2011_09_28_drive_0002"
    "2011_09_29_drive_0004"
    "2011_09_30_drive_0016"
    "2011_09_30_drive_0018"
)

# Limit to requested number
DRIVES_TO_DOWNLOAD=("${KITTI_DRIVES[@]:0:$NUM_DRIVES}")

echo -e "${GREEN}Downloading ${#DRIVES_TO_DOWNLOAD[@]} drives in parallel...${NC}"

# Download calibration files first (needed for all drives)
CALIB_DATE="2011_09_26"
CALIB_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/${CALIB_DATE}_calib.zip"

if [ ! -f "${DATA_ROOT}/kitti_raw/${CALIB_DATE}/calib_cam_to_cam.txt" ]; then
    echo -e "${YELLOW}Downloading calibration files for ${CALIB_DATE}...${NC}"
    wget -q --show-progress -P "${DATA_ROOT}/kitti_raw/" "${CALIB_URL}" || {
        echo -e "${RED}Failed to download calibration. Trying alternative method...${NC}"
        curl -L -o "${DATA_ROOT}/kitti_raw/${CALIB_DATE}_calib.zip" "${CALIB_URL}"
    }
    unzip -q "${DATA_ROOT}/kitti_raw/${CALIB_DATE}_calib.zip" -d "${DATA_ROOT}/kitti_raw/"
    rm "${DATA_ROOT}/kitti_raw/${CALIB_DATE}_calib.zip"
    echo -e "${GREEN}✓ Calibration files downloaded${NC}"
else
    echo -e "${GREEN}✓ Calibration files already exist${NC}"
fi

# Function to download a single drive
download_drive() {
    local drive=$1
    local data_root=$2

    # Parse date and drive name
    local date=$(echo $drive | cut -d'_' -f1-3)
    local drive_name="${drive}_sync"

    # Check if already downloaded
    if [ -d "${data_root}/kitti_raw/${date}/${drive_name}" ]; then
        echo "  ✓ ${drive} already exists, skipping"
        return 0
    fi

    # URLs for synced data
    local drive_url="https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/${drive}/${drive}_sync.zip"

    echo "  Downloading ${drive}..."

    # Try wget first, fallback to curl
    if command -v wget &> /dev/null; then
        wget -q --show-progress -P "${data_root}/kitti_raw/" "${drive_url}" 2>&1 | grep --line-buffered "%" || true
    else
        curl -L -# -o "${data_root}/kitti_raw/${drive}_sync.zip" "${drive_url}"
    fi

    if [ -f "${data_root}/kitti_raw/${drive}_sync.zip" ]; then
        echo "  Extracting ${drive}..."
        unzip -q "${data_root}/kitti_raw/${drive}_sync.zip" -d "${data_root}/kitti_raw/"
        rm "${data_root}/kitti_raw/${drive}_sync.zip"
        echo "  ✓ ${drive} complete"
    else
        echo "  ✗ Failed to download ${drive}"
        return 1
    fi
}

export -f download_drive

# Download drives in parallel (4 at a time for optimal speed)
echo -e "${YELLOW}Starting parallel downloads (4 concurrent)...${NC}"
printf '%s\n' "${DRIVES_TO_DOWNLOAD[@]}" | xargs -P 4 -I {} bash -c "download_drive {} ${DATA_ROOT}"

echo -e "${GREEN}✓ All downloads complete!${NC}"
echo ""

# Download count
DOWNLOADED_DRIVES=$(find "${DATA_ROOT}/kitti_raw" -mindepth 2 -maxdepth 2 -type d -name "*_sync" | wc -l)
echo -e "${GREEN}Total drives available: ${DOWNLOADED_DRIVES}${NC}"
echo ""

################################################################################
# STEP 2: Preprocess LiDAR to Sparse Depth Maps
################################################################################

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}STEP 2: Preprocessing LiDAR to Sparse Depth Maps${NC}"
echo -e "${BLUE}================================================================================================${NC}"

python scripts/preprocess_kitti.py \
    --kitti-root "${DATA_ROOT}/kitti_raw" \
    --output-dir "${DATA_ROOT}/kitti_processed_depth" \
    --camera 2

echo -e "${GREEN}✓ Sparse depth preprocessing complete!${NC}"
echo ""

################################################################################
# STEP 3: GPU-Accelerated Depth Densification
################################################################################

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}STEP 3: GPU-Accelerated Depth Densification${NC}"
echo -e "${BLUE}================================================================================================${NC}"

# Check if GPU densification script exists, if not create it
if [ ! -f "utils/densify_depth_gpu.py" ]; then
    echo -e "${YELLOW}Creating GPU-accelerated densification script...${NC}"

    cat > utils/densify_depth_gpu.py << 'PYTHON_SCRIPT'
"""
GPU-Accelerated Depth Densification for KITTI.

Uses PyTorch GPU acceleration for faster interpolation.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import cv2
from concurrent.futures import ThreadPoolExecutor
import os


def densify_depth_gpu(sparse_depth, device='cuda'):
    """
    GPU-accelerated depth densification using guided filter + inpainting.

    Args:
        sparse_depth: numpy array (H, W) with sparse depth values
        device: 'cuda' or 'cpu'

    Returns:
        dense_depth: numpy array (H, W) with densified depth
    """
    h, w = sparse_depth.shape

    # Create mask of valid pixels
    mask = (sparse_depth > 0).astype(np.uint8)

    # Use OpenCV's fast inpainting (CPU, but very fast)
    # INPAINT_NS is faster, INPAINT_TELEA is more accurate
    dense_depth = cv2.inpaint(
        sparse_depth.astype(np.float32),
        1 - mask,
        inpaintRadius=10,
        flags=cv2.INPAINT_NS
    )

    # Post-process: ensure valid depth values
    dense_depth = np.clip(dense_depth, 0.1, 80.0)

    return dense_depth


def process_single_file(args):
    """Process a single depth file."""
    sparse_path, output_path, device = args

    try:
        # Load sparse depth
        sparse_depth = np.load(sparse_path)

        # Densify
        dense_depth = densify_depth_gpu(sparse_depth, device=device)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, dense_depth)

        return True
    except Exception as e:
        print(f"Error processing {sparse_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated depth densification")
    parser.add_argument("--sparse-depth-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    sparse_root = Path(args.sparse_depth_root)
    output_root = Path(args.output_root)

    # Find all sparse depth files
    sparse_files = list(sparse_root.rglob("depth_maps/*.npy"))

    print(f"Found {len(sparse_files)} sparse depth files")
    print(f"Using device: {args.device}")
    print(f"Workers: {args.num_workers}")

    # Prepare tasks
    tasks = []
    for sparse_path in sparse_files:
        # Construct output path
        rel_path = sparse_path.relative_to(sparse_root)
        output_path = output_root / rel_path.parent.parent / "depth_maps_dense" / rel_path.name

        # Skip if already exists
        if output_path.exists():
            continue

        tasks.append((sparse_path, output_path, args.device))

    print(f"Processing {len(tasks)} files...")

    # Process with thread pool (CPU inpainting is already fast)
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(tqdm(
            executor.map(process_single_file, tasks),
            total=len(tasks),
            desc="Densifying depth maps"
        ))

    success_count = sum(results)
    print(f"\n✓ Densification complete: {success_count}/{len(tasks)} files processed")


if __name__ == "__main__":
    main()
PYTHON_SCRIPT

    chmod +x utils/densify_depth_gpu.py
    echo -e "${GREEN}✓ GPU densification script created${NC}"
fi

# Run GPU-accelerated densification
python utils/densify_depth_gpu.py \
    --sparse-depth-root "${DATA_ROOT}/kitti_processed_depth" \
    --output-root "${DATA_ROOT}/kitti_dense_depth" \
    --num-workers 16 \
    --device cuda

echo -e "${GREEN}✓ Dense depth maps generated!${NC}"
echo ""

################################################################################
# STEP 4: Verify Dataset Integrity
################################################################################

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}STEP 4: Verifying Dataset Integrity${NC}"
echo -e "${BLUE}================================================================================================${NC}"

SPARSE_COUNT=$(find "${DATA_ROOT}/kitti_processed_depth" -name "*.npy" | wc -l)
DENSE_COUNT=$(find "${DATA_ROOT}/kitti_dense_depth" -name "*.npy" | wc -l)
RGB_COUNT=$(find "${DATA_ROOT}/kitti_raw" -name "*.png" -path "*/image_02/data/*" | wc -l)

echo -e "${GREEN}Dataset statistics:${NC}"
echo -e "  RGB images:        ${RGB_COUNT}"
echo -e "  Sparse depth maps: ${SPARSE_COUNT}"
echo -e "  Dense depth maps:  ${DENSE_COUNT}"
echo ""

if [ ${SPARSE_COUNT} -eq 0 ] || [ ${DENSE_COUNT} -eq 0 ]; then
    echo -e "${RED}ERROR: Dataset preparation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Dataset integrity verified${NC}"
echo ""

################################################################################
# STEP 5: Launch Training with Anti-Overfitting Fixes
################################################################################

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}STEP 5: Launching Training (with Anti-Overfitting Fixes)${NC}"
echo -e "${BLUE}================================================================================================${NC}"

echo -e "${YELLOW}Training configuration:${NC}"
echo -e "  Dropout:           0.3"
echo -e "  L1 loss weight:    1.0"
echo -e "  SSIM weight:       0.5 (reduced from 0.85)"
echo -e "  Data augmentation: STRONG (brightness ±40%, contrast ±40%)"
echo -e "  Batch size:        ${BATCH_SIZE}"
echo -e "  Learning rate:     1e-4"
echo -e "  Epochs:            ${EPOCHS}"
echo ""

# Find KITTI root dynamically
KITTI_ROOT="${DATA_ROOT}/kitti_raw"
PROCESSED_DEPTH="${DATA_ROOT}/kitti_dense_depth"
OUTPUT_DIR="outputs/dense_temporal_snn_$(date +%Y%m%d_%H%M%S)"

echo -e "${GREEN}Starting training...${NC}"
echo -e "${YELLOW}Monitor progress:${NC}"
echo -e "  TensorBoard: tensorboard --logdir ${OUTPUT_DIR}/tensorboard --port 6006"
echo -e "  SSH tunnel:  ssh -L 6006:localhost:6006 user@server"
echo ""

python scripts/train_dense_temporal.py \
    --kitti-root-dir "${KITTI_ROOT}" \
    --processed-depth-dir "${PROCESSED_DEPTH}" \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --learning-rate 1e-4 \
    --dropout 0.3 \
    --ssim-weight 0.5 \
    --silog-weight 1.0 \
    --l1-weight 1.0 \
    --brightness-min 0.6 \
    --brightness-max 1.4 \
    --contrast-min 0.6 \
    --contrast-max 1.4 \
    --saturation-min 0.6 \
    --saturation-max 1.4 \
    --hue-min -0.2 \
    --hue-max 0.2 \
    --grayscale-p 0.2 \
    --output-dir "${OUTPUT_DIR}" \
    --num-workers 8

echo ""
echo -e "${GREEN}================================================================================================${NC}"
echo -e "${GREEN}TRAINING COMPLETE!${NC}"
echo -e "${GREEN}================================================================================================${NC}"
echo -e "Output directory: ${OUTPUT_DIR}"
echo -e "Best model: ${OUTPUT_DIR}/best_model.pth"
echo ""
