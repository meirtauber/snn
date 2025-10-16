#!/bin/bash
################################################################################
# Continue Setup After Download Complete
#
# Run this if downloads finished but preprocessing failed
################################################################################

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

echo "Working directory: ${SCRIPT_DIR}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Find data directory
DATA_ROOT="./data"
if [ ! -d "${DATA_ROOT}/kitti_raw" ]; then
    echo -e "${YELLOW}Looking for KITTI data...${NC}"
    # Check common locations
    for dir in ~/project/snn/data ./data ~/data /mnt/data; do
        if [ -d "${dir}/kitti_raw" ]; then
            DATA_ROOT="${dir}"
            echo -e "${GREEN}Found KITTI data at: ${DATA_ROOT}${NC}"
            break
        fi
    done
fi

if [ ! -d "${DATA_ROOT}/kitti_raw" ]; then
    echo -e "${RED}ERROR: Cannot find KITTI raw data!${NC}"
    echo "Please specify data directory:"
    echo "  bash continue_setup.sh /path/to/data"
    exit 1
fi

# Override with argument if provided
if [ -n "$1" ]; then
    DATA_ROOT="$1"
fi

echo -e "${GREEN}Data root: ${DATA_ROOT}${NC}"
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

# Check if GPU densification script exists
if [ ! -f "utils/densify_depth_gpu.py" ]; then
    echo -e "${YELLOW}Creating GPU-accelerated densification script...${NC}"

    cat > utils/densify_depth_gpu.py << 'PYTHON_SCRIPT'
"""
GPU-Accelerated Depth Densification for KITTI.
Uses OpenCV inpainting for fast interpolation.
"""

import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import cv2
from concurrent.futures import ThreadPoolExecutor


def densify_depth_gpu(sparse_depth, device='cuda'):
    """
    GPU-accelerated depth densification using guided filter + inpainting.
    """
    h, w = sparse_depth.shape

    # Create mask of valid pixels
    mask = (sparse_depth > 0).astype(np.uint8)

    # Use OpenCV's fast inpainting
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

    # Process with thread pool
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

SPARSE_COUNT=$(find "${DATA_ROOT}/kitti_processed_depth" -name "*.npy" 2>/dev/null | wc -l)
DENSE_COUNT=$(find "${DATA_ROOT}/kitti_dense_depth" -name "*.npy" 2>/dev/null | wc -l)
RGB_COUNT=$(find "${DATA_ROOT}/kitti_raw" -name "*.png" -path "*/image_02/data/*" 2>/dev/null | wc -l)

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
# STEP 5: Launch Training
################################################################################

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}STEP 5: Ready to Launch Training${NC}"
echo -e "${BLUE}================================================================================================${NC}"

KITTI_ROOT="${DATA_ROOT}/kitti_raw"
PROCESSED_DEPTH="${DATA_ROOT}/kitti_dense_depth"
OUTPUT_DIR="outputs/dense_temporal_snn_$(date +%Y%m%d_%H%M%S)"

echo -e "${YELLOW}To start training, run:${NC}"
echo ""
echo "python scripts/train_dense_temporal.py \\"
echo "    --kitti-root-dir \"${KITTI_ROOT}\" \\"
echo "    --processed-depth-dir \"${PROCESSED_DEPTH}\" \\"
echo "    --epochs 50 \\"
echo "    --batch-size 32 \\"
echo "    --learning-rate 1e-4 \\"
echo "    --dropout 0.3 \\"
echo "    --ssim-weight 0.5 \\"
echo "    --silog-weight 1.0 \\"
echo "    --l1-weight 1.0 \\"
echo "    --output-dir \"${OUTPUT_DIR}\""
echo ""

read -p "Start training now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Starting training...${NC}"

    python scripts/train_dense_temporal.py \
        --kitti-root-dir "${KITTI_ROOT}" \
        --processed-depth-dir "${PROCESSED_DEPTH}" \
        --epochs 50 \
        --batch-size 32 \
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
else
    echo -e "${YELLOW}Training command saved above. Run it when ready!${NC}"
fi

echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
