#!/bin/bash
################################################################################
# Resume RGB Download for KITTI Benchmark
# Handles interrupted downloads with proper error checking
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

DATA_ROOT="${1:-./data_benchmark}"

echo -e "${YELLOW}Resuming RGB image download for KITTI Benchmark${NC}"
echo ""

cd "${DATA_ROOT}"
mkdir -p raw_kitti

# Clean up any corrupted zip files (smaller than 1MB - likely incomplete)
echo "Cleaning up corrupted downloads..."
find raw_kitti -name "*.zip" -size -1M -delete 2>/dev/null || true

# Get list of drives that need RGB images
# train structure: train/2011_09_26_drive_0001_sync/proj_depth/...
DRIVES=$(find train -maxdepth 1 -type d -name "*_drive_*_sync" | sed 's|train/||' | sort -u)

total=$(echo "$DRIVES" | wc -l)
echo "Found ${total} unique drives in benchmark"
echo ""

# Debug: show first few drives
echo "Sample drives found:"
echo "$DRIVES" | head -3
echo ""

count=0
success=0
skipped=0
failed=0

for drive_path in $DRIVES; do
    count=$((count + 1))
    # drive_path format: "2011_09_26_drive_0001_sync"
    # Extract date from drive name (first 10 chars: YYYY_MM_DD)
    date=$(echo $drive_path | cut -d'_' -f1-3)  # Gets "2011_09_26"
    drive=$drive_path

    echo -e "${YELLOW}[$count/$total] Processing $date/$drive...${NC}"

    # Check if RGB images already exist
    if [ -d "raw_kitti/$date/$drive/image_02/data" ]; then
        rgb_count=$(find "raw_kitti/$date/$drive/image_02/data" -name "*.png" 2>/dev/null | wc -l)
        if [ $rgb_count -gt 0 ]; then
            echo -e "  ${GREEN}✓ Already exists ($rgb_count images), skipping${NC}"
            skipped=$((skipped + 1))
            continue
        fi
    fi

    # Download calibration if needed
    if [ ! -f "raw_kitti/$date/calib_cam_to_cam.txt" ]; then
        echo "  Downloading calibration for $date..."
        mkdir -p "raw_kitti/$date"
        if wget --timeout=30 --tries=3 -c -P raw_kitti/ \
            "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/${date}_calib.zip" 2>/dev/null; then

            if [ -f "raw_kitti/${date}_calib.zip" ]; then
                if unzip -q -o "raw_kitti/${date}_calib.zip" -d raw_kitti/ 2>/dev/null; then
                    rm "raw_kitti/${date}_calib.zip"
                    echo "  ✓ Calibration downloaded"
                else
                    echo "  ⚠ Failed to extract calibration"
                    rm -f "raw_kitti/${date}_calib.zip"
                fi
            fi
        else
            echo "  ⚠ Calibration download failed (non-critical)"
        fi
    fi

    # Download drive with resume capability and better error handling
    echo "  Downloading RGB images..."
    if wget --timeout=60 --tries=3 -c -P raw_kitti/ \
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/${drive}/${drive}.zip" 2>&1 | \
        grep -E "saved|ERROR|failed" || true; then

        if [ -f "raw_kitti/${drive}.zip" ]; then
            # Verify zip is valid before extracting
            if unzip -t "raw_kitti/${drive}.zip" >/dev/null 2>&1; then
                echo "  Extracting..."
                mkdir -p "raw_kitti/${date}"
                if unzip -q -o "raw_kitti/${drive}.zip" -d raw_kitti/${date}/ 2>/dev/null; then
                    rm "raw_kitti/${drive}.zip"

                    # Verify extraction
                    if [ -d "raw_kitti/$date/$drive/image_02/data" ]; then
                        img_count=$(find "raw_kitti/$date/$drive/image_02/data" -name "*.png" 2>/dev/null | wc -l)
                        echo -e "  ${GREEN}✓ Complete ($img_count images)${NC}"
                        success=$((success + 1))
                    else
                        echo -e "  ${RED}✗ Extraction failed - no images found${NC}"
                        failed=$((failed + 1))
                    fi
                else
                    echo -e "  ${RED}✗ Extraction failed${NC}"
                    rm -f "raw_kitti/${drive}.zip"
                    failed=$((failed + 1))
                fi
            else
                echo -e "  ${RED}✗ Downloaded zip is corrupted, removing${NC}"
                rm -f "raw_kitti/${drive}.zip"
                failed=$((failed + 1))
            fi
        else
            echo -e "  ${RED}✗ Download failed${NC}"
            failed=$((failed + 1))
        fi
    else
        echo -e "  ${RED}✗ Download failed - skipping${NC}"
        failed=$((failed + 1))
    fi

    echo ""
done

echo -e "${GREEN}================================================================================================${NC}"
echo -e "${GREEN}Download Summary:${NC}"
echo -e "${GREEN}================================================================================================${NC}"
echo "  Total drives:    $total"
echo "  Already had:     $skipped"
echo "  Downloaded:      $success"
echo "  Failed:          $failed"
echo ""

if [ $failed -gt 0 ]; then
    echo -e "${YELLOW}Note: Some downloads failed. You can re-run this script to retry.${NC}"
    echo -e "${YELLOW}Training can still proceed with partial data.${NC}"
fi

# Show final statistics
TRAIN_DEPTH=$(find train -name "*.png" -path "*/groundtruth/image_02/*" 2>/dev/null | wc -l)
RGB_IMAGES=$(find raw_kitti -name "*.png" -path "*/image_02/data/*" 2>/dev/null | wc -l)

echo -e "${GREEN}Dataset Status:${NC}"
echo "  Training depth maps: ${TRAIN_DEPTH}"
echo "  RGB images:          ${RGB_IMAGES}"
echo ""

if [ $RGB_IMAGES -lt $TRAIN_DEPTH ]; then
    missing=$((TRAIN_DEPTH - RGB_IMAGES))
    echo -e "${YELLOW}⚠ Missing ~${missing} RGB images (some drives failed)${NC}"
    echo "  Re-run this script to retry failed downloads"
else
    echo -e "${GREEN}✓ Dataset complete!${NC}"
fi

################################################################################
# Run Training with .venv Python
################################################################################

cd - > /dev/null  # Return to project root

echo ""
echo -e "${GREEN}================================================================================================${NC}"
echo -e "${GREEN}Starting Training${NC}"
echo -e "${GREEN}================================================================================================${NC}"
echo ""

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: .venv not found. Please create virtual environment first:${NC}"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Verify Python is from .venv
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

BATCH_SIZE=32
EPOCHS=50

echo -e "${GREEN}Training configuration:${NC}"
echo "  - 93K+ dense depth annotations"
echo "  - Official train/val splits"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Epochs: ${EPOCHS}"
echo "  - Anti-overfitting: dropout=0.3, strong aug, L1 loss"
echo ""

python scripts/train_dense_temporal.py \
    --kitti-root-dir "${DATA_ROOT}/raw_kitti" \
    --processed-depth-dir "${DATA_ROOT}" \
    --benchmark-mode \
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
    --output-dir outputs/benchmark_training_$(date +%Y%m%d_%H%M%S) \
    --num-workers 8

echo ""
echo -e "${GREEN}================================================================================================${NC}"
echo -e "${GREEN}TRAINING COMPLETE!${NC}"
echo -e "${GREEN}================================================================================================${NC}"
echo ""
echo -e "Target metrics to beat:"
echo "  G2I (Rank #1):      SILog = 7.34"
echo "  UniDepthV2 (#3):    SILog = 7.74"
echo "  UniDepth (#4):      SILog = 8.13"
echo "  BTS (#26):          SILog = 11.67"
echo ""
