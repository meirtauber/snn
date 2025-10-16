#!/bin/bash
################################################################################
# KITTI Depth Prediction BENCHMARK - Complete Pipeline
#
# Downloads the OFFICIAL benchmark dataset (93K samples) and trains
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DATA_ROOT="${1:-./data_benchmark}"
BATCH_SIZE=32
EPOCHS=50

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}KITTI DEPTH PREDICTION BENCHMARK - OFFICIAL DATASET${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${GREEN}This downloads the PROPER benchmark with:${NC}"
echo "  ✓ 93,000+ training samples (dense annotations!)"
echo "  ✓ Official validation/test splits"
echo "  ✓ Direct comparison with SOTA (G2I: SILog=7.34)"
echo ""
echo -e "${YELLOW}vs your current raw KITTI:${NC}"
echo "  ✗ Only ~3,000-5,000 samples"
echo "  ✗ Sparse LiDAR (4%) + self-densified"
echo "  ✗ No standard splits"
echo ""

mkdir -p "${DATA_ROOT}"
cd "${DATA_ROOT}"

################################################################################
# STEP 1: Download Official Benchmark Dataset
################################################################################

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}STEP 1: Downloading Official KITTI Depth Benchmark (21 GB)${NC}"
echo -e "${BLUE}================================================================================================${NC}"

# 1. Annotated depth maps (14 GB) - THIS IS THE DENSE GT!
echo -e "${YELLOW}Downloading annotated depth maps (14 GB)...${NC}"
if [ ! -f "data_depth_annotated.zip" ]; then
    wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip
else
    echo "  ✓ Already downloaded"
fi

# 2. Projected raw LiDAR (5 GB) - Sparse LiDAR projections
echo -e "${YELLOW}Downloading raw LiDAR projections (5 GB)...${NC}"
if [ ! -f "data_depth_velodyne.zip" ]; then
    wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_velodyne.zip
else
    echo "  ✓ Already downloaded"
fi

# 3. Validation/test selection (2 GB)
echo -e "${YELLOW}Downloading validation/test sets (2 GB)...${NC}"
if [ ! -f "data_depth_selection.zip" ]; then
    wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip
else
    echo "  ✓ Already downloaded"
fi

################################################################################
# STEP 2: Extract Archives
################################################################################

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}STEP 2: Extracting Archives${NC}"
echo -e "${BLUE}================================================================================================${NC}"

if [ ! -d "train" ]; then
    echo "Extracting annotated depth maps..."
    unzip -q data_depth_annotated.zip
    echo "  ✓ Training data extracted"
fi

if [ ! -d "val" ]; then
    echo "Extracting validation/test sets..."
    unzip -q data_depth_selection.zip
    echo "  ✓ Val/test data extracted"
fi

################################################################################
# STEP 3: Download RGB Images from Raw KITTI
################################################################################

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}STEP 3: Downloading RGB Images (Required)${NC}"
echo -e "${BLUE}================================================================================================${NC}"

# The benchmark depth maps are aligned with raw KITTI images
# We need to download the corresponding RGB images

echo -e "${YELLOW}NOTE: Benchmark depth maps need matching RGB images from raw KITTI${NC}"
echo "This will download ~20 drives (~30GB) with RGB images"
echo ""

mkdir -p raw_kitti

# Get list of unique dates/drives from depth maps
DRIVES=$(find train -type d -name "*_drive_*_sync" | sed 's|train/||' | sort -u)

echo "Found $(echo "$DRIVES" | wc -l) unique drives in benchmark"
echo ""

# Download RGB images for each drive
count=0
total=$(echo "$DRIVES" | wc -l)

for drive_path in $DRIVES; do
    count=$((count + 1))
    date=$(echo $drive_path | cut -d'/' -f1)
    drive=$(echo $drive_path | cut -d'/' -f2 | sed 's/_sync//')

    echo -e "${YELLOW}[$count/$total] Downloading $drive...${NC}"

    # Check if already downloaded
    if [ -d "raw_kitti/$drive_path/image_02" ]; then
        echo "  ✓ Already exists, skipping"
        continue
    fi

    # Download calibration if needed
    if [ ! -f "raw_kitti/$date/calib_cam_to_cam.txt" ]; then
        echo "  Downloading calibration for $date..."
        wget -q -P raw_kitti/ "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/${date}_calib.zip" || true
        if [ -f "raw_kitti/${date}_calib.zip" ]; then
            unzip -q "raw_kitti/${date}_calib.zip" -d raw_kitti/
            rm "raw_kitti/${date}_calib.zip"
        fi
    fi

    # Download drive
    wget -q -P raw_kitti/ "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/$drive/${drive}_sync.zip" || {
        echo "  ⚠ Failed to download $drive, will skip"
        continue
    }

    if [ -f "raw_kitti/${drive}_sync.zip" ]; then
        unzip -q "raw_kitti/${drive}_sync.zip" -d raw_kitti/
        rm "raw_kitti/${drive}_sync.zip"
        echo "  ✓ $drive complete"
    fi
done

################################################################################
# STEP 4: Verify Dataset
################################################################################

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}STEP 4: Verifying Dataset${NC}"
echo -e "${BLUE}================================================================================================${NC}"

TRAIN_DEPTH=$(find train -name "*.png" -path "*/groundtruth/image_02/*" 2>/dev/null | wc -l)
VAL_DEPTH=$(find val -name "*.png" 2>/dev/null | wc -l)
RGB_IMAGES=$(find raw_kitti -name "*.png" -path "*/image_02/data/*" 2>/dev/null | wc -l)

echo -e "${GREEN}Dataset Statistics:${NC}"
echo "  Training depth maps (DENSE GT): ${TRAIN_DEPTH}"
echo "  Validation depth maps:          ${VAL_DEPTH}"
echo "  RGB images:                     ${RGB_IMAGES}"
echo ""

if [ $TRAIN_DEPTH -lt 80000 ]; then
    echo -e "${YELLOW}⚠ Warning: Expected ~93K training samples, got ${TRAIN_DEPTH}${NC}"
    echo "  Some drives may have failed to download. Training will still work."
fi

echo -e "${GREEN}Directory structure:${NC}"
echo "  ${DATA_ROOT}/"
echo "    ├── train/                           # 93K training samples"
echo "    │   └── 2011_XX_YY/"
echo "    │       └── 2011_XX_YY_drive_ZZZZ_sync/"
echo "    │           ├── proj_depth/"
echo "    │           │   └── groundtruth/     # ← DENSE depth annotations"
echo "    │           │       └── image_02/*.png"
echo "    │           └── proj_depth/velodyne_raw/  # Sparse LiDAR"
echo "    ├── val/                             # Validation set"
echo "    └── raw_kitti/                       # RGB images"
echo "        └── 2011_XX_YY/"
echo "            └── 2011_XX_YY_drive_ZZZZ_sync/"
echo "                └── image_02/data/*.png  # ← RGB images"
echo ""

echo -e "${GREEN}✓ Dataset ready!${NC}"
echo ""

################################################################################
# STEP 5: Train on Benchmark Dataset
################################################################################

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}STEP 5: Training on Official Benchmark${NC}"
echo -e "${BLUE}================================================================================================${NC}"

cd - > /dev/null  # Return to project root

echo -e "${GREEN}Starting training with:${NC}"
echo "  - 93K+ dense depth annotations"
echo "  - Official train/val splits"
echo "  - Anti-overfitting fixes (dropout=0.3, strong aug, L1 loss)"
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
