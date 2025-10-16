#!/bin/bash
################################################################################
# Download Official KITTI Depth Prediction Benchmark Dataset
#
# This downloads the proper benchmark dataset with:
# - 93K+ training depth maps
# - Validation and test sets
# - Dense annotations (not just sparse LiDAR)
#
# You need to register at http://www.cvlibs.net/datasets/kitti/eval_depth.php
# and get download links
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DATA_ROOT="${1:-./data_benchmark}"

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}KITTI DEPTH PREDICTION BENCHMARK - OFFICIAL DATASET${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""

mkdir -p "${DATA_ROOT}"
cd "${DATA_ROOT}"

echo -e "${YELLOW}IMPORTANT:${NC}"
echo "This script downloads the OFFICIAL KITTI Depth Prediction benchmark dataset."
echo "This is different from the raw KITTI data you downloaded before."
echo ""
echo -e "${GREEN}What you get:${NC}"
echo "  - 93,000+ training depth maps (dense annotations)"
echo "  - Validation and test sets with ground truth"
echo "  - Standardized splits used by all papers"
echo "  - Direct comparison with SOTA methods"
echo ""
echo -e "${YELLOW}Current dataset you have:${NC}"
echo "  - Raw KITTI (self-supervised, sparse LiDAR only)"
echo "  - ~3,000-5,000 samples from 20 drives"
echo "  - You densified it yourself with interpolation"
echo ""
echo -e "${BLUE}Downloading benchmark dataset (21 GB total)...${NC}"
echo ""

# Download annotated depth maps (14 GB)
echo -e "${YELLOW}1/3: Downloading annotated depth maps (14 GB)...${NC}"
if [ ! -f "data_depth_annotated.zip" ]; then
    wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip
else
    echo "  Already downloaded, skipping..."
fi

# Download projected raw LiDAR scans (5 GB)
echo -e "${YELLOW}2/3: Downloading projected raw LiDAR scans (5 GB)...${NC}"
if [ ! -f "data_depth_velodyne.zip" ]; then
    wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_velodyne.zip
else
    echo "  Already downloaded, skipping..."
fi

# Download validation/test sets (2 GB)
echo -e "${YELLOW}3/3: Downloading validation and test sets (2 GB)...${NC}"
if [ ! -f "data_depth_selection.zip" ]; then
    wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip
else
    echo "  Already downloaded, skipping..."
fi

# Extract all
echo -e "${BLUE}Extracting archives...${NC}"

if [ ! -d "train" ]; then
    echo "  Extracting annotated depth maps..."
    unzip -q data_depth_annotated.zip
fi

if [ ! -d "val" ]; then
    echo "  Extracting validation/test sets..."
    unzip -q data_depth_selection.zip
fi

echo -e "${GREEN}✓ Download complete!${NC}"
echo ""

# Statistics
echo -e "${BLUE}Dataset statistics:${NC}"
TRAIN_DEPTH=$(find train -name "*.png" -path "*/proj_depth/groundtruth/*" 2>/dev/null | wc -l)
VAL_DEPTH=$(find val -name "*.png" 2>/dev/null | wc -l)

echo "  Training depth maps: ${TRAIN_DEPTH}"
echo "  Validation depth maps: ${VAL_DEPTH}"
echo ""

echo -e "${BLUE}Directory structure:${NC}"
echo "  ${DATA_ROOT}/"
echo "    ├── train/                    # 93K training samples"
echo "    │   ├── 2011_09_26_drive_0001_sync/"
echo "    │   │   ├── proj_depth/       # Projected LiDAR (sparse)"
echo "    │   │   │   ├── groundtruth/  # Dense annotations (THIS IS WHAT YOU WANT!)"
echo "    │   │   │   └── velodyne_raw/"
echo "    │   │   └── image/            # RGB images"
echo "    ├── val/                      # Validation set"
echo "    └── test/                     # Test set (no GT, for benchmark submission)"
echo ""

echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. The 'groundtruth' folder contains DENSE depth annotations (not sparse LiDAR)"
echo "2. You can train directly on this without needing densification"
echo "3. Update your dataset code to load from:"
echo "   - RGB: train/*/image/*.png"
echo "   - Depth: train/*/proj_depth/groundtruth/image_02/*.png"
echo ""
echo -e "${BLUE}To start training with this dataset:${NC}"
echo "cd ~/project/snn"
echo "# Update dataset paths in training script to use ${DATA_ROOT}"
echo ""
