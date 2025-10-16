# KITTI Depth Prediction Benchmark - Quick Start

## 🎯 What's the Difference?

### ❌ What You Were Using (Raw KITTI)
- ~3,000-5,000 samples from 20 drives
- Sparse LiDAR (4% coverage)
- Self-densified with interpolation
- No standard evaluation
- Can't compare to papers

### ✅ What You Should Use (Official Benchmark)
- **93,000+ training samples** (30x more data!)
- **Dense depth annotations** (real ground truth, not interpolated)
- **Official train/val/test splits** (standard evaluation)
- **Direct leaderboard comparison** with SOTA methods
- **Target: SILog < 10** (top methods: 7-8)

## 🚀 One-Command Setup

```bash
cd ~/project/snn

# Kill current training if running
pkill -f densify_depth.py
pkill -f train_dense_temporal.py

# Download and train on benchmark (will take 3-4 hours total)
bash setup_benchmark_and_train.sh data_benchmark
```

This will:
1. Download 21GB of benchmark data (dense annotations!)
2. Download RGB images from raw KITTI
3. Automatically start training with anti-overfitting fixes
4. Target metrics comparable to published papers

## 📊 What to Expect

### Current Results (Your Raw Data)
- Training MAE: 3.9m (epoch 3)
- Validation: Unknown (self-densified, not comparable)

### Expected Results (Benchmark)
With proper training you should achieve:

| Epoch | Train SILog | Val SILog | Status |
|-------|-------------|-----------|--------|
| 10 | ~15 | ~18 | Early learning |
| 30 | ~10 | ~13 | Getting competitive |
| 50 | ~8 | ~11 | **Target: Beat BTS (11.67)** |

**Top leaderboard scores to compare against:**
- G2I (Rank #1): SILog = 7.34
- UniDepthV2 (#3): SILog = 7.74
- UniDepth (#4): SILog = 8.13
- BTS (#26): SILog = 11.67 ← **You can beat this!**

## 📁 Dataset Structure

```
data_benchmark/
├── train/                           # 93K training samples
│   └── 2011_09_26/
│       └── 2011_09_26_drive_0001_sync/
│           └── proj_depth/
│               └── groundtruth/     # ← DENSE depth (real GT!)
│                   └── image_02/*.png
├── val/                             # Validation set
│   └── depth_selection/
│       └── val_selection_cropped/
└── raw_kitti/                       # RGB images (auto-downloaded)
    └── 2011_09_26/
        └── 2011_09_26_drive_0001_sync/
            └── image_02/data/*.png
```

## 🔧 Manual Training

If you want to download data separately:

```bash
# 1. Download benchmark data
cd ~/project/snn
mkdir -p data_benchmark
cd data_benchmark

wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_velodyne.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip

unzip data_depth_annotated.zip
unzip data_depth_selection.zip

# 2. The script will auto-download RGB images during training
cd ~/project/snn

# 3. Train with benchmark
python scripts/train_dense_temporal.py \
    --benchmark-mode \
    --kitti-root-dir data_benchmark/raw_kitti \
    --processed-depth-dir data_benchmark \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --dropout 0.3 \
    --ssim-weight 0.5 \
    --silog-weight 1.0 \
    --l1-weight 1.0 \
    --output-dir outputs/benchmark_training
```

## 📈 Monitoring

### TensorBoard
```bash
tensorboard --logdir outputs/benchmark_training/tensorboard --port 6006
```

### SSH Tunnel (from your laptop)
```bash
ssh -L 6006:localhost:6006 ubuntu@192.222.51.87
# Then open: http://localhost:6006
```

### Watch for:
- **SILog metric** (primary benchmark metric, lower is better)
- **Abs Rel** (should decrease below 0.10 for competitive results)
- **Train/Val gap** (should stay < 30% with regularization)

## 🎯 Success Criteria

### Minimum Goal
- Val SILog < 15 by epoch 30
- Val Abs Rel < 0.15
- Better than raw KITTI self-densified results

### Competitive Goal
- Val SILog < 12 by epoch 50
- Val Abs Rel < 0.12
- **Beat BTS (2019): SILog = 11.67** ✅

### Stretch Goal
- Val SILog < 10
- Comparable to 2023 methods (UniDepth: 8.13)
- Potential paper submission!

## 🔥 Why This Matters

Your SNN temporal depth model with:
- Swin Transformer backbone
- Temporal fusion via SNNs
- 4-frame sequences
- Dense depth supervision

...is a **novel combination** that hasn't been tested on the official benchmark!

If you achieve **SILog < 11**, you'd be competitive with established methods and could:
1. Submit to KITTI leaderboard
2. Write a paper comparing to SOTA
3. Demonstrate SNNs can compete with standard CNNs for depth estimation

## ⚡ Quick Commands

```bash
# Download and start training (one command)
cd ~/project/snn && bash setup_benchmark_and_train.sh data_benchmark

# Check progress
tail -f outputs/benchmark_training_*/training.log

# Monitor TensorBoard
tensorboard --logdir outputs/benchmark_training_*/tensorboard --port 6006

# Check if training is running
ps aux | grep train_dense_temporal

# Kill training
pkill -f train_dense_temporal
```

## 📞 Troubleshooting

### Download fails
The benchmark is 21GB. If download interrupts, the script will resume with `-c` flag.

### RGB images missing
The script auto-downloads RGB images for each drive. If some fail, training will skip those samples.

### OOM (Out of Memory)
```bash
# Reduce batch size in the script or command:
--batch-size 16  # instead of 32
```

### Dataset not found
```bash
# Check structure:
ls data_benchmark/train/*/*/proj_depth/groundtruth/image_02/
ls data_benchmark/raw_kitti/*/*/image_02/data/
```

## 🎉 Ready?

```bash
cd ~/project/snn
bash setup_benchmark_and_train.sh data_benchmark
```
