# Overfitting Fix - Training Configuration Changes

## Problem Identified

**Validation metrics plateaued at epoch ~8 while training continued to improve:**
- Train Abs Rel: 0.133 (epoch 9)
- Val Abs Rel: 0.203 (epoch 9) ← STUCK, not improving
- Generalization gap: 0.070 (severe overfitting)
- Best Val Abs Rel achieved: 0.196 (epoch 8)

**Root Cause:** Model is memorizing training data rather than learning generalizable depth features.

## Changes Implemented

### 1. **Dropout Regularization (NEW)**
Added Dropout2d layers throughout the model architecture:

**SNN Temporal Fusion Module:**
- Dropout after conv1 + BN (p=0.3)
- Dropout after conv2 + BN (p=0.3)

**Fusion Decoder (4 stages):**
- Dropout after each upconv + iconv layer (p=0.3)
- Total: 6 dropout layers added

**Model changes:** `models/dense_temporal_snn_depth.py`

### 2. **Stronger Data Augmentation**
Increased augmentation strength to create more diverse training samples:

| Parameter | Old Default | New Default | Change |
|-----------|-------------|-------------|--------|
| `brightness` | 0.8-1.2 | **0.6-1.4** | ±40% wider range |
| `contrast` | 0.8-1.2 | **0.6-1.4** | ±40% wider range |
| `saturation` | 0.8-1.2 | **0.6-1.4** | ±40% wider range |
| `hue` | ±0.1 | **±0.2** | 2x stronger |
| `grayscale_p` | 0.1 | **0.2** | 2x more frequent |
| `hflip_p` | 0.5 | 0.5 | (unchanged) |

**Training script changes:** `scripts/train_dense_temporal.py`

### 3. **Rebalanced Loss Function**
Added L1 (MAE) loss component and reduced SSIM weight:

| Loss Component | Old Weight | New Weight | Rationale |
|----------------|------------|------------|-----------|
| **SSIM** | 0.85 | **0.5** | Reduced - was over-emphasizing structure |
| **SILog** | 1.0 | 1.0 | (unchanged) |
| **L1 (MAE)** | N/A | **1.0** | **NEW** - directly optimizes validation metric |

**Loss formula:**
```
Total Loss = (0.5 × SSIM) + (1.0 × SILog) + (1.0 × L1)
```

**Rationale:** 
- Previous config optimized structural similarity (SSIM) which doesn't directly correlate with Abs Rel or MAE
- L1 loss directly minimizes Mean Absolute Error, the metric we care about
- This should make validation metrics improve alongside training metrics

**Loss changes:** `utils/losses.py`

## New Training Command

### Recommended Configuration
```bash
python scripts/train_dense_temporal.py \
  --kitti-root-dir /path/to/kitti/raw \
  --processed-depth-dir /path/to/dense/depth \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --dropout 0.3 \
  --ssim-weight 0.5 \
  --silog-weight 1.0 \
  --l1-weight 1.0 \
  --output-dir outputs/dense_temporal_snn_v2
```

### Conservative Configuration (if still overfitting)
```bash
python scripts/train_dense_temporal.py \
  --kitti-root-dir /path/to/kitti/raw \
  --processed-depth-dir /path/to/dense/depth \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 5e-5 \
  --dropout 0.4 \
  --weight-decay 2e-2 \
  --ssim-weight 0.3 \
  --silog-weight 1.0 \
  --l1-weight 1.5 \
  --output-dir outputs/dense_temporal_snn_v2_conservative
```

### Aggressive Regularization (last resort)
```bash
python scripts/train_dense_temporal.py \
  --kitti-root-dir /path/to/kitti/raw \
  --processed-depth-dir /path/to/dense/depth \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 5e-5 \
  --dropout 0.5 \
  --weight-decay 3e-2 \
  --ssim-weight 0.2 \
  --silog-weight 1.0 \
  --l1-weight 2.0 \
  --brightness-min 0.5 \
  --brightness-max 1.5 \
  --output-dir outputs/dense_temporal_snn_v2_aggressive
```

## Expected Improvements

### Training Behavior
- **Train loss:** Should decrease slower than before (dropout + stronger aug)
- **Val loss:** Should track train loss more closely (smaller gap)
- **Val Abs Rel:** Should improve beyond 0.196, targeting **<0.150** by epoch 20-30

### Success Criteria
✅ **Good:** Val Abs Rel < 0.150 by epoch 30
✅ **Great:** Val Abs Rel < 0.130 by epoch 50  
✅ **Excellent:** Generalization gap < 0.030 throughout training

### Warning Signs
⚠️ **Still overfitting:** Val metrics plateau while train improves → increase dropout to 0.4-0.5
⚠️ **Underfitting:** Both train and val metrics plateau early → reduce dropout to 0.2, increase LR
⚠️ **Unstable training:** Loss spikes or NaN → reduce LR, check augmentation strength

## Monitoring

Watch TensorBoard for these metrics:
```bash
tensorboard --logdir outputs/dense_temporal_snn_v2/tensorboard --port 6006
```

**Key plots to monitor:**
1. **Loss/train vs Loss/val** - should converge together, not diverge
2. **AbsRel/train vs AbsRel/val** - gap should be <0.03 for good generalization
3. **MAE/train vs MAE/val** - primary metric, both should decrease
4. **Learning Rate** - should decay smoothly with cosine schedule

## Checkpoint Management

The training script automatically saves:
- `best_model.pth` - lowest validation loss checkpoint
- `checkpoint_epoch_N.pth` - periodic checkpoints

**Current run (epoch 8 checkpoint):**
If you stopped the previous run, the best checkpoint is at epoch 8 with:
- Val Loss: 0.5054
- Val Abs Rel: 0.196
- Val MAE: ~4-5m (estimated)

## Next Steps After This Run

If validation metrics improve but still plateau:
1. **Try learning rate warmup** - start at 1e-6, warm up to 1e-4 over 5 epochs
2. **Add weight decay to SNN layers** - currently only in AdamW optimizer
3. **Implement early stopping** - stop if no improvement for 10 epochs
4. **Test on KITTI Eigen split** - official benchmark to compare with published methods

## Comparison with SOTA

Current target (after fixes):
- **BTS (2020):** 3.8-4.5m MAE, Abs Rel ~0.10-0.12
- **DPT (2021):** 4.0-4.8m MAE, Abs Rel ~0.11-0.13
- **Monodepth2 (2019):** 4.5-5.5m MAE, Abs Rel ~0.12-0.15

**Your target:** Val Abs Rel < 0.130, MAE < 5.0m (competitive with Monodepth2)

## Files Modified

1. `models/dense_temporal_snn_depth.py` - Added dropout layers
2. `utils/losses.py` - Added L1 loss, rebalanced weights
3. `scripts/train_dense_temporal.py` - Updated defaults, added arguments

**No breaking changes** - all modifications are backward compatible.
Old training runs can still be resumed without code changes.
