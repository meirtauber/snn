# Dense Temporal SNN Depth Estimation

State-of-the-art depth estimation using Spiking Neural Networks with temporal fusion on the CARLA simulator dataset.

## Architecture Overview

This project implements a novel architecture combining:

1. **Swin Transformer Spatial Encoder** - Per-frame feature extraction with hierarchical multi-scale outputs
2. **SNN Temporal Fusion** - Sequential processing of temporal sequences using Leaky Integrate-and-Fire neurons
3. **U-Net Decoder with Skip Connections** - Dense prediction with multi-scale feature fusion
4. **Composite Loss** - SILog + SSIM for both pixel accuracy and structural correctness

### Key Features

- **Temporal reasoning**: Processes 4 consecutive frames to leverage motion parallax
- **Energy efficient**: Spiking neurons enable deployment on neuromorphic hardware
- **Dense supervision**: Trained on 20,000 diverse CARLA frames with ground truth depth
- **Multi-scale**: Hierarchical encoder-decoder with skip connections

## Dataset

### CARLA Diverse 20K Dataset

Located in `data/diverse_20k/`:
- **Total frames**: ~20,000 (40 batches × 500 frames)
- **Resolution**: 640×480 (resized to 224×224 for training)
- **Captured at**: 20 FPS
- **Depth range**: 0.1 - 80 meters
- **Diversity**:
  - 10 weather conditions (clear day, night, rain, fog, etc.)
  - 7 camera configurations (standard, wide angle, hood mount, etc.)
  - Town10HD map with varied scenes

### Data Split

- **Training**: 32 batches (15,904 sequences)
- **Validation**: 8 batches (3,976 sequences)
- **Sequence length**: 4 consecutive frames per sample

## Installation

### Requirements

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm snntorch numpy opencv-python tqdm
```

### Verify Installation

```bash
python -c "import torch; import timm; import snntorch; print('✓ All dependencies installed')"
```

## Usage

### Training

Basic training with default parameters:

```bash
python scripts/train_dense_temporal.py \
    --epochs 50 \
    --batch-size 4 \
    --learning-rate 1e-4
```

Full training with custom parameters:

```bash
python scripts/train_dense_temporal.py \
    --data-dir data/diverse_20k \
    --epochs 100 \
    --batch-size 8 \
    --learning-rate 5e-5 \
    --min-lr 1e-6 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --ssim-weight 0.85 \
    --silog-weight 1.0 \
    --output-dir outputs/dense_temporal_snn \
    --num-workers 4 \
    --save-freq 10
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `data/diverse_20k` | Path to dataset directory |
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 4 | Batch size (reduce if OOM) |
| `--learning-rate` | 1e-4 | Initial learning rate |
| `--min-lr` | 1e-6 | Minimum LR for cosine scheduler |
| `--weight-decay` | 1e-2 | AdamW weight decay |
| `--clip-grad` | 1.0 | Gradient clipping max norm |
| `--ssim-weight` | 0.85 | Weight for SSIM loss |
| `--silog-weight` | 1.0 | Weight for SILog loss |
| `--output-dir` | `outputs/dense_temporal_snn` | Checkpoint directory |
| `--num-workers` | 4 | Data loading workers |
| `--save-freq` | 10 | Save checkpoint every N epochs |

### Expected Training Time

On NVIDIA RTX 4060 Laptop GPU:
- **Per epoch**: ~3-4 hours (with batch size 4)
- **50 epochs**: ~150-200 hours (6-8 days)
- **Recommendation**: Start with 10-20 epochs to verify convergence

### Memory Requirements

- **Batch size 2**: ~8GB VRAM
- **Batch size 4**: ~12GB VRAM
- **Batch size 8**: ~20GB VRAM

Adjust `--batch-size` based on your GPU memory.

## Model Architecture Details

### Encoder (Swin Transformer)

```
Input: (B, T, 3, 224, 224) - T=4 frames
↓
Per-frame Swin Transformer feature extraction:
- Level 0: 96 channels, /4 resolution
- Level 1: 192 channels, /8 resolution
- Level 2: 384 channels, /16 resolution
- Level 3: 768 channels, /32 resolution (deepest)
```

### Temporal Fusion (SNN)

```
Deepest features from all frames → Sequential SNN processing:
- SNN Conv Layer 1: 768 → 768 channels
- SNN Conv Layer 2: 768 → 768 channels
- SNN Conv Layer 3: 768 → 768 channels
Each processed over 5 time steps with LIF neurons (β=0.9)
↓
Weighted spike aggregation → Temporal feature map
```

### Decoder (U-Net with Skip Connections)

```
Temporal features (768 channels, /32 resolution)
↓ Upsample + Skip from Level 2 (384 ch)
384 channels, /16 resolution
↓ Upsample + Skip from Level 1 (192 ch)
192 channels, /8 resolution
↓ Upsample + Skip from Level 0 (96 ch)
96 channels, /4 resolution
↓ Final upsample
1 channel, full resolution (224×224)
↓ Scale to depth range
Output: (B, 1, 224, 224) depth in meters [0.1, 80.0]
```

## Loss Function

Composite loss with two components:

1. **SILog (Scale-Invariant Logarithmic)**: Focuses on relative depth relationships
   ```
   L_silog = Var(log(pred) - log(target))
   ```

2. **SSIM (Structural Similarity)**: Preserves edges and structural details
   ```
   L_ssim = (1 - SSIM(pred, target)) / 2
   ```

Total loss:
```
L_total = β_silog × L_silog + α_ssim × L_ssim
```

Default weights: `α_ssim = 0.85`, `β_silog = 1.0`

## Evaluation Metrics

Standard depth estimation metrics:

- **MAE (Mean Absolute Error)**: Average absolute difference in meters
- **RMSE (Root Mean Square Error)**: Penalizes large errors
- **Abs Rel**: Mean absolute relative error (scale-invariant)
- **δ < 1.25**: % of pixels where max(pred/gt, gt/pred) < 1.25
- **δ < 1.25²**: Threshold accuracy at 1.5625
- **δ < 1.25³**: Threshold accuracy at 1.953

Target performance (after training):
- MAE < 5 meters
- RMSE < 8 meters
- Abs Rel < 0.15
- δ < 1.25 > 0.85

## Project Structure

```
snn-depth/
├── data/
│   └── diverse_20k/              # CARLA dataset
│       ├── batch_001.npz         # 500 frames per batch
│       ├── ...
│       ├── batch_040.npz
│       ├── metadata.json         # Dataset metadata
│       └── statistics.json       # Weather/camera distributions
├── models/
│   ├── __init__.py
│   └── dense_temporal_snn_depth.py  # Main model architecture
├── scripts/
│   ├── train_dense_temporal.py   # Training script
│   ├── train_dense.py            # Alternative training (placeholder)
│   └── collect_diverse_dataset.py  # Data collection script
├── utils/
│   ├── __init__.py
│   ├── carla_diverse_dataset.py  # Dataset loader
│   ├── losses.py                 # Composite loss (SILog + SSIM)
│   ├── metrics.py                # Evaluation metrics
│   └── visualization.py          # Plotting utilities
├── outputs/
│   └── dense_temporal_snn/       # Training outputs
│       ├── best_model.pth        # Best checkpoint
│       ├── checkpoint_epoch_*.pth
│       └── training_history.json
├── agents/                        # CARLA agents (for future work)
├── notebooks/                     # Analysis notebooks
└── requirements.txt
```

## Advanced Usage

### Resume Training from Checkpoint

```python
import torch
from models.dense_temporal_snn_depth import DenseTemporalSNNDepth

# Load checkpoint
checkpoint = torch.load('outputs/dense_temporal_snn/best_model.pth')

# Create model and load weights
model = DenseTemporalSNNDepth()
model.load_state_dict(checkpoint['model_state_dict'])

# Resume training with loaded optimizer state
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Custom Data Loading

```python
from utils.carla_diverse_dataset import CARLADiverseDataset
from torch.utils.data import DataLoader

# Create dataset with custom parameters
dataset = CARLADiverseDataset(
    data_dir='data/diverse_20k',
    split='train',
    num_frames=4,        # Temporal window size
    resize=(224, 224),   # Target resolution
    stride=2,            # Sequence stride (1=all sequences, 2=every other)
    min_depth=0.1,
    max_depth=80.0,
    normalize_rgb=True   # ImageNet normalization
)

# Create dataloader
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

### Inference Example

```python
import torch
from models.dense_temporal_snn_depth import DenseTemporalSNNDepth
from utils.carla_diverse_dataset import CARLADiverseDataset

# Load model
model = DenseTemporalSNNDepth()
checkpoint = torch.load('outputs/dense_temporal_snn/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.cuda()

# Load data
dataset = CARLADiverseDataset('data/diverse_20k', split='val')
rgb_sequence, depth_target = dataset[0]  # Get first sample

# Run inference
with torch.no_grad():
    rgb_sequence = rgb_sequence.unsqueeze(0).cuda()  # Add batch dimension
    depth_pred = model(rgb_sequence)  # (1, 1, 224, 224)
    
print(f'Predicted depth range: [{depth_pred.min():.2f}, {depth_pred.max():.2f}] meters')
```

## Troubleshooting

### Out of Memory (OOM)

**Solution**: Reduce batch size
```bash
python scripts/train_dense_temporal.py --batch-size 2
```

### Training is Too Slow

**Solutions**:
1. Reduce number of workers: `--num-workers 2`
2. Increase stride: `--stride 2` (fewer sequences)
3. Use smaller model (would require code modification)

### NaN or Inf Loss

This should be fixed in the current implementation. If you encounter this:
1. Check that `already_in_meters=True` is set in metrics computation
2. Reduce learning rate: `--learning-rate 5e-5`
3. Increase gradient clipping: `--clip-grad 0.5`

### Slow Data Loading

The first epoch is slower due to loading batches into cache. Subsequent epochs are faster.

## Performance Benchmarks

### Untrained Model (Random Initialization)

- MAE: ~30-35 meters
- Abs Rel: ~5.0
- δ < 1.25: ~0.08

### Expected After Training (50 epochs)

- MAE: < 5 meters
- RMSE: < 8 meters
- Abs Rel: < 0.15
- δ < 1.25: > 0.85

## Future Work

1. **Event Camera Integration**: Replace RGB input with DVS events
2. **Neuromorphic Hardware Deployment**: Port to Intel Loihi or BrainChip Akida
3. **Real-time Inference**: Optimize for embedded systems
4. **Domain Adaptation**: Transfer learning to real-world datasets (KITTI, nuScenes)
5. **Multi-task Learning**: Joint depth + segmentation + object detection

## References

### Papers

1. **StereoSpike** (Rançon et al., 2021): Spiking neural networks for depth estimation
2. **Swin Transformer** (Liu et al., 2021): Hierarchical vision transformer
3. **AdaBins** (Bhat et al., 2021): Adaptive bins for depth prediction
4. **SILog Loss** (Eigen et al., 2014): Scale-invariant logarithmic loss

### Resources

- **CARLA Simulator**: https://carla.org
- **snnTorch**: https://snntorch.readthedocs.io
- **timm**: https://github.com/huggingface/pytorch-image-models

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dense_temporal_snn_2025,
  title={Dense Temporal SNN for Depth Estimation},
  author={Dense Temporal SNN Team},
  year={2025},
  url={https://github.com/your-repo/snn-depth}
}
```

## License

This project is for educational and research purposes. Please ensure compliance with CARLA's license terms.

## Contact

For questions or issues, please open an issue on the project repository.

---

**Status**: Ready for training on CARLA diverse dataset (20k frames)
**Last Updated**: October 2025
