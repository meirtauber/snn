#!/bin/bash
#
# Training script for Dense Temporal SNN
#
# Usage:
#   ./train.sh                      # Start training with default parameters
#   ./train.sh test                 # Quick test run (1 epoch, batch size 2, test output dir)
#   ./train.sh full                 # Full training (50 epochs, default batch size)
#   ./train.sh large                # Large batch (50 epochs, batch size 8)
#
# Additional arguments can be passed to the Python script directly, e.g.,
#   ./train.sh test --kitti-root-dir /path/to/kitti_raw --processed-depth-dir /path/to/kitti_processed
#   ./train.sh --epochs 20 --learning-rate 1e-5

set -e # Exit on error

cd "$(dirname "$0")"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║           Dense Temporal SNN - Training Script                       ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if GPU is available
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo "✓ GPU detected: $GPU_NAME"
else
    echo "⚠ WARNING: No GPU detected. Training will be very slow on CPU."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Default parameters (can be overridden by mode or command line arguments)
PYTHON_ARGS=(
    --epochs 10
    --batch-size 4
    --output-dir "outputs/dense_temporal_snn"
    --learning-rate 1e-4
    --min-lr 1e-6
    --weight-decay 1e-2
    --clip-grad 1.0
    --ssim-weight 0.85
    --silog-weight 1.0
    --num-workers 4
    --save-freq 10
    # KITTI-specific defaults
    --kitti-root-dir "data"
    --processed-depth-dir "data/kitti_processed_depth_test" # Using the test dir for quick check
    --img-height 384
    --img-width 1280
    --min-depth 0.1
    --max-depth 80.0
    --brightness-min 0.8
    --brightness-max 1.2
    --contrast-min 0.8
    --contrast-max 1.2
    --saturation-min 0.8
    --saturation-max 1.2
    --hue-min -0.1
    --hue-max 0.1
    --grayscale-p 0.1
    --hflip-p 0.5
)

# Parse mode
MODE=${1} # Keep the first argument for mode parsing
shift # Remove the mode argument, so remaining arguments are for Python script

case $MODE in
    test)
        echo "Mode: Quick Test (1 epoch, batch size 2, custom output dir)"
        PYTHON_ARGS=( "${PYTHON_ARGS[@]}" --epochs 1 --batch-size 2 --output-dir "outputs/test_run" )
        ;;
    full)
        echo "Mode: Full Training (50 epochs)"
        PYTHON_ARGS=( "${PYTHON_ARGS[@]}" --epochs 50 --output-dir "outputs/dense_temporal_snn" )
        ;;
    large)
        echo "Mode: Large Batch (50 epochs, batch size 8)"
        PYTHON_ARGS=( "${PYTHON_ARGS[@]}" --epochs 50 --batch-size 8 --output-dir "outputs/dense_temporal_snn_bs8" )
        ;;
    default|"")
        echo "Mode: Default (10 epochs, batch size 4)"
        ;;
    *)
        # If no known mode, assume the first argument is also a Python arg
        # and prepend it to the remaining arguments
        set -- "$MODE" "$@"
        echo "Mode: Custom (defaults apply unless overridden by arguments)"
        ;;
esac

echo "Configuration for Python script:"
for arg in "${PYTHON_ARGS[@]}"; do
    echo "  $arg"
done
echo "Additional arguments from command line: $@"
echo ""

read -p "Start training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo "Starting training..."
echo "══════════════════════════════════════════════════════════════════════"
echo ""

# Run training
python scripts/train_dense_temporal.py "${PYTHON_ARGS[@]}" "$@"

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "Training complete!"
echo "Results saved to: ${OUTPUT_DIR:-outputs/dense_temporal_snn}" # Use default if OUTPUT_DIR not set by mode
echo "Best model: ${OUTPUT_DIR:-outputs/dense_temporal_snn}/best_model.pth"
echo "Training history: ${OUTPUT_DIR:-outputs/dense_temporal_snn}/training_history.json"
echo "══════════════════════════════════════════════════════════════════════"
