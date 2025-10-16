# Dense Temporal SNN: Architecture and Implementation Plan

This document outlines the strategy for evolving the existing `TemporalSNNDepth` model from a sparse-supervision architecture to a state-of-the-art, dense-supervision model for the CARLA dataset.

## 1. Guiding Principles & Objectives

- **From Interpolation to Reconstruction:** The primary goal is to shift from inferring sparse layouts to reconstructing dense, pixel-perfect geometry.
- **Embrace State-of-the-Art:** We will integrate proven concepts from modern computer vision, namely Vision Transformers (ViT), advanced temporal fusion, multi-scale decoders, and composite loss functions.
- **Modularity:** The new architecture will be designed in a modular way to facilitate experimentation and future upgrades. We will separate the spatial backbone, temporal core, and reconstruction head into distinct components.
- **Maintain Core Hypothesis:** The central role of the Spiking Neural Network (SNN) as a temporal reasoner will be preserved and enhanced.

---

## 2. Architectural Overhaul: A Three-Pillar Approach

The new model, `DenseTemporalSNN`, will be built upon three foundational pillars.

### Pillar I: The Foundational Backbone (ViT Encoder)

- **Objective:** Replace the shallow CNN encoder with a powerful, pre-trained Vision Transformer backbone to achieve a global understanding of scene structure from the very first layer.
- **Component:** `ViTEncoder` module.
- **Implementation Steps:**
    1.  **Select Backbone:** We will use a pre-trained ViT from the `timm` library (e.g., `vit_small_patch16_224` or similar). This provides a robust starting point with rich features learned from a massive dataset.
    2.  **Feature Extraction Mode:** The ViT will be configured to act as a feature extractor, outputting feature maps from multiple intermediate layers, not just a final class token. This is crucial for the multi-scale decoder. `timm`'s `features_only=True` functionality will be leveraged.
    3.  **Create Wrapper Module:** A new `ViTEncoder(nn.Module)` will be created.
        -   It will take a single image frame `(B, 3, H, W)` as input.
        -   It will handle any necessary resizing and normalization to match the ViT's pre-trained requirements.
        -   Its `forward` method will return a list or dictionary of feature maps at different spatial resolutions (e.g., `[feat_1/16, feat_1/8, feat_1/4]`). The lowest resolution feature map will be fed to the temporal core, while the others will serve as skip connections.

### Pillar II: The Temporal Fusion Core (SNN Reasoner)

- **Objective:** Elevate the SNN's role from a low-level feature processor to a high-level temporal reasoner, operating on the rich semantic features provided by the ViT.
- **Component:** `SNNTemporalFusion` module.
- **Implementation Steps:**
    1.  **Isolate SNN Logic:** The core SNN layers (`snn.Leaky`, `nn.Conv2d`) will be extracted from the old architecture and placed into a new, dedicated module.
    2.  **Define Inputs/Outputs:**
        -   **Input:** A sequence of the most abstract feature maps from the `ViTEncoder` for each of the `T` frames: `(B, T, C, H_low, W_low)`.
        -   **Output:** A single, temporally-fused feature map representing the model's understanding at frame `t`: `(B, C, H_low, W_low)`.
    3.  **Refine Processing Loop:** The module's `forward` pass will explicitly loop through the time dimension (`T`), feeding each frame's feature map into the recurrent SNN layers for a set number of `num_steps`. The SNN's hidden state (membrane potential) will naturally carry information from one frame to the next.

### Pillar III: The Reconstruction Head (Multi-Scale Fusion Decoder)

- **Objective:** Replace the simple transposed convolution decoder with a modern fusion decoder that re-injects high-resolution spatial details to produce sharp, accurate depth maps.
- **Component:** `FusionDecoder` module.
- **Implementation Steps:**
    1.  **Create Upsampling Blocks:** The decoder will be composed of several `UpsampleBlock` modules. Each block will:
        a. Take a low-resolution feature map as input.
        b. Upsample it using `nn.Upsample` (bilinear) followed by a `nn.Conv2d`.
        c. Concatenate the upsampled features with the corresponding high-resolution skip connection from the `ViTEncoder`.
        d. Pass the fused features through a "fusion" `nn.Conv2d` layer to merge the information.
    2.  **Orchestrate Data Flow:**
        -   **Initial Input:** The output of the `SNNTemporalFusion` module.
        -   **Skip Connections:** The multi-scale feature maps from the `ViTEncoder` (for frame `t`).
    3.  **Final Prediction Head:**
        -   **Phase 1 (Direct Regression):** The initial implementation will use a final `nn.Conv2d` to produce a single-channel output, followed by a `nn.Sigmoid` to normalize depth values.
        -   **Phase 2 (Binning - Enhancement):** As an advanced refinement, we will explore implementing a binning head (inspired by AdaBins) to transform the regression problem into classification, which often yields higher precision.

---

## 3. The Loss Function: A More Sophisticated Teacher

- **Objective:** Implement a composite loss function that evaluates both pixel-wise accuracy and structural correctness, providing a rich gradient for the entire image.
- **Component:** A new `losses.py` file.
- **Implementation Steps:**
    1.  **Implement SILog Loss:**
        -   Create a function `compute_silog_loss(prediction, target, mask)`.
        -   It will compute `d = log(prediction) - log(target)`.
        -   The loss will be the variance of `d` over the valid pixels indicated by the mask: `(d**2).mean() - (d.mean())**2`.
    2.  **Implement SSIM Loss:**
        -   Create a function `compute_ssim_loss(prediction, target)`.
        -   We will leverage a reliable, existing implementation of SSIM (e.g., from `torchmetrics` or `piqa`) to ensure correctness and numerical stability.
        -   The loss will be formulated as `(1 - ssim) / 2`.
    3.  **Create Composite Loss Class:**
        -   A `CompositeLoss(nn.Module)` will be created.
        -   It will take weights as `__init__` arguments (e.g., `alpha_silog`, `beta_ssim`).
        -   The `forward` method will compute the individual losses and return their weighted sum. This allows for easy monitoring of each loss component during training.

---

## 4. Project Structure & Workflow

1.  **File Creation:**
    -   `models/dense_temporal_snn.py`: Will contain `DenseTemporalSNN`, `ViTEncoder`, `SNNTemporalFusion`, and `FusionDecoder`.
    -   `scripts/train_dense_temporal.py`: A new training script for the CARLA dataset.
    -   `utils/losses.py`: Will contain the SILog, SSIM, and Composite loss implementations.

2.  **Dependency Management:**
    -   Add `timm` to `requirements.txt` for the ViT backbone.
    -   Add `torchmetrics` or `piqa` for the SSIM calculation.

3.  **Development Milestones:**
    -   **M1: Backbone & Decoder:** Implement and test the `ViTEncoder` and `FusionDecoder`. Verify that a single frame can pass through this partial model and produce a correctly-sized output.
    -   **M2: SNN Integration:** Implement the `SNNTemporalFusion` module and integrate it into the full `DenseTemporalSNN` model.
    -   **M3: Loss Implementation:** Implement and unit-test the `CompositeLoss`.
    -   **M4: End-to-End Test:** Create the `train_dense_temporal.py` script and successfully run a single batch through the entire forward and backward pass without errors.
    -   **M5: Full Training & Tuning:** Launch a full training run. Monitor, debug, and tune hyperparameters (loss weights, learning rate, etc.).
