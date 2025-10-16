import torch
import torch.nn as nn
import timm
import snntorch as snn
from snntorch import surrogate


# --- Pillar I: The Foundational Backbone ---
class HierarchicalEncoder(nn.Module):
    """
    Hierarchical Backbone for Multi-Scale Feature Extraction.
    Uses a Swin Transformer, which produces feature maps at different
    spatial resolutions (e.g., /4, /8, /16, /32 of the input size),
    which is essential for the U-Net style Fusion Decoder.
    """

    def __init__(
        self,
        backbone_name="swin_tiny_patch4_window7_224",
        pretrained=True,
        img_size=(384, 1280),
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=img_size,  # Set input size for KITTI
        )
        self.feature_info = self.backbone.feature_info

    def forward(self, x):
        """The timm Swin backbone returns features in 'channels-last' format."""
        features = self.backbone(x)
        return features


# --- Pillar II: The Temporal Fusion Core ---
class SNNTemporalFusion(nn.Module):
    """
    SNN Core for High-Level Temporal Reasoning.
    Processes a sequence of feature maps from the encoder and fuses them
    into a single, temporally-aware feature map using spiking dynamics.
    """

    def __init__(self, in_channels, out_channels, num_steps=5, beta=0.9):
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lif1 = snn.Leaky(
            beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lif2 = snn.Leaky(
            beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0
        )

    def forward(self, feature_sequence):
        """Expects 'channels-first' input: (B, T, C, H, W)"""
        self.lif1.reset_hidden()
        self.lif2.reset_hidden()
        for t in range(feature_sequence.size(1)):
            frame_features = feature_sequence[:, t]
            for _ in range(self.num_steps):
                cur1 = self.bn1(self.conv1(frame_features))
                spk1 = self.lif1(cur1)
                cur2 = self.bn2(self.conv2(spk1))
                _ = self.lif2(cur2)
        return self.lif2.mem


# --- Pillar III: The Reconstruction Head ---
class FusionDecoder(nn.Module):
    """
    Multi-Scale Fusion Decoder. Upsamples the temporally-fused features,
    incorporating skip connections from the hierarchical backbone at each stage.
    """

    def __init__(self, snn_channels, skip_channels, decoder_channels):
        super().__init__()
        # skip_channels are ordered from deep to shallow (e.g., [384, 192, 96])
        # decoder_channels are internal layer sizes (e.g., [256, 128, 64, 32])

        self.upconv1 = nn.ConvTranspose2d(snn_channels, decoder_channels[0], 2, 2)
        self.iconv1 = nn.Conv2d(
            decoder_channels[0] + skip_channels[0], decoder_channels[0], 3, padding=1
        )

        self.upconv2 = nn.ConvTranspose2d(
            decoder_channels[0], decoder_channels[1], 2, 2
        )
        self.iconv2 = nn.Conv2d(
            decoder_channels[1] + skip_channels[1], decoder_channels[1], 3, padding=1
        )

        self.upconv3 = nn.ConvTranspose2d(
            decoder_channels[1], decoder_channels[2], 2, 2
        )
        self.iconv3 = nn.Conv2d(
            decoder_channels[2] + skip_channels[2], decoder_channels[2], 3, padding=1
        )

        self.upconv4 = nn.ConvTranspose2d(
            decoder_channels[2], decoder_channels[3], 2, 2
        )
        self.iconv4 = nn.Conv2d(decoder_channels[3], decoder_channels[3], 3, padding=1)

        self.pred_conv = nn.Conv2d(decoder_channels[3], 1, 1)

    def forward(self, temporal_features, skip_connections):
        # Skip connections are ordered shallow to deep. Reverse for bottom-up decoder.
        skips = skip_connections[::-1]

        x = nn.functional.relu(self.upconv1(temporal_features))
        x = nn.functional.relu(self.iconv1(torch.cat([x, skips[0]], dim=1)))

        x = nn.functional.relu(self.upconv2(x))
        x = nn.functional.relu(self.iconv2(torch.cat([x, skips[1]], dim=1)))

        x = nn.functional.relu(self.upconv3(x))
        x = nn.functional.relu(self.iconv3(torch.cat([x, skips[2]], dim=1)))

        x = nn.functional.relu(self.upconv4(x))
        x = nn.functional.relu(self.iconv4(x))

        return self.pred_conv(x)


# --- Main Orchestrator Model ---
class DenseTemporalSNNDepth(nn.Module):
    def __init__(self, min_depth=0.1, max_depth=80.0, img_width=1280, img_height=384):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.img_width = img_width
        self.img_height = img_height

        self.encoder = HierarchicalEncoder(img_size=(img_height, img_width))

        # Get channel info from the backbone
        # Swin-T: [96, 192, 384, 768]
        encoder_channels = [info["num_chs"] for info in self.encoder.feature_info]
        snn_channels = encoder_channels[-1]  # Deepest feature: 768 channels
        skip_channels = encoder_channels[:-1]  # Shallower features: [96, 192, 384]
        decoder_channels = [256, 128, 64, 32]  # Internal decoder layer sizes

        self.temporal_fusion = SNNTemporalFusion(
            in_channels=snn_channels, out_channels=snn_channels
        )
        self.decoder = FusionDecoder(
            snn_channels=snn_channels,
            skip_channels=skip_channels[::-1],  # Pass reversed: [384, 192, 96]
            decoder_channels=decoder_channels,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, sequence, intrinsics=None):
        batch_size, num_frames, C, H, W = sequence.shape
        all_snn_features = []
        skip_connections = None

        # 1. Process each frame through the encoder
        for t in range(num_frames):
            frame = sequence[:, t]
            features_ch_last = self.encoder(frame)

            # Permute from Channels-Last [B,H,W,C] to Channels-First [B,C,H,W]
            features_ch_first = [f.permute(0, 3, 1, 2) for f in features_ch_last]

            # The last (deepest) feature is for the SNN
            all_snn_features.append(features_ch_first[-1])

            # Save the shallower features from the last frame for the decoder
            if t == num_frames - 1:
                skip_connections = features_ch_first[:-1]

        # 2. Fuse features over time with the SNN
        temporal_input = torch.stack(all_snn_features, dim=1)
        fused_features = self.temporal_fusion(temporal_input)

        # 3. Decode to a depth map using the fusion decoder
        depth_output = self.decoder(fused_features, skip_connections)

        # 4. Upsample final prediction to original image size
        depth_output = nn.functional.interpolate(
            depth_output,
            size=(self.img_height, self.img_width),
            mode="bilinear",
            align_corners=False,
        )

        # 5. Normalize and scale to the desired depth range
        depth_normalized = self.sigmoid(depth_output)
        final_depth = (
            depth_normalized * (self.max_depth - self.min_depth) + self.min_depth
        )
        return final_depth


if __name__ == "__main__":
    print("--- Testing DenseTemporalSNNDepth Model (Corrected Skip Logic) ---")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Define expected KITTI input size
        test_img_height = 384
        test_img_width = 1280

        model = DenseTemporalSNNDepth(
            img_height=test_img_height, img_width=test_img_width
        ).to(device)
        model.eval()
        print("✓ Model instantiated successfully.")

        dummy_sequence = torch.randn(2, 4, 3, test_img_height, test_img_width).to(
            device
        )
        # Dummy intrinsics (batch_size, 4, 4)
        dummy_intrinsics = torch.eye(4).unsqueeze(0).repeat(2, 1, 1).to(device)

        print(f"\nCreated dummy input sequence with shape: {dummy_sequence.shape}")
        print(f"Created dummy intrinsics with shape: {dummy_intrinsics.shape}")

        print("\nPerforming a forward pass...")
        with torch.no_grad():
            output_depth = model(dummy_sequence, dummy_intrinsics)  # Pass intrinsics
        print("✓ Forward pass complete.")

        print(f"\nOutput depth map shape: {output_depth.shape}")
        expected_shape = (2, 1, test_img_height, test_img_width)
        print(f"Expected shape: {expected_shape}")
        assert output_depth.shape == expected_shape

        min_val, max_val = torch.min(output_depth), torch.max(output_depth)
        print(f"Output depth range: min={min_val.item():.2f}, max={max_val.item():.2f}")
        assert model.min_depth <= min_val and max_val <= model.max_depth
        print("\n--- Test Passed ---")

    except ImportError as e:
        print(f"\nError: A required library is not installed. {e}")
        print("Please run: pip install timm snntorch")
    except Exception as e:
        import traceback

        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()
