import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

# --- Component 1: Structural Similarity Index (SSIM) Loss ---
# This implementation is a widely-used PyTorch version, ensuring stability and correctness.


def gaussian(window_size, sigma):
    """Generates a 1D Gaussian kernel."""
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """Creates a 2D Gaussian window."""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """Helper function to calculate SSIM."""
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss module.
    Acts as a structural regularizer, penalizing predictions that are structurally
    dissimilar to the ground truth. It is excellent at preserving sharp edges.
    """

    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            window = window.to(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        # The SSIM metric is between 0 and 1, where 1 is perfect similarity.
        # The loss is formulated as (1 - SSIM) to be minimized.
        ssim_val = _ssim(
            img1, img2, window, self.window_size, channel, self.size_average
        )
        loss = (1.0 - ssim_val) / 2.0

        # Ensure loss is finite
        if not torch.isfinite(loss):
            return torch.tensor(0.5, device=img1.device, dtype=img1.dtype)

        return loss


# --- Component 2: Scale-Invariant Logarithmic (SILog) Loss ---


class SILogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic (SILog) loss module.
    This loss is the variance of the log-difference between the prediction
    and the ground truth. It is invariant to scale and focuses on the
    structural integrity and relative correctness of the depth relationships.
    """

    def __init__(self, variance_focus=0.85):
        super(SILogLoss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, prediction, target, mask):
        # Apply mask to both prediction and target to ignore invalid pixels
        # (e.g., sky or areas with no depth information).
        prediction = prediction[mask]
        target = target[mask]

        # Check if we have any valid pixels
        if prediction.numel() == 0:
            # No valid pixels - return zero loss
            return torch.tensor(0.0, device=prediction.device, dtype=prediction.dtype)

        # Add a small epsilon to avoid taking the log of zero.
        d = torch.log(prediction + 1e-8) - torch.log(target + 1e-8)

        # The loss is the variance of the log differences.
        # A variance_focus term can be used to tune the emphasis on the mean vs. variance.
        variance = (d**2).mean()
        mean_term = (d.mean()) ** 2

        loss = variance - self.variance_focus * mean_term

        # Clamp loss to prevent -inf or inf
        loss = torch.clamp(loss, min=-100.0, max=100.0)

        return loss


# --- The "Teacher": Composite Loss Function ---


class CompositeLoss(nn.Module):
    """
    Combines multiple loss functions with specified weights.
    This provides a comprehensive training signal that rewards both
    pixel-wise accuracy and structural correctness.
    """

    def __init__(self, alpha_ssim=0.85, beta_silog=1.0, variance_focus=0.85):
        super(CompositeLoss, self).__init__()
        self.alpha_ssim = alpha_ssim
        self.beta_silog = beta_silog

        self.ssim_loss = SSIMLoss()
        self.silog_loss = SILogLoss(variance_focus=variance_focus)

        print(
            f"CompositeLoss initialized with weights: SSIM={self.alpha_ssim}, SILog={self.beta_silog}"
        )

    def forward(self, prediction, target):
        """
        Args:
            prediction (torch.Tensor): The predicted depth map (B, 1, H, W).
            target (torch.Tensor): The ground truth depth map (B, 1, H, W).

        Returns:
            torch.Tensor: The final, combined loss value.
        """
        # A mask is crucial for dense depth datasets where some pixels (e.g., sky)
        # have no valid depth. We assume valid depth is > 0.
        mask = target > 0

        # Calculate individual losses
        silog = self.silog_loss(prediction, target, mask)
        ssim = self.ssim_loss(prediction, target)

        # Combine them with weights
        total_loss = (self.beta_silog * silog) + (self.alpha_ssim * ssim)

        return total_loss


if __name__ == "__main__":
    # --- Unit Test for the Loss Functions ---
    print("\n--- Testing Loss Functions ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dummy prediction and target tensors
    pred = torch.rand(2, 1, 10, 10, device=device) * 10 + 1
    target = pred * 1.1  # A target that is structurally similar but different in scale

    # Introduce some noise/error
    pred[0, 0, 5, 5] = 20.0

    # Create a mask (e.g., top row has no valid depth)
    target[:, :, 0, :] = 0
    mask = target > 0

    print(f"Prediction shape: {pred.shape}")
    print(f"Target shape: {target.shape}")

    # Test SSIM Loss
    ssim_loss_fn = SSIMLoss().to(device)
    ssim_value = ssim_loss_fn(pred, target)
    print(f"\nSSIM Loss: {ssim_value.item():.4f} (Expected: a small positive value)")

    # Test SILog Loss
    silog_loss_fn = SILogLoss().to(device)
    silog_value = silog_loss_fn(pred, target, mask)
    print(f"SILog Loss: {silog_value.item():.4f} (Expected: a small positive value)")

    # Test Composite Loss
    composite_loss_fn = CompositeLoss(alpha_ssim=0.5, beta_silog=0.5).to(device)
    composite_value = composite_loss_fn(pred, target)
    expected_composite = 0.5 * ssim_value + 0.5 * silog_value
    print(f"Composite Loss: {composite_value.item():.4f}")
    print(f"Manually calculated composite: {expected_composite.item():.4f}")

    assert torch.isclose(composite_value, expected_composite), (
        "Composite loss calculation is incorrect!"
    )
    print("\n--- All Loss Tests Passed ---")
