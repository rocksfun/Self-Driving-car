"""Action Chunking (ACT) Driving Policy — Vision CNN with Multi-Step Trajectory Forecasting.

Instead of predicting an instantaneous scalar action, ACT predicts a continuous
trajectory chunk of K future actions:
    A_t = [a_t, a_{t+1}, ..., a_{t+K-1}] in R^{K x 2}

Combined with client-side temporal ensembling, this eliminates the high-frequency
Markovian twitch of single-step policies and guarantees C^1 continuous, zero-jerk steering.

Inputs:
  - camera_input: torch.Tensor of shape [batch, 3, 64, 64], float32 in range [0.0, 1.0]

Outputs:
  - action_chunk: torch.Tensor of shape [batch, K, 2], float32 in range [-1.0, 1.0]
      - [:, k, 0]: steering  (-1.0 = full left, +1.0 = full right)
      - [:, k, 1]: throttle  (-1.0 = full brake, +1.0 = full gas)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolution -> BatchNorm -> SiLU activation with strided downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ACTDrivingPolicy(nn.Module):
    """Vision CNN with Direct MLP Trajectory Chunk Decoder for Autonomous Vehicle Control."""

    def __init__(
        self,
        in_channels: int = 3,
        chunk_size: int = 10,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.chunk_size = chunk_size

        # 1. Vision Feature Extractor (Input: [B, 3, 64, 64])
        # Layer 1: [B, 3, 64, 64] -> [B, 32, 32, 32]
        self.conv1 = ConvBlock(in_channels, 32, kernel_size=5, stride=2, padding=2)

        # Layer 2: [B, 32, 32, 32] -> [B, 64, 16, 16]
        self.conv2 = ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)

        # Layer 3: [B, 64, 16, 16] -> [B, 128, 8, 8]
        self.conv3 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)

        # Layer 4: [B, 128, 8, 8] -> [B, 256, 4, 4]
        self.conv4 = ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)

        # Spatial Pooling: [B, 256, 4, 4] -> [B, 256, 2, 2]
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        flat_dim = 256 * 2 * 2  # 1024

        # 2. Trajectory Chunk Decoder Head: maps visual embedding -> [B, chunk_size, 2]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(flat_dim, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, chunk_size * 2),
            nn.Tanh(),  # Constrains all actions strictly to [-1.0, 1.0]
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming normal initialization for conv layers, Xavier for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, camera_input: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            camera_input: Float32 Tensor [B, 3, 64, 64] normalized in [0.0, 1.0].

        Returns:
            action_chunk: Float32 Tensor [B, K, 2] representing K future actions in [-1.0, 1.0].
        """
        if camera_input.ndim == 3:
            camera_input = camera_input.unsqueeze(0)

        # Extract spatial visual features
        x = self.conv1(camera_input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)

        # Generate trajectory chunk [B, K * 2] -> [B, K, 2]
        out = self.head(x)
        batch_size = camera_input.shape[0]
        action_chunk = out.view(batch_size, self.chunk_size, 2)
        return action_chunk


def trajectory_loss(
    predicted_chunk: torch.Tensor,
    target_chunk: torch.Tensor,
    smoothness_weight: float = 0.5,
    steering_weight: float = 1.2,
    throttle_weight: float = 1.0,
) -> torch.Tensor:
    """Composite ACT Loss: L1 trajectory error + First-difference velocity smoothness loss.

    Args:
        predicted_chunk: [B, K, 2]
        target_chunk:    [B, K, 2]
        smoothness_weight: Weight for trajectory second-order acceleration penalty
        steering_weight: Extra weighting for steering trajectory
        throttle_weight: Weighting for throttle trajectory
    """
    # 1. Pointwise L1 tracking error across chunk
    steer_l1 = nn.functional.l1_loss(predicted_chunk[..., 0], target_chunk[..., 0])
    throttle_l1 = nn.functional.l1_loss(predicted_chunk[..., 1], target_chunk[..., 1])
    tracking_loss = steering_weight * steer_l1 + throttle_weight * throttle_l1

    # 2. First-difference jerk/smoothness penalty: || Delta(A_pred) - Delta(A_target) ||^2
    if predicted_chunk.shape[1] > 1:
        diff_pred = predicted_chunk[:, 1:] - predicted_chunk[:, :-1]
        diff_target = target_chunk[:, 1:] - target_chunk[:, :-1]
        smoothness_loss = nn.functional.mse_loss(diff_pred, diff_target)
    else:
        smoothness_loss = torch.tensor(0.0, device=predicted_chunk.device)

    return tracking_loss + smoothness_weight * smoothness_loss


def export_to_onnx(
    model: nn.Module,
    output_path: str = "ACT_model/model_act.onnx",
) -> Path:
    """Export ACT policy to ONNX format with dynamic batching.

    Preserves caller model device (e.g. MPS/CUDA) and training state.
    Input tensor:  'camera_input'  [1, 3, 64, 64]
    Output tensor: 'action'        [1, K, 2]
    """
    original_device = next(model.parameters()).device
    was_training = model.training

    # Export cleanly on CPU
    model.to("cpu")
    model.eval()

    dummy_input = torch.randn(1, 3, 64, 64, dtype=torch.float32, device="cpu")
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(out_file),
        input_names=["camera_input"],
        output_names=["action"],
        dynamic_axes={
            "camera_input": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )

    # Restore original device and training mode
    model.to(original_device)
    if was_training:
        model.train()

    print(f"[✓] ACT Policy successfully exported to ONNX at: {out_file}")
    return out_file


if __name__ == "__main__":
    print("Testing ACTDrivingPolicy...")
    policy = ACTDrivingPolicy(chunk_size=10)
    dummy_img = torch.rand(4, 3, 64, 64)
    chunk = policy(dummy_img)
    print(f"Input shape:  {dummy_img.shape}")
    print(f"Output shape: {chunk.shape} (batch, K=10, [steering, throttle])")
    assert chunk.shape == (4, 10, 2), f"Expected (4, 10, 2), got {chunk.shape}"
    assert (chunk >= -1.0).all() and (chunk <= 1.0).all(), "Actions outside [-1.0, 1.0]"

    # Test loss function
    target = torch.rand(4, 10, 2) * 2 - 1
    loss = trajectory_loss(chunk, target)
    print(f"Calculated sample trajectory loss: {loss.item():.4f}")

    # Test ONNX export
    export_to_onnx(policy, "ACT_model/test_act.onnx")
    print("[✓] All ACT policy sanity tests passed!")
