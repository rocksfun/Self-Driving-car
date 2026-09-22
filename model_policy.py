"""Autonomous Driving Policy — Feedforward Vision CNN.

A lightweight, high-performance Convolutional Neural Network designed to drive the
3D car from a single 64x64 RGB dashcam frame.

Inputs:
  - camera_input: torch.Tensor of shape [batch, 3, 64, 64], float32 in range [0.0, 1.0]

Outputs:
  - action: torch.Tensor of shape [batch, 2], float32 in range [-1.0, 1.0]
      - action[:, 0]: steering  (-1.0 = full left, +1.0 = full right)
      - action[:, 1]: throttle  (-1.0 = full brake, +1.0 = full gas)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolution -> BatchNorm -> SiLU activation with optional downsampling."""

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


class FeedforwardDrivingPolicy(nn.Module):
    """Feedforward CNN Policy for autonomous vehicle control."""

    def __init__(self, in_channels: int = 3, dropout_prob: float = 0.1):
        super().__init__()

        # 1. Convolutional Vision Backbone (Input: [B, 3, 64, 64])
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

        # 2. Control MLP Head
        flat_dim = 256 * 2 * 2  # 1024
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(flat_dim, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh(),  # Constrains both steering & throttle strictly to [-1.0, 1.0]
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Kaiming normal for conv layers."""
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
            camera_input: Float32 Tensor [batch, 3, 64, 64] normalized in [0.0, 1.0].

        Returns:
            action: Float32 Tensor [batch, 2] representing [steering, throttle] in [-1.0, 1.0].
        """
        # Ensure shape matches expected [batch, 3, 64, 64]
        if camera_input.ndim == 3:
            camera_input = camera_input.unsqueeze(0)

        # Extract spatial visual features
        x = self.conv1(camera_input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)

        # Predict continuous actions
        action = self.head(x)
        return action

    def get_action(self, image_np) -> Tuple[float, float]:
        """Convenience method to run single-frame inference from a numpy array.

        Args:
            image_np: numpy array of shape [64, 64, 3] or [3, 64, 64] uint8 or float32.

        Returns:
            (steering, throttle) as Python floats in [-1.0, 1.0].
        """
        self.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(image_np, dtype=torch.float32)
            # If [64, 64, 3] HWC, permute to [3, 64, 64] CHW
            if tensor.ndim == 3 and tensor.shape[-1] == 3:
                tensor = tensor.permute(2, 0, 1)

            # If [0, 255] uint8 range, normalize to [0.0, 1.0]
            if tensor.max() > 1.0:
                tensor = tensor / 255.0

            tensor = tensor.unsqueeze(0)
            device = next(self.parameters()).device
            out = self.forward(tensor.to(device)).squeeze(0).cpu().numpy()
            return float(out[0]), float(out[1])


def compute_imitation_loss(
    predicted_actions: torch.Tensor,
    target_actions: torch.Tensor,
    steering_weight: float = 2.0,
    throttle_weight: float = 1.0,
) -> torch.Tensor:
    """Weighted MSE loss for imitation learning / DAgger training.

    Gives extra weight to steering precision to ensure stable path-following.
    """
    steer_loss = nn.functional.mse_loss(predicted_actions[:, 0], target_actions[:, 0])
    throttle_loss = nn.functional.mse_loss(predicted_actions[:, 1], target_actions[:, 1])
    return steering_weight * steer_loss + throttle_weight * throttle_loss


def export_to_onnx(
    model: nn.Module,
    output_path: str = "model.onnx",
    device: str = "cpu",
) -> Path:
    """Export policy to ONNX format compatible with Three.js (app3d.js) and Isaac Sim.

    Preserves the caller's model device (e.g. MPS/CUDA) and training state.
    Input tensor name: 'camera_input'  [1, 3, 64, 64]
    Output tensor name: 'action'        [1, 2]
    """
    # 1. Record original device and training mode
    original_device = next(model.parameters()).device
    was_training = model.training

    # 2. Export on CPU for maximum ONNX operator compatibility
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

    # 3. Restore original device and training mode
    model.to(original_device)
    if was_training:
        model.train()

    print(f"[✓] Policy successfully exported to ONNX at: {out_file}")
    return out_file


if __name__ == "__main__":
    print("Testing FeedforwardDrivingPolicy...")
    model = FeedforwardDrivingPolicy()
    dummy_camera = torch.rand(4, 3, 64, 64)  # batch of 4 frames
    actions = model(dummy_camera)

    print(f"Input shape:  {dummy_camera.shape}")
    print(f"Output shape: {actions.shape} (batch, [steering, throttle])")
    print(f"Sample output: {actions[0].detach().numpy()}")
    assert actions.shape == (4, 2), f"Expected [4, 2], got {actions.shape}"
    assert (actions >= -1.0).all() and (actions <= 1.0).all(), "Actions outside [-1.0, 1.0]"

    # Test single-frame numpy inference helper
    import numpy as np

    fake_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    steer, throttle = model.get_action(fake_frame)
    print(f"Single-frame numpy test -> Steering: {steer:.3f}, Throttle: {throttle:.3f}")

    # Test ONNX export
    export_to_onnx(model, "test_policy.onnx")
    print("\n[✓] All policy sanity checks passed!")
