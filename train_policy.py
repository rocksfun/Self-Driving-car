"""Training script for FeedforwardDrivingPolicy using Imitation Learning / DAgger.

Loads collected demonstrations, trains the CNN policy using weighted MSE loss,
and exports the trained model to 3d/model.onnx for immediate live driving.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model_policy import FeedforwardDrivingPolicy, compute_imitation_loss, export_to_onnx


class DrivingDataset(Dataset):
    """Dataset for camera images [N, 3, 64, 64] and expert actions [N, 2]."""

    def __init__(self, dataset_path: str):
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        print(f"[*] Loading dataset from: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)

        # Support both dictionary format and tensor tuples
        if isinstance(data, dict):
            # If saved as list of episodes
            if "episodes" in data:
                all_imgs, all_acts = [], []
                for ep in data["episodes"]:
                    all_imgs.append(ep["images"])
                    all_acts.append(ep["actions"][:, :2])
                self.images = torch.cat(all_imgs, dim=0)
                self.actions = torch.cat(all_acts, dim=0)
            else:
                self.images = data["images"]
                self.actions = data["actions"][:, :2]
        elif isinstance(data, (tuple, list)):
            self.images, self.actions = data[0], data[1][:, :2]
        else:
            raise ValueError("Unrecognized dataset format.")

        # Keep images in uint8 in RAM (296 MB vs 1.2 GB) to avoid system memory pressure/swapping
        self.actions = self.actions.float()

        print(f" -> Loaded {len(self.images)} transitions.")
        print(f" -> Images shape: {self.images.shape} ({self.images.dtype}), Actions shape: {self.actions.shape}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx], self.actions[idx]


def train(
    dataset_path: str,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    output_onnx: str = "3d/model.onnx",
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
):
    print(f"[*] Training on device: {device}")
    dataset = DrivingDataset(dataset_path)

    # 80/20 Train/Validation Split via Subsets (zero copy)
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size * 2,
        shuffle=False,
    )

    model = FeedforwardDrivingPolicy().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for images, targets in train_loader:
            if images.dtype == torch.uint8:
                images = images.to(device).float() / 255.0
            else:
                images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = compute_imitation_loss(preds, targets, steering_weight=2.5, throttle_weight=1.0)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(images)

        train_loss /= train_size
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                if images.dtype == torch.uint8:
                    images = images.to(device).float() / 255.0
                else:
                    images = images.to(device)
                targets = targets.to(device)
                preds = model(images)
                loss = compute_imitation_loss(preds, targets, steering_weight=2.5, throttle_weight=1.0)
                val_loss += loss.item() * len(images)
        val_loss /= val_size

        dt = time.time() - t0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            status_tag = f" -> Best model (epoch {epoch})"
        else:
            status_tag = ""

        print(f"Epoch {epoch:02d}/{epochs:02d} [{dt:.2f}s] - Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}{status_tag}")

    print(f"\n[✓] Training complete! Best validation loss: {best_val_loss:.5f} (achieved at Epoch {best_epoch:02d})")
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Export once at the end of training
    export_to_onnx(model, output_onnx, device="cpu")
    print(f"Exported policy ready to drive at: {output_onnx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Feedforward CNN Driving Policy")
    parser.add_argument("--dataset", type=str, default="dataset_dagger.pt", help="Path to .pt dataset file (default: dataset_dagger.pt)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", type=str, default="3d/model.onnx", help="Output ONNX model path")
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_onnx=args.output,
    )
