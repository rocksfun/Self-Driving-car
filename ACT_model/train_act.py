#!/usr/bin/env python3
"""Training pipeline for Action Chunking (ACT) Driving Policy.

Slices sequential DAgger transitions into K-step trajectory chunks and trains
the Vision CNN with L1 trajectory tracking loss and second-order smoothness loss.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil
import sys
import time

# Ensure both the project root and current directory are in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for p in [str(PROJECT_ROOT), str(SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from ACT_model.model_act import ACTDrivingPolicy, export_to_onnx, trajectory_loss
except ModuleNotFoundError:
    from model_act import ACTDrivingPolicy, export_to_onnx, trajectory_loss


class ACTDrivingDataset(Dataset):
    """Memory-efficient PyTorch Dataset for Action Chunking.

    Stores camera frames as uint8 in RAM to minimize memory footprint.
    Slices K-step action trajectories respecting episode boundaries.
    """

    def __init__(self, pt_path: str = "dataset_dagger.pt", chunk_size: int = 10):
        super().__init__()
        self.chunk_size = chunk_size

        resolved_pt = Path(pt_path)
        if not resolved_pt.exists():
            candidate = PROJECT_ROOT / pt_path
            if candidate.exists():
                resolved_pt = candidate

        print(f"[*] Loading dataset from: {resolved_pt}")
        data = torch.load(resolved_pt, weights_only=False)

        raw_images = data["images"]
        if raw_images.dtype != torch.uint8:
            raw_images = (raw_images * 255.0).clamp(0, 255).to(torch.uint8)
        self.images = raw_images  # [N, 3, 64, 64] uint8

        self.actions = data["actions"].float()  # [N, 2] float32
        self.offsets = data["episode_offsets"].tolist()
        self.n_samples = len(self.actions)

        # Precompute episode boundaries for every frame index
        self.ep_ends = torch.empty(self.n_samples, dtype=torch.long)
        for i in range(len(self.offsets) - 1):
            s_idx = self.offsets[i]
            e_idx = self.offsets[i + 1]
            self.ep_ends[s_idx:e_idx] = e_idx

        print(f" -> Loaded {self.n_samples} transitions across {len(self.offsets) - 1} episodes.")
        print(f" -> Chunk Horizon: K = {self.chunk_size} steps ({self.chunk_size * 50} ms lookahead)")
        print(f" -> Memory Footprint: {self.images.element_size() * self.images.nelement() / (1024**2):.1f} MB (RAM)")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Convert uint8 image to float32 [0.0, 1.0] on the fly
        img = self.images[idx].float() / 255.0

        # Slice action trajectory chunk
        ep_end = self.ep_ends[idx].item()
        slice_len = min(self.chunk_size, ep_end - idx)

        chunk = torch.zeros((self.chunk_size, 2), dtype=torch.float32)
        chunk[:slice_len] = self.actions[idx : idx + slice_len]
        # Any remaining steps beyond ep_end represent the terminal parked vehicle ([0.0, 0.0])

        return img, chunk


def train_act_policy(
    dataset_path: str = "dataset_dagger.pt",
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    chunk_size: int = 10,
    output_path: str = "ACT_model/model_act.onnx",
    device: str | None = None,
) -> Path:
    """Train the ACT Driving Policy and export to ONNX."""
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    print(f"[*] Training ACT policy on device: {device}")

    # 1. Dataset & Split (85% Train / 15% Validation)
    dataset = ACTDrivingDataset(dataset_path, chunk_size=chunk_size)
    n_val = max(1, int(len(dataset) * 0.15))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # 2. Model & Optimizer
    model = ACTDrivingPolicy(in_channels=3, chunk_size=chunk_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = 1

    # 3. Epoch Loop
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        # Training phase
        model.train()
        train_loss = 0.0
        train_steps = 0
        for imgs, chunk_targets in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            chunk_targets = chunk_targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred_chunks = model(imgs)
            loss = trajectory_loss(pred_chunks, chunk_targets, smoothness_weight=0.5)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

        scheduler.step()
        avg_train_loss = train_loss / max(1, train_steps)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for imgs, chunk_targets in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                chunk_targets = chunk_targets.to(device, non_blocking=True)

                pred_chunks = model(imgs)
                loss = trajectory_loss(pred_chunks, chunk_targets, smoothness_weight=0.5)
                val_loss += loss.item()
                val_steps += 1

        avg_val_loss = val_loss / max(1, val_steps)
        dt = time.perf_counter() - t0

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        tag = f" -> Best model (epoch {epoch})" if is_best else ""
        print(f"Epoch {epoch:02d}/{epochs:02d} [{dt:.2f}s] - Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}{tag}")

    print(f"\n[✓] Training complete! Best validation loss: {best_val_loss:.5f} (achieved at Epoch {best_epoch})")

    # 4. Export Best Model Checkpoint
    model.load_state_dict(best_state_dict)
    out_file = Path(output_path)
    if not out_file.is_absolute() and not out_file.parent.exists():
        out_file = PROJECT_ROOT / output_path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_path = export_to_onnx(model, str(out_file))

    # Also deploy directly to 3d/ folder for browser access
    deploy_path = PROJECT_ROOT / "3d" / "model_act.onnx"
    if deploy_path.parent.exists():
        shutil.copyfile(out_path, deploy_path)
        print(f"[✓] Deployed ACT policy to simulation at: {deploy_path}")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Action Chunking (ACT) Driving Policy")
    parser.add_argument("--dataset", type=str, default="dataset_dagger.pt", help="Path to .pt dataset")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--chunk-size", type=int, default=10, help="Action chunk horizon (K)")
    parser.add_argument("--output", type=str, default="ACT_model/model_act.onnx", help="Output ONNX path")
    args = parser.parse_args()

    train_act_policy(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        chunk_size=args.chunk_size,
        output_path=args.output,
    )
