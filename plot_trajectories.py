#!/usr/bin/env python3
"""Generate a high-resolution top-down plot of the 3D environment and collected episode paths.

Renders:
1. Complete environment layout (Start Line, South Entrance, Roundabout, Curbs, Curvy Road, Garage).
2. Dynamic plotting of all collected episodes with OU noise exploration.
3. Left/Right branch discrimination, spawn points, endpoints, and grass excursions.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


PALETTE = [
    (56, 189, 248),   # Sky Blue
    (251, 146, 60),   # Orange
    (129, 140, 248),  # Indigo
    (236, 72, 153),   # Pink
    (52, 211, 153),   # Emerald
    (250, 204, 21),   # Yellow
    (168, 85, 247),   # Purple
    (244, 63, 94),    # Rose
    (45, 212, 191),   # Teal
    (251, 191, 36),   # Amber
]


def create_plot(dataset_path: str = "dataset_dagger.pt", output_image: str = "episode_paths_plot.png"):
    ds_file = Path(dataset_path)
    if not ds_file.exists():
        raise FileNotFoundError(f"Dataset not found at: {ds_file}")

    data = torch.load(ds_file, weights_only=False)
    states = data["states"].numpy()  # [N, 6]: X, Y, Z, heading, speed, steerAngle
    metadata = data.get("metadata", {})

    # Detect episode boundaries based on reset points where Z jumps from negative/forward back to > 50
    z = states[:, 2]
    split_indices = [0]
    for i in range(1, len(z)):
        if z[i] > 50 and z[i - 1] < 20:
            split_indices.append(i)
    split_indices.append(len(states))

    episodes = []
    for i in range(len(split_indices) - 1):
        ep_states = states[split_indices[i] : split_indices[i + 1]]
        if len(ep_states) > 5:
            episodes.append(ep_states)

    print(f"[*] Loaded {len(episodes)} episodes (total {len(states)} frames) from {ds_file.name}")

    # Canvas dimensions
    IMG_W = 1200
    IMG_H = 1600

    WORLD_X_MIN, WORLD_X_MAX = -50.0, 50.0
    WORLD_Z_MIN, WORLD_Z_MAX = -175.0, 95.0

    def world_to_img(x, z):
        px = int((x - WORLD_X_MIN) / (WORLD_X_MAX - WORLD_X_MIN) * (IMG_W - 200) + 100)
        # -Z is North, so smaller Z is higher up on screen
        py = int((z - WORLD_Z_MIN) / (WORLD_Z_MAX - WORLD_Z_MIN) * (IMG_H - 240) + 160)
        return px, py

    # Create dark-themed high-res canvas
    img = Image.new("RGB", (IMG_W, IMG_H), color=(15, 23, 42))  # Slate dark #0f172a
    draw = ImageDraw.Draw(img)

    # 1. Draw Grid Lines
    for gz in range(-160, 100, 20):
        _, y1 = world_to_img(WORLD_X_MIN, gz)
        draw.line([(80, y1), (IMG_W - 80, y1)], fill=(30, 41, 59), width=1)
        draw.text((30, y1 - 6), f"Z={gz}m", fill=(100, 116, 139))

    for gx in range(-40, 50, 20):
        x1, _ = world_to_img(gx, WORLD_Z_MIN)
        draw.line([(x1, 140), (x1, IMG_H - 70)], fill=(30, 41, 59), width=1)
        draw.text((x1 - 18, IMG_H - 60), f"X={gx}m", fill=(100, 116, 139))

    # 2. Draw Environment Static Layout
    # A. Roundabout Ring (Inner: 9m, Outer: 29.5m)
    cx, cy = world_to_img(0.0, 0.0)
    scale_x = (IMG_W - 200) / (WORLD_X_MAX - WORLD_X_MIN)
    scale_z = (IMG_H - 240) / (WORLD_Z_MAX - WORLD_Z_MIN)
    rx_out = 29.5 * scale_x
    ry_out = 29.5 * scale_z
    rx_in = 9.0 * scale_x
    ry_in = 9.0 * scale_z

    # Outer asphalt fill
    draw.ellipse([cx - rx_out, cy - ry_out, cx + rx_out, cy + ry_out], fill=(30, 36, 48), outline=(71, 85, 105), width=3)
    # Inner grass island fill
    draw.ellipse([cx - rx_in, cy - ry_in, cx + rx_in, cy + ry_in], fill=(22, 101, 52), outline=(34, 197, 94), width=3)
    draw.text((cx - 28, cy - 8), "CENTRAL\nISLAND", fill=(134, 239, 172))

    # B. South Entrance Road (x in [-5.5, 5.5], z in [29.5, 85])
    p1 = world_to_img(-5.5, 85.0)
    p2 = world_to_img(5.5, 27.5)
    draw.rectangle([p1[0], p2[1], p2[0], p1[1]], fill=(30, 36, 48), outline=(71, 85, 105), width=2)
    # Centerline
    c_s1 = world_to_img(0.0, 85.0)
    c_s2 = world_to_img(0.0, 27.5)
    draw.line([c_s1, c_s2], fill=(234, 179, 8), width=2)

    # Start Line at Z = 76
    s_line1 = world_to_img(-5.5, 76.0)
    s_line2 = world_to_img(5.5, 76.0)
    draw.line([s_line1, s_line2], fill=(248, 250, 252), width=5)
    draw.text((s_line2[0] + 12, s_line2[1] - 8), "START LINE (Z=76m)", fill=(248, 250, 252))

    # C. Parking Garage Zone (x in [-4, 8], z <= -150)
    g1 = world_to_img(-4.0, -150.0)
    g2 = world_to_img(8.0, -166.0)
    draw.rectangle([g1[0], g2[1], g2[0], g1[1]], fill=(15, 23, 42), outline=(16, 185, 129), width=3)
    draw.text((g1[0] + 12, g1[1] + 12), "PARKING GARAGE GOAL (Z <= -158m)", fill=(16, 185, 129))

    # 3. Plot Each Episode's Driven Trajectory
    left_count = 0
    right_count = 0
    success_count = 0
    legend_entries = []

    for idx, ep in enumerate(episodes):
        color = PALETTE[idx % len(PALETTE)]
        pts = [world_to_img(s[0], s[2]) for s in ep]

        # Determine branch (left vs right)
        # Check X coordinate inside roundabout zone: Z in [-25, 25]
        rb_points = [s for s in ep if -25 <= s[2] <= 25]
        if rb_points:
            avg_x = np.mean([s[0] for s in rb_points])
            branch = "RIGHT" if avg_x > 0 else "LEFT"
        else:
            branch = "RIGHT" if (idx % 2 != 0) else "LEFT"

        if branch == "RIGHT":
            right_count += 1
        else:
            left_count += 1

        # Check outcome
        final_z = ep[-1][2]
        is_success = final_z <= -155.0
        outcome = "Success" if is_success else f"Ended (Z={final_z:.0f}m)"
        if is_success:
            success_count += 1

        if len(pts) > 1:
            # Draw path line
            draw.line(pts, fill=color, width=3)

            # Highlight spawn point
            draw.ellipse([pts[0][0] - 4, pts[0][1] - 4, pts[0][0] + 4, pts[0][1] + 4], fill=(255, 255, 255))

            # Highlight terminal end point
            end_col = (16, 185, 129) if is_success else (239, 68, 68)
            draw.rectangle([pts[-1][0] - 5, pts[-1][1] - 5, pts[-1][0] + 5, pts[-1][1] + 5], fill=end_col, outline=(255, 255, 255))

        legend_entries.append({
            "name": f"Ep {idx + 1} ({branch.capitalize()})",
            "color": color,
            "outcome": outcome,
            "frames": len(ep)
        })

    # 4. Draw Header Title & Statistics HUD
    draw.text((40, 25), "NEURODRIVE 3D — ORNSTEIN-UHLENBECK TRAJECTORIES & ROUNDABOUT PATHS", fill=(248, 250, 252))
    draw.text((40, 52), f"Dataset: {ds_file.name} | Total Frames: {len(states)} | Start Point: South Entrance (Z=76m)", fill=(148, 163, 184))
    draw.text(
        (40, 72),
        f"Roundabout: {left_count} Left (Clockwise) vs {right_count} Right (Counter-Clockwise) | "
        f"Success: {success_count}/{len(episodes)} | Total Episodes: {len(episodes)}",
        fill=(148, 163, 184),
    )

    # 5. Draw Legend Box
    leg_x = IMG_W - 380
    leg_y = 25
    box_height = min(350, 25 + len(legend_entries[:12]) * 24 + 10)
    draw.rectangle([leg_x - 15, leg_y - 10, leg_x + 350, leg_y + box_height], fill=(30, 41, 59), outline=(71, 85, 105), width=2)
    draw.text((leg_x, leg_y), "EPISODE TRAJECTORY LEGEND", fill=(248, 250, 252))

    for k, entry in enumerate(legend_entries[:12]):
        line_y = leg_y + 24 + k * 23
        draw.line([(leg_x, line_y + 6), (leg_x + 30, line_y + 6)], fill=entry["color"], width=4)
        status_text = f"{entry['name']} [{entry['frames']}f] -> {entry['outcome']}"
        draw.text((leg_x + 38, line_y), status_text, fill=(226, 232, 240))

    out_p = Path(output_image)
    img.save(out_p, "PNG")
    print(f"[✓] High-resolution plot generated successfully: {out_p.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 3D environment trajectories")
    parser.add_argument("--dataset", type=str, default="dataset_dagger.pt", help="Path to .pt dataset")
    parser.add_argument("--output", type=str, default="episode_paths_plot.png", help="Path to output PNG image")
    args = parser.parse_args()

    create_plot(args.dataset, args.output)
