#!/usr/bin/env python3
"""Autonomous Driving Agent — Standalone Python Driver with Action Chunking (ACT).

This standalone agent completely decouples the autonomous driver logic from the 3D
simulation environment (car3d.js / track3d.js):

1. The Car & Environment (car3d.js / track3d.js):
   - Treated as an untouched, 100% third-party physical vehicle plant.
   - Accepts raw actuator signals across the boundary: [steering, throttle].

2. The Autonomous Agent (agent.py):
   - Receives raw 64x64 RGB camera frames over a high-speed local WebSocket (ws://127.0.0.1:8765).
   - Runs model inference (Action Chunking Policy or Baseline CNN).
   - Manages the Temporal Ensembling queue across overlapping trajectory chunks.
   - Bridges policy decisions to the third-party vehicle plant.
   - Sends real-time continuous control commands back to the vehicle in under 2 ms.

Modes:
  - Default: Starts the live WebSocket agent server to drive any browser window.
  - Headless (--headless): Spawns a headless simulation session via Playwright and
    drives complete laps autonomously while displaying live terminal HUD telemetry.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
import time
from typing import Optional, Tuple

import numpy as np
import onnxruntime as ort


class AutonomousAgent:
    """Independent Autonomous Vehicle Agent with Action Chunking & Temporal Ensembling."""

    def __init__(
        self,
        model_path: str = "3d/model_act.onnx",
        chunk_horizon: int = 10,
        decay_rate: float = 0.25,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            # Fallback to ACT_model/ or baseline model.onnx
            alt_paths = [
                Path("ACT_model/model_act.onnx"),
                Path("3d/model.onnx"),
                Path("model.onnx"),
            ]
            for alt in alt_paths:
                if alt.exists():
                    self.model_path = alt
                    break

        print(f"[*] Initializing Autonomous Agent with model: {self.model_path}")
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.output_shape = self.session.get_outputs()[0].shape

        # Detect whether the loaded policy is an ACT chunking model or single-step
        self.is_chunked = len(self.output_shape) == 3 and self.output_shape[1] > 1
        self.chunk_size = self.output_shape[1] if self.is_chunked else 1
        self.decay_rate = decay_rate

        # Temporal Ensembling FIFO buffer: stores [{'chunk': np.ndarray, 'age': int}]
        self.history: list[dict] = []
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)

        policy_type = f"ACT Action Chunking (K={self.chunk_size})" if self.is_chunked else "Single-Step Baseline CNN"
        print(f" -> Policy Architecture: {policy_type}")
        print(f" -> Input: {self.input_name} {self.session.get_inputs()[0].shape}")
        print(f" -> Output: {self.output_name} {self.output_shape}")

    def reset(self):
        """Reset internal agent memory and temporal ensembling queue for a new episode."""
        self.history.clear()
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)

    def preprocess_image(self, raw_bytes: bytes) -> np.ndarray:
        """Preprocess 12,288 raw RGB bytes into a normalized Float32 tensor [1, 3, 64, 64]."""
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        if len(arr) == 64 * 64 * 4:
            # Handle RGBA from canvas getImageData: drop alpha
            arr = arr.reshape((64, 64, 4))[:, :, :3]
        elif len(arr) == 64 * 64 * 3:
            arr = arr.reshape((64, 64, 3))
        else:
            raise ValueError(f"Unexpected pixel byte length: {len(arr)} (expected 12288 RGB or 16384 RGBA)")

        # Convert [H, W, C] uint8 -> [1, C, H, W] float32 in [0.0, 1.0]
        tensor = arr.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(tensor, axis=0)

    def act(self, raw_bytes: bytes, current_speed: Optional[float] = None) -> Tuple[float, float, float]:
        """Perform perception, model inference, temporal ensembling, and plant bridging.

        Args:
            raw_bytes: Raw 64x64 RGB or RGBA camera bytes.
            current_speed: Optional vehicle telemetry speed (m/s) for actuator envelope regulation.

        Returns:
            steering (float): Steer angle in [-1.0, 1.0]
            throttle (float): Throttle/brake in [-1.0, 1.0]
            latency_ms (float): Inference execution time in milliseconds
        """
        t0 = time.perf_counter()
        img_tensor = self.preprocess_image(raw_bytes)

        # 1. Model Forward Pass
        results = self.session.run([self.output_name], {self.input_name: img_tensor})
        out = results[0]
        dt_ms = (time.perf_counter() - t0) * 1000.0

        if self.is_chunked:
            # Output shape: [1, K, 2]
            current_chunk = np.clip(out[0], -1.0, 1.0)  # [K, 2]

            # 2. Advance age of previous predictions
            for item in self.history:
                item["age"] += 1

            # Discard stale chunks older than K
            self.history = [item for item in self.history if item["age"] < self.chunk_size]

            # Add fresh prediction chunk at age 0
            self.history.insert(0, {"chunk": current_chunk, "age": 0})

            # 3. Temporal Ensembling across overlapping predictions
            total_weight = 0.0
            weighted_action = np.zeros(2, dtype=np.float32)

            for item in self.history:
                age = item["age"]
                if age < len(item["chunk"]):
                    w = np.exp(-self.decay_rate * age)
                    weighted_action += w * item["chunk"][age]
                    total_weight += w

            final_action = weighted_action / max(1e-6, total_weight)
        else:
            # Single-step model: apply light 70/30 EMA smoothing
            raw_action = np.clip(out[0], -1.0, 1.0)
            final_action = 0.70 * raw_action + 0.30 * self.prev_action

        self.prev_action = final_action
        steer = float(np.clip(final_action[0], -1.0, 1.0))
        throttle = float(np.clip(final_action[1], -1.0, 1.0))

        # 4. Plant Actuator Bridging
        # The third-party car3d.js vehicle lacks aerodynamic drag equations; in stock physics
        # any positive throttle accelerates continuously without drag.
        # When cruising (>13 m/s) under light throttle (0.05 < T <= 0.32), the agent modulates
        # throttle to coast (0.0) so engine drag maintains safe cruise velocity.
        if current_speed is not None and current_speed > 13.0 and 0.0 < throttle <= 0.32:
            throttle = 0.0

        return steer, throttle, dt_ms


# =============================================================================
# WebSocket Communication Bridge (ws://127.0.0.1:8765)
# =============================================================================
async def run_websocket_server(agent: AutonomousAgent, host: str = "127.0.0.1", port: int = 8765):
    """Run an asynchronous WebSocket server bridging the agent with browser clients."""
    import websockets

    step_counter = 0

    async def handler(websocket):
        nonlocal step_counter
        client_addr = websocket.remote_address
        print(f"[+] Simulation client connected from: {client_addr}")
        agent.reset()

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # Binary Camera Frame (12,288 bytes RGB or 16,384 bytes RGBA)
                    step_counter += 1
                    steer, throttle, latency = agent.act(message)

                    response = {
                        "type": "action",
                        "step": step_counter,
                        "steering": round(steer, 4),
                        "throttle": round(throttle, 4),
                        "latencyMs": round(latency, 2),
                    }
                    await websocket.send(json.dumps(response))

                elif isinstance(message, str):
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "observe":
                        # JSON Observation with raw RGB base64 and vehicle telemetry
                        step_counter += 1
                        raw_bytes = base64.b64decode(data["rgbBase64"])
                        speed = float(data.get("speed", 0.0))
                        req_id = data.get("reqId", step_counter)
                        steer, throttle, latency = agent.act(raw_bytes, current_speed=speed)

                        response = {
                            "type": "action",
                            "reqId": req_id,
                            "step": step_counter,
                            "steering": round(steer, 4),
                            "throttle": round(throttle, 4),
                            "latencyMs": round(latency, 2),
                        }
                        await websocket.send(json.dumps(response))

                    elif msg_type == "reset":
                        agent.reset()
                        step_counter = 0
                        print(" -> Agent reset for new episode.")
                        await websocket.send(json.dumps({"type": "reset_ack", "status": "ok"}))

                    elif msg_type == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))

        except websockets.exceptions.ConnectionClosed:
            print(f"[-] Simulation client disconnected: {client_addr}")
        except Exception as exc:
            print(f"[!] Error in agent connection: {exc}")

    print(f"\n===========================================================================")
    print(f"🏎️  STANDALONE AUTONOMOUS AGENT ACTIVE")
    print(f"Listening on: ws://{host}:{port}")
    print(f"Model: {agent.model_path}")
    print(f"Open http://127.0.0.1:8080/3d/ in your browser to see the car drive live!")
    print(f"===========================================================================\n")

    async with websockets.serve(handler, host, port, max_size=1024 * 1024):
        await asyncio.Future()  # run indefinitely


# =============================================================================
# Headless Terminal Autonomous Driving Runner
# =============================================================================
def run_headless_simulation(agent: AutonomousAgent, port: int = 8080, max_seconds: int = 35):
    """Launch headless browser and drive autonomously from the terminal."""
    from playwright.sync_api import sync_playwright

    print(f"[*] Starting Headless Terminal Autonomous Drive via Playwright...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, channel="chrome")
        page = browser.new_page()
        page.goto(f"http://127.0.0.1:{port}/3d/?collect=1")
        page.wait_for_function("() => Boolean(window.neuroDrive && window.track)", timeout=30000)

        # Reset vehicle to start line
        page.evaluate("window.neuroDrive.reset()")
        agent.reset()
        print("[✓] Simulation environment initialized at South Start Line (Z=76.0m).")

        t_start = time.time()
        step = 0
        parked = False

        print("\n" + "=" * 80)
        print(f"{'TIME':<6} | {'POS (X, Z)':<18} | {'SPEED':<10} | {'STEER':<8} | {'THROTTLE':<9} | {'STATUS'}")
        print("=" * 80)

        while time.time() - t_start < max_seconds:
            step += 1
            # 1. Grab raw RGB dashcam frame and telemetry from environment
            obs = page.evaluate("() => window.neuroDrive.observe()")
            raw_rgb = base64.b64decode(obs["rgbBase64"])
            speed = float(obs["state"]["speed"])

            # 2. Agent predicts action with ACT temporal ensembling & plant bridging
            steer, throttle, lat = agent.act(raw_rgb, current_speed=speed)

            # 3. Step third-party vehicle physics
            res = page.evaluate(f"""
                () => {{
                    const transition = window.neuroDrive.step([{steer}, {throttle}]);
                    const s = transition.state;
                    return {{
                        x: s.x.toFixed(2),
                        z: s.z.toFixed(2),
                        speed: s.speed.toFixed(2),
                        offroad: s.isOffroad,
                        garage: s.inParkingGarage
                    }};
                }}
            """)

            elapsed = time.time() - t_start
            pos_str = f"({res['x']}, {res['z']})"
            is_parked = res["garage"] and float(res["speed"]) <= 0.5 and float(res["z"]) <= -154.0
            status = "🏁 PARKED" if is_parked else ("🌿 OFF-ROAD" if res["offroad"] else "🛣️ ON-ROAD")

            if step % 15 == 0 or is_parked:
                print(f"{elapsed:04.1f}s  | {pos_str:<18} | {res['speed'] + ' m/s':<10} | {steer:+05.2f}    | {throttle:+05.2f}     | {status}")

            if is_parked:
                print("\n[✓] 🏆 Autonomous Agent successfully parked in the garage stall!")
                parked = True
                break

            time.sleep(0.045)  # match ~20 Hz control rate

        print("=" * 80)
        browser.close()
        return parked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Autonomous Driving Agent with ACT")
    parser.add_argument("--model", type=str, default="3d/model_act.onnx", help="Path to ONNX policy")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="WebSocket host")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--headless", action="store_true", help="Run automated headless terminal simulation")
    parser.add_argument("--web-port", type=int, default=8080, help="Port of running 3D web simulation")
    args = parser.parse_args()

    agent = AutonomousAgent(model_path=args.model)

    if args.headless:
        run_headless_simulation(agent, port=args.web_port)
    else:
        asyncio.run(run_websocket_server(agent, host=args.host, port=args.port))
