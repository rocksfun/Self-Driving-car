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
        decay_rate: float = 0.05,
        intent: str = "take right",
    ):
        self.chunk_horizon = chunk_horizon
        self.decay_rate = decay_rate

        # Language Intent State (VLA)
        self.current_intent = "take right"
        self.current_intent_id = 1  # 0: 'take left', 1: 'take right'
        self.has_language_input = False
        self.lang_input_name: Optional[str] = None
        self.set_intent(intent)

        # Temporal Ensembling FIFO buffer: stores [{'chunk': np.ndarray, 'age': int}]
        self.history: list[dict] = []
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_throttle: float = 0.0
        self.prev_steer: float = 0.0

        self.model_path: Optional[Path] = None
        self.model_name: str = ""
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = ""
        self.output_name: str = ""
        self.output_shape: list = []
        self.is_chunked: bool = False
        self.chunk_size: int = 1

        self.load_model(model_path)

    def load_model(self, target: str | Path | bytes, name: Optional[str] = None) -> dict:
        """Dynamically load or switch policy model at runtime."""
        if isinstance(target, (str, Path)):
            str_target = str(target).strip()
            if str_target == "act":
                candidates = [
                    Path("3d/model_act.onnx"),
                    Path("ACT_model/model_act.onnx"),
                ]
            elif str_target in ("baseline", "onnx", "cnn"):
                candidates = [
                    Path("3d/model.onnx"),
                    Path("model.onnx"),
                ]
            else:
                candidates = [
                    Path(str_target),
                    Path("3d") / str_target,
                    Path("ACT_model") / str_target,
                ]

            found_path = None
            for p in candidates:
                if p.exists():
                    found_path = p
                    break

            if not found_path:
                raise FileNotFoundError(f"Could not find model file for '{target}' (searched {[str(c) for c in candidates]})")

            self.model_path = found_path
            self.model_name = name or (
                "Action Chunking Policy (ACT)" if "act" in found_path.name.lower()
                else ("Baseline Vision CNN" if "model.onnx" in found_path.name.lower() else found_path.name)
            )
            print(f"[*] Loading model from file: {found_path}")
            session = ort.InferenceSession(str(found_path), providers=["CPUExecutionProvider"])
        elif isinstance(target, bytes):
            self.model_path = None
            self.model_name = name or "custom_model.onnx"
            print(f"[*] Loading custom model from {len(target)} bytes: {self.model_name}")
            session = ort.InferenceSession(target, providers=["CPUExecutionProvider"])
        else:
            raise TypeError(f"Invalid model target type: {type(target)}")

        self.session = session
        all_inputs = self.session.get_inputs()
        self.input_name = all_inputs[0].name
        self.input_names = [inp.name for inp in all_inputs]
        self.has_language_input = any("lang" in n.lower() or "intent" in n.lower() for n in self.input_names)
        self.lang_input_name = next((n for n in self.input_names if "lang" in n.lower() or "intent" in n.lower()), None)

        self.output_name = self.session.get_outputs()[0].name
        self.output_shape = self.session.get_outputs()[0].shape

        self.is_chunked = len(self.output_shape) == 3 and self.output_shape[1] > 1
        self.chunk_size = self.output_shape[1] if self.is_chunked else 1
        self.reset()

        policy_type = f"ACT Action Chunking (K={self.chunk_size})" if self.is_chunked else "Single-Step Baseline CNN"
        print(f" -> Policy Architecture: {policy_type}")
        print(f" -> Inputs: {[f'{inp.name} {inp.shape}' for inp in all_inputs]}")
        if self.has_language_input:
            print(f" -> VLA Conditioning: Language input enabled ({self.lang_input_name})")
        print(f" -> Output: {self.output_name} {self.output_shape}")
        print(f" -> Active Language Intent: '{self.current_intent}' (ID: {self.current_intent_id})")

        return self.get_info()

    def parse_intent(self, text: str) -> Optional[str]:
        """Parse natural language command using substring and intent mapping."""
        if not text or not isinstance(text, str):
            return None
        t = text.strip().lower()
        if "left" in t:
            return "take left"
        if "right" in t:
            return "take right"
        return None

    def set_intent(self, text: str) -> str:
        """Update current driving language intent from text input."""
        parsed = self.parse_intent(text)
        if parsed:
            self.current_intent = parsed
            self.current_intent_id = 0 if parsed == "take left" else 1
        return self.current_intent

    def get_info(self) -> dict:
        """Return model metadata for simulation UI."""
        return {
            "name": self.model_name,
            "filename": self.model_path.name if self.model_path else self.model_name,
            "isChunked": self.is_chunked,
            "chunkSize": self.chunk_size,
            "current_intent": self.current_intent,
            "intent_id": self.current_intent_id,
            "hasLanguageInput": self.has_language_input,
            "inputShape": [int(x) if isinstance(x, (int, np.integer)) else str(x) for x in self.session.get_inputs()[0].shape],
            "outputShape": [int(x) if isinstance(x, (int, np.integer)) else str(x) for x in self.output_shape],
        }

    def reset(self):
        """Reset internal agent memory and temporal ensembling queue for a new episode."""
        self.history.clear()
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_throttle = 0.0
        self.prev_steer = 0.0

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

    def act(
        self,
        raw_bytes: bytes,
        current_speed: Optional[float] = None,
        intent: Optional[str] = None,
    ) -> Tuple[float, float, float]:
        """Perform perception, model inference, temporal ensembling, and plant bridging.

        Args:
            raw_bytes: Raw 64x64 RGB or RGBA camera bytes.
            current_speed: Optional vehicle telemetry speed (m/s) for actuator envelope regulation.
            intent: Optional language intent string (e.g. 'take left' or 'take right').

        Returns:
            steering (float): Steer angle in [-1.0, 1.0]
            throttle (float): Throttle/brake in [-1.0, 1.0]
            latency_ms (float): Inference execution time in milliseconds
        """
        if intent:
            self.set_intent(intent)

        t0 = time.perf_counter()
        img_tensor = self.preprocess_image(raw_bytes)

        # 1. Model Forward Pass
        feed = {self.input_name: img_tensor}
        if self.has_language_input and self.lang_input_name:
            # Pass discrete language token [1]
            feed[self.lang_input_name] = np.array([self.current_intent_id], dtype=np.int64)

        results = self.session.run([self.output_name], feed)
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

        # 4. Plant Actuator Bridging — Continuous Zero-Jerk Cruise Governor
        # In stock car3d.js physics, any throttle > 0.05 accelerates at 24 m/s² with zero aerodynamic drag,
        # while throttle <= 0.05 engages friction at 12 m/s².
        # To eliminate bang-bang oscillation and longitudinal jerk:
        # 1) Smoothly taper positive cruising throttle between 11.5 m/s and 13.5 m/s using smoothstep
        # 2) Rate-limit throttle changes (|dThrottle/dt| <= 2.5 / s) so acceleration transitions are C^1 continuous
        if current_speed is not None and throttle > 0.0:
            if current_speed >= 13.5:
                throttle = 0.0
            elif current_speed > 11.5:
                # C^1 cubic smoothstep taper from 1.0 down to 0.0
                t = (current_speed - 11.5) / (13.5 - 11.5)
                scale = 1.0 - (3.0 * t * t - 2.0 * t * t * t)
                throttle = float(throttle * scale)

        # Slew-rate limiter on throttle to eliminate step shocks in acceleration
        max_rate = 2.5  # Max throttle change per second
        max_delta = max_rate * 0.05  # At 20 Hz control loop (dt ≈ 0.05s)
        if throttle > self.prev_throttle:
            throttle = min(throttle, self.prev_throttle + max_delta)
        elif throttle < self.prev_throttle:
            # Faster response for intentional braking (< 0), smooth for coasting
            decel_rate = 5.0 if throttle < 0.0 else max_rate
            throttle = max(throttle, self.prev_throttle - decel_rate * 0.05)

        # Slew-rate limiter on steering to eliminate pilot-induced yaw oscillation
        # (at 20 Hz, max_steer_rate = 1.8 / s allows full lock-to-lock [-1 to +1] in ~1.1s,
        # max_steer_delta = 1.8 * 0.05 = 0.09 per step)
        max_steer_rate = 1.8
        max_steer_delta = max_steer_rate * 0.05
        steer = float(np.clip(steer, self.prev_steer - max_steer_delta, self.prev_steer + max_steer_delta))
        self.prev_steer = steer

        self.prev_throttle = throttle
        return steer, throttle, dt_ms


# =============================================================================
# WebSocket Communication Bridge (ws://localhost:8765 / ws://127.0.0.1:8765)
# =============================================================================
async def run_websocket_server(agent: AutonomousAgent, host: str | None = "0.0.0.0", port: int = 8765):
    """Run an asynchronous WebSocket server bridging the agent with browser clients."""
    import websockets

    step_counter = 0

    async def handler(websocket):
        nonlocal step_counter
        client_addr = websocket.remote_address
        print(f"[+] Simulation client connected from: {client_addr}")
        agent.reset()

        try:
            await websocket.send(json.dumps({"type": "model_info", **agent.get_info()}))
        except Exception:
            pass

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
                        intent = data.get("intent")
                        steer, throttle, latency = agent.act(raw_bytes, current_speed=speed, intent=intent)

                        response = {
                            "type": "action",
                            "reqId": req_id,
                            "step": step_counter,
                            "steering": round(steer, 4),
                            "throttle": round(throttle, 4),
                            "latencyMs": round(latency, 2),
                            "intent": agent.current_intent,
                            "intentId": agent.current_intent_id,
                        }
                        await websocket.send(json.dumps(response))

                    elif msg_type == "set_intent":
                        raw_text = data.get("intent", "") or data.get("command", "") or data.get("rawText", "")
                        parsed = agent.set_intent(raw_text)
                        print(f"[Agent] 💬 Language intent updated: '{agent.current_intent}' (ID: {agent.current_intent_id}) [raw: '{raw_text}']")
                        await websocket.send(json.dumps({
                            "type": "intent_ack",
                            "status": "ok",
                            "intent": agent.current_intent,
                            "intentId": agent.current_intent_id,
                            "rawText": raw_text,
                        }))

                    elif msg_type == "reset":
                        agent.reset()
                        step_counter = 0
                        print(" -> Agent reset for new episode.")
                        await websocket.send(json.dumps({"type": "reset_ack", "status": "ok"}))

                    elif msg_type == "get_model_info":
                        info = agent.get_info()
                        await websocket.send(json.dumps({"type": "model_info", **info}))

                    elif msg_type == "load_model":
                        model_key = data.get("model", "act")
                        try:
                            info = agent.load_model(model_key)
                            step_counter = 0
                            print(f"[+] Switched model to: {info['name']}")
                            await websocket.send(json.dumps({"type": "model_loaded", "status": "ok", **info}))
                        except Exception as exc:
                            print(f"[!] Error loading model '{model_key}': {exc}")
                            await websocket.send(json.dumps({"type": "error", "message": str(exc)}))

                    elif msg_type == "upload_model":
                        model_name = data.get("name", "custom_model.onnx")
                        b64_str = data.get("modelBase64", "")
                        try:
                            model_bytes = base64.b64decode(b64_str)
                            # Save custom model to 3d directory
                            save_path = Path("3d") / model_name
                            save_path.write_bytes(model_bytes)
                            info = agent.load_model(model_bytes, name=model_name)
                            info["filename"] = model_name
                            step_counter = 0
                            print(f"[+] Successfully loaded uploaded model: {model_name}")
                            await websocket.send(json.dumps({"type": "model_loaded", "status": "ok", **info}))
                        except Exception as exc:
                            print(f"[!] Error loading uploaded model '{model_name}': {exc}")
                            await websocket.send(json.dumps({"type": "error", "message": str(exc)}))

                    elif msg_type == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))

        except websockets.exceptions.ConnectionClosed:
            print(f"[-] Simulation client disconnected: {client_addr}")
        except Exception as exc:
            print(f"[!] Error in agent connection: {exc}")

    # Ensure dual-stack binding so both localhost (IPv6 ::1) and 127.0.0.1 (IPv4) work
    if host in ("0.0.0.0", "localhost", None, "", "all", "*"):
        bind_host = ["0.0.0.0", "::"]
        display_host = "localhost / 127.0.0.1"
    else:
        bind_host = host
        display_host = host

    print(f"\n===========================================================================")
    print(f"🏎️  STANDALONE AUTONOMOUS AGENT ACTIVE")
    print(f"Listening on: ws://localhost:{port} & ws://127.0.0.1:{port}")
    print(f"Model: {agent.model_path}")
    print(f"Open http://localhost:8080/3d/ or http://127.0.0.1:8080/3d/ in your browser to see the car drive live!")
    print(f"===========================================================================\n")

    async with websockets.serve(handler, bind_host, port, max_size=1024 * 1024):
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
        print(f"[✓] Simulation environment initialized at South Start Line (Z=76.0m).")
        print(f"[*] Autonomous Agent active language intent: {agent.current_intent.upper()} (ID: {agent.current_intent_id})")

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
            steer, throttle, lat = agent.act(raw_rgb, current_speed=speed, intent=agent.current_intent)

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
    parser.add_argument("--intent", type=str, default="take right", help="Starting language intent command ('take left' or 'take right')")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="WebSocket host")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--headless", action="store_true", help="Run automated headless terminal simulation")
    parser.add_argument("--web-port", type=int, default=8080, help="Port of running 3D web simulation")
    args = parser.parse_args()

    agent = AutonomousAgent(model_path=args.model, intent=args.intent)

    if args.headless:
        run_headless_simulation(agent, port=args.web_port)
    else:
        asyncio.run(run_websocket_server(agent, host=args.host, port=args.port))
