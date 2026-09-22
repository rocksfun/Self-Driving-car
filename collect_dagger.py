#!/usr/bin/env python3
"""Collect DAgger demonstrations with persistent grass-drift bursts and unbiased roundabout paths.

Key Capabilities:
1. Robust Noise & Grass Recovery:
   - Injects correlated steering bursts (18-24 steps) that push the car across curbs onto the grass.
   - Captures extensive grass recovery demonstrations where the oracle steers back to the road.
2. Consistent Start Line Respawn:
   - Always respawns the vehicle at the South start line (z ≈ 76.0m) facing North.
3. Unbiased 50/50 Roundabout Coverage:
   - Alternates between right-side (counter-clockwise) and left-side (clockwise) roundabout routes.
"""

from __future__ import annotations

import argparse
import base64
import errno
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
import threading
import time
from typing import List

import numpy as np
import torch
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
FRAME_BYTES = 64 * 64 * 3


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


class FrameUploadError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class FrameUploadStore:
    """One bounded, single-use frame upload for the current episode."""

    def __init__(self, max_bytes):
        if max_bytes < 1:
            raise ValueError("Frame upload bound must be positive.")
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._path = None
        self._payload = None
        self._receiving = False
        self._error = None

    def expect_episode(self):
        with self._lock:
            if self._path is not None:
                raise RuntimeError("Previous episode frame upload was not consumed.")
            self._path = f"/__collection_frames/{secrets.token_urlsafe(32)}"
            self._payload = None
            self._receiving = False
            self._error = None
            return self._path

    def begin_upload(self, path, length):
        with self._lock:
            if path != self._path or self._path is None:
                raise FrameUploadError(404, "Unknown frame upload endpoint.")
            if self._receiving or self._payload is not None or self._error is not None:
                raise FrameUploadError(409, "This episode already received an upload.")
            if length < 0:
                raise FrameUploadError(400, "Invalid frame payload length.")
            if length > self.max_bytes:
                raise FrameUploadError(413, "Frame payload exceeds the episode bound.")
            self._receiving = True

    def complete_upload(self, path, payload, declared_length):
        with self._lock:
            if path != self._path or not self._receiving:
                raise FrameUploadError(409, "No matching frame upload is active.")
            if len(payload) != declared_length:
                self._error = "Incomplete frame payload."
                self._receiving = False
                raise FrameUploadError(400, self._error)
            self._payload = payload
            self._receiving = False

    def abort_upload(self, path, error):
        with self._lock:
            if path == self._path:
                self._error = str(error)
                self._receiving = False

    def take_episode(self, path, expected_bytes):
        with self._lock:
            if path != self._path or self._path is None:
                raise ValueError("Frame payload belongs to a different episode.")
            try:
                if self._error or self._receiving:
                    raise ValueError(self._error or "Frame upload has not completed.")
                if expected_bytes is None:
                    if self._payload is not None:
                        raise ValueError("Unexpected frames for a rejected episode.")
                    return None
                if self._payload is None or len(self._payload) != expected_bytes:
                    raise ValueError("Camera payload length does not match the recorded frame count.")
                return self._payload
            finally:
                self._path = None
                self._payload = None


class FrameUploadHandler(QuietHandler):
    def __init__(self, *args, frame_store, **kwargs):
        self.frame_store = frame_store
        super().__init__(*args, **kwargs)

    def do_POST(self):
        self.close_connection = True
        if self.headers.get("Transfer-Encoding"):
            self.send_error(400, "A Content-Length frame payload is required.")
            return
        try:
            header = self.headers.get("Content-Length")
            if header is None:
                raise FrameUploadError(411, "Frame payload length is required.")
            try:
                length = int(header)
            except ValueError:
                raise FrameUploadError(400, "Invalid frame payload length.") from None
            self.frame_store.begin_upload(self.path, length)
        except FrameUploadError as exc:
            self.send_error(exc.status, str(exc))
            return
        try:
            self.connection.settimeout(120)
            payload = self.rfile.read(length)
            self.frame_store.complete_upload(self.path, payload, length)
        except (OSError, FrameUploadError) as exc:
            self.frame_store.abort_upload(self.path, exc)
            self.send_error(getattr(exc, "status", 408), str(exc))
            return
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()


def start_frame_server(port, max_frame_bytes, directory=ROOT):
    """Serve assets and frame uploads on one private, same-origin loopback port."""
    store = FrameUploadStore(max_frame_bytes)
    handler = partial(FrameUploadHandler, directory=str(directory), frame_store=store)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, store


SEED_BROWSER_JS = """seed => {
    let state = seed >>> 0;
    Math.random = () => {
        state = (state + 0x6D2B79F5) | 0;
        let value = Math.imul(state ^ (state >>> 15), 1 | state);
        value ^= value + Math.imul(value ^ (value >>> 7), 61 | value);
        return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
}"""

RENDERER_INFO_JS = """() => {
    const renderer = window.dashcamRenderer;
    const gl = renderer.getContext();
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    return {
        vendor: gl.getParameter(ext ? ext.UNMASKED_VENDOR_WEBGL : gl.VENDOR),
        renderer: gl.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER),
        version: gl.getParameter(gl.VERSION),
        width: gl.drawingBufferWidth, height: gl.drawingBufferHeight,
        async_readback: typeof gl.fenceSync === 'function' && typeof gl.getBufferSubData === 'function'
    };
}"""

COLLECT_EPISODE_JS = """async ({ branch, maxSteps, frameStride, denseRecovery, ouThetaSteer, ouThetaThrottle, noiseSteerStd, noiseThrottleStd, enableGrassBursts, maxAttempts, readback, uploadUrl }) => {
    const env = window.neuroDrive;
    const rolloutStart = performance.now();
    const stride = Number(frameStride ?? 4);
    const useDenseRecovery = (denseRecovery !== false);
    const dt = 0.05;
    const thetaS = Number(ouThetaSteer ?? 0.20);
    const thetaT = Number(ouThetaThrottle ?? 0.25);
    const sigmaS = Number(noiseSteerStd ?? 0.35);
    const sigmaT = Number(noiseThrottleStd ?? 0.10);
    const maxRolloutAttempts = Number(maxAttempts ?? 8);

    function gaussianRandom(mean, std) {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return mean + std * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    let attempts = 0;
    let validTrajectory = null;
    let allSimSteps = 0;
    const attemptOutcomes = [];

    // =========================================================================
    // PHASE 1: Validate noisy rollouts before rendering observations.
    // Physics, oracle policy, random draws and sample selection stay unchanged.
    // =========================================================================
    while (attempts < maxRolloutAttempts) {
        attempts++;
        window.collectionOracle = new NeuroDriveOracle(window.track, branch);
        window.ouNoise = { steer: 0.0, throttle: 0.0 };
        const s = window.collectionOracle.spawnStart(Math.random);
        env.reset(s.state, { render: false });

        const trajStates = [];
        const trajCleanActions = [];
        const trajExecutedActions = [];
        const trajSampleSteps = [];

        let inBurst = false;
        let burstRemaining = 0;
        let burstSteer = 0.0;
        let cooldown = 40;
        const burstTrigger1 = Math.floor(24 + Math.random() * 12);
        const burstTrigger2 = Math.floor(200 + Math.random() * 30);

        let offroadFrames = 0;
        let epOutcome = "timeout";
        let actualSimSteps = 0;

        for (let step = 1; step <= maxSteps; step++) {
            actualSimSteps = step;
            allSimSteps++;
            const state = env.getState();
            const cte = window.collectionOracle.cte(state);
            const isOffroad = Boolean(state.isOffroad || Math.abs(cte) > 5.5);

            // 1. Clean Oracle target action
            const cleanAction = window.collectionOracle.action(state);

            // 2. Safety check: disable noise in garage zone
            const inGarageZone = state.z <= -140 || window.collectionOracle.isSuccess(state);

            cooldown--;
            const isScheduled = (step === burstTrigger1 || step === burstTrigger2);
            const isRandom = (cooldown <= 0 && Math.random() < 0.02);
            if (enableGrassBursts && !inBurst && (isScheduled || isRandom)) {
                inBurst = true;
                burstRemaining = Math.floor(20 + Math.random() * 6);
                const dir = Math.random() < 0.5 ? -1.0 : 1.0;
                burstSteer = dir * (0.78 + Math.random() * 0.12);
            }
            if (inBurst) {
                burstRemaining--;
                if (burstRemaining <= 0) {
                    inBurst = false;
                    cooldown = 75;
                }
            }

            // OU noise update
            if (!inGarageZone) {
                const dW_s = gaussianRandom(0, 1.0);
                const dW_t = gaussianRandom(0, 1.0);
                window.ouNoise.steer += thetaS * (0.0 - window.ouNoise.steer) * dt + sigmaS * Math.sqrt(dt) * dW_s;
                window.ouNoise.throttle += thetaT * (0.0 - window.ouNoise.throttle) * dt + sigmaT * Math.sqrt(dt) * dW_t;
                window.ouNoise.steer = Math.max(-0.60, Math.min(0.60, window.ouNoise.steer));
                window.ouNoise.throttle = Math.max(-0.25, Math.min(0.25, window.ouNoise.throttle));
            } else {
                window.ouNoise.steer *= 0.75;
                window.ouNoise.throttle *= 0.75;
            }

            const executed = [cleanAction[0], cleanAction[1]];
            if (!inGarageZone) {
                executed[0] = Math.max(-1.0, Math.min(1.0, cleanAction[0] + window.ouNoise.steer));
                executed[1] = Math.max(-1.0, Math.min(1.0, cleanAction[1] + window.ouNoise.throttle));
                if (inBurst) {
                    executed[0] = Math.max(-1.0, Math.min(1.0, executed[0] + burstSteer));
                    executed[1] = Math.max(0.35, Math.min(0.70, executed[1]));
                }
            }

            const isHighCTE = Math.abs(cte) > 1.8;
            const isStride = (step % stride === 0);
            const shouldRecord = (useDenseRecovery && (isOffroad || inBurst || isHighCTE)) || isStride;

            if (shouldRecord) {
                trajStates.push({
                    x: state.x, y: state.y, z: state.z,
                    heading: state.heading, speed: state.speed, steerAngle: state.steerAngle
                });
                trajCleanActions.push(cleanAction);
                trajExecutedActions.push(executed);
                trajSampleSteps.push(step);
            }

            // Step vehicle physics purely via math (no WebGL render!)
            const transition = env.stepPhysics ? env.stepPhysics(executed) : env.step(executed);
            const nextState = transition.state;

            if (transition.events.offroad) {
                offroadFrames++;
                if (inBurst) {
                    inBurst = false;
                    cooldown = 80;
                }
            }

            const outcome = window.collectionOracle.terminalOutcome(nextState);
            if (outcome.success) {
                epOutcome = "success";
                break;
            } else if (outcome.crashed) {
                epOutcome = `crash (${outcome.reason || 'collision'})`;
                break;
            }
        }

        attemptOutcomes.push({ outcome: epOutcome, simSteps: actualSimSteps, offroadFrames });
        // Keep the existing success-only acceptance policy.
        if (epOutcome === "success") {
            validTrajectory = {
                states: trajStates,
                cleanActions: trajCleanActions,
                executedActions: trajExecutedActions,
                sampleSteps: trajSampleSteps,
                offroadFrames: offroadFrames,
                simSteps: actualSimSteps,
                outcome: epOutcome,
                attempts: attempts
            };
            break;
        }
    }

    const rolloutMs = performance.now() - rolloutStart;
    if (!validTrajectory) {
        return { steps: 0, simSteps: 0, offroadFrames: 0, outcome: "failed_attempts", attempts,
            allSimSteps, attemptOutcomes, timingsMs: { rollout: rolloutMs, capture: 0, encode: 0, upload: 0 } };
    }

    // =========================================================================
    // PHASE 2: Selective High-Speed Camera Baking
    // Steps through and renders ONLY the validated winning trajectory!
    // =========================================================================
    const FRAME_BYTES = 64 * 64 * 3;
    const states = validTrajectory.states;
    const bakedCount = states.length;
    const allFrames = new Uint8Array(bakedCount * FRAME_BYTES);
    const captureStart = performance.now();
    if (readback !== 'sync' && env.observeRGBBatch) {
        await env.observeRGBBatch(states, allFrames);
    } else {
        if (!env.observeRGBDirect) throw new Error('Collection requires direct camera readback.');
        for (let i = 0; i < bakedCount; i++) {
            env.setPose(states[i]);
            env.observeRGBDirect(allFrames, i * FRAME_BYTES);
        }
    }
    const captureMs = performance.now() - captureStart;
    let framesBase64 = null, encodeMs = 0, uploadMs = 0;
    if (uploadUrl) {
        const uploadStart = performance.now();
        const response = await fetch(uploadUrl, {
            method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: allFrames
        });
        if (!response.ok) throw new Error(`Frame upload failed: HTTP ${response.status}`);
        uploadMs = performance.now() - uploadStart;
    } else {
        const encodeStart = performance.now();
        const blob = new Blob([allFrames], { type: 'application/octet-stream' });
        framesBase64 = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result.split(',')[1]);
            reader.onerror = () => reject(reader.error || new Error('Frame encoding failed.'));
            reader.readAsDataURL(blob);
        });
        encodeMs = performance.now() - encodeStart;
    }

    return {
        framesBase64: framesBase64,
        frameBytes: allFrames.byteLength,
        cleanActions: validTrajectory.cleanActions,
        executedActions: validTrajectory.executedActions,
        states: states.map(st => [st.x, st.y, st.z, st.heading, st.speed, st.steerAngle]),
        sampleSteps: validTrajectory.sampleSteps,
        steps: bakedCount,
        simSteps: validTrajectory.simSteps,
        offroadFrames: validTrajectory.offroadFrames,
        outcome: validTrajectory.outcome,
        attempts: attempts,
        allSimSteps, attemptOutcomes,
        timingsMs: { rollout: rolloutMs, capture: captureMs, encode: encodeMs, upload: uploadMs }
    };
}"""


def run_collection(
    episodes: int = 30,
    max_steps_per_episode: int = 1000,
    frame_stride: int = 4,
    dense_recovery: bool = True,
    ou_theta_steer: float = 0.20,
    ou_theta_throttle: float = 0.25,
    noise_steer: float = 0.35,
    noise_throttle: float = 0.10,
    enable_grass_bursts: bool = True,
    output_path: str = "dataset_dagger.pt",
    seed: int = 42,
    port: int = 8080,
    max_attempts: int = 8,
    readback: str = "batch",
):
    if episodes < 1 or max_steps_per_episode < 1 or frame_stride < 1 or max_attempts < 1:
        raise ValueError("Episodes, max steps, frame stride and max attempts must be positive.")
    if readback not in ("batch", "sync"):
        raise ValueError("Readback must be 'batch' or 'sync'.")
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must be an unsigned 32-bit integer.")
    for name, value in (("ou_theta_steer", ou_theta_steer), ("ou_theta_throttle", ou_theta_throttle),
                        ("noise_steer", noise_steer), ("noise_throttle", noise_throttle)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    print("=" * 75)
    print("🏎️  DAGGER COLLECTOR (DECOUPLED 5Hz VISION + 20Hz OU NOISE + RECOVERY)")
    print(f"Episodes: {episodes} (Always starting from Start Line at South Entrance)")
    print(f"Roundabout Paths: 50% Left (Clockwise) / 50% Right (Counter-Clockwise)")
    print(f"Sampling: {20 / frame_stride:g} Hz Baseline (Stride {frame_stride}) | Dense Grass/CTE Recovery: {dense_recovery}")
    print(f"OU Noise: Steer(θ={ou_theta_steer}, σ={noise_steer}) | Throttle(θ={ou_theta_throttle}, σ={noise_throttle})")
    print(f"Grass Bursts: {enable_grass_bursts} | Target Output: {output_path}")
    print("=" * 75)

    np.random.seed(seed)
    torch.manual_seed(seed)

    collected_images: List[torch.Tensor] = []
    collected_actions: List[List[float]] = []
    collected_executed: List[List[float]] = []
    collected_states: List[List[float]] = []
    collected_sample_steps: List[int] = []
    episode_offsets = [0]
    episode_records = []
    branch_episodes = {"right": 0, "left": 0}
    branch_frames = {"right": 0, "left": 0}
    phase_seconds = {"rollout": 0.0, "capture": 0.0, "encode": 0.0, "transport": 0.0, "decode": 0.0}

    success_count = 0
    collision_count = 0
    timeout_count = 0
    total_offroad_frames = 0
    total_sim_steps = 0
    total_attempted_sim_steps = 0

    server = None
    server_thread = None
    import urllib.request

    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/3d/", timeout=1)
        active_url = f"http://127.0.0.1:{port}/3d/?collect=1"
        print(f"[*] Connected to active server at http://127.0.0.1:{port}")
    except Exception:
        import threading

        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(ROOT)))
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        active_url = f"http://127.0.0.1:{server.server_port}/3d/?collect=1"
        print(f"[*] Spawned fallback server at {active_url}")

    try:
        with sync_playwright() as p:
            options = {
                "headless": True,
                "args": [
                    "--enable-gpu",
                    "--enable-gpu-rasterization",
                    "--enable-zero-copy",
                    "--ignore-gpu-blocklist",
                ],
            }
            chrome_path = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
            if chrome_path.exists():
                options["channel"] = "chrome"

            browser = p.chromium.launch(**options)
            try:
                page = browser.new_page(viewport={"width": 960, "height": 720})
                page.route("**/onnxruntime-web@*/**", lambda route: route.abort())
                # Seed scene construction separately from each episode. The old
                # --seed only seeded Python, leaving browser trajectories random.
                page.add_init_script(f"({SEED_BROWSER_JS})({seed})")

                print(f"[*] Navigating to {active_url}...")
                page.goto(active_url, wait_until="load")
                page.wait_for_function(
                    "() => Boolean(window.neuroDrive && window.NeuroDriveOracle && window.track)",
                    timeout=30000,
                )
                renderer_info = page.evaluate(RENDERER_INFO_JS)
                print(f"[*] WebGL: {renderer_info['renderer']} | Readback: {readback}")
                if any(name in renderer_info["renderer"].lower() for name in ("swiftshader", "llvmpipe", "software")):
                    print("[!] Software WebGL renderer detected; camera capture may be slow.")

                t_start = time.perf_counter()
                global_step = 0

                for ep in range(1, episodes + 1):
                    # Alternate 50/50 between Right and Left roundabout branches
                    branch = "left" if (ep % 2 == 0) else "right"
                    episode_seed = (seed + (ep - 1) * 1009) & 0xFFFFFFFF
                    page.evaluate(SEED_BROWSER_JS, episode_seed)
                    t_ep_start = time.perf_counter()

                    ep_data = page.evaluate(
                        COLLECT_EPISODE_JS,
                        {
                            "branch": branch,
                            "maxSteps": max_steps_per_episode,
                            "frameStride": frame_stride,
                            "denseRecovery": dense_recovery,
                            "ouThetaSteer": ou_theta_steer,
                            "ouThetaThrottle": ou_theta_throttle,
                            "noiseSteerStd": noise_steer,
                            "noiseThrottleStd": noise_throttle,
                            "enableGrassBursts": enable_grass_bursts,
                            "maxAttempts": max_attempts,
                            "readback": readback,
                        },
                    )
                    evaluate_time = time.perf_counter() - t_ep_start
                    timings = ep_data["timingsMs"]
                    for phase in ("rollout", "capture", "encode"):
                        phase_seconds[phase] += timings[phase] / 1000
                    # Includes protocol serialization and scheduling, not just copying.
                    transport_time = max(0.0, evaluate_time - sum(timings.values()) / 1000)
                    phase_seconds["transport"] += transport_time
                    decode_start = time.perf_counter()

                    ep_steps = ep_data["steps"]
                    ep_sim_steps = ep_data.get("simSteps", ep_steps)
                    ep_offroad = ep_data["offroadFrames"]
                    ep_outcome = ep_data["outcome"]
                    total_sim_steps += ep_sim_steps
                    total_attempted_sim_steps += ep_data["allSimSteps"]
                    start_index = global_step

                    if ep_steps > 0:
                        raw_bytes = base64.b64decode(ep_data["framesBase64"])
                        if len(raw_bytes) != ep_steps * 64 * 64 * 3:
                            raise ValueError("Camera payload length does not match the recorded frame count.")
                        if any(len(ep_data[key]) != ep_steps for key in ("cleanActions", "executedActions", "states", "sampleSteps")):
                            raise ValueError("Camera frames, labels and sample steps are misaligned.")
                        img_array = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((ep_steps, 64, 64, 3))
                        img_tensor = torch.from_numpy(img_array.copy()).permute(0, 3, 1, 2)
                        collected_images.append(img_tensor)

                        collected_actions.extend(ep_data["cleanActions"])
                        collected_executed.extend(ep_data["executedActions"])
                        collected_states.extend(ep_data["states"])
                        collected_sample_steps.extend(ep_data["sampleSteps"])
                        global_step += ep_steps
                        total_offroad_frames += ep_offroad
                        episode_offsets.append(global_step)
                        branch_episodes[branch] += 1
                        branch_frames[branch] += ep_steps

                    if "success" in ep_outcome:
                        success_count += 1
                    elif "crash" in ep_outcome:
                        collision_count += 1
                    else:
                        timeout_count += 1

                    decode_time = time.perf_counter() - decode_start
                    phase_seconds["decode"] += decode_time
                    ep_time = time.perf_counter() - t_ep_start
                    sim_fps = ep_sim_steps / max(0.001, ep_time)
                    ep_attempts = ep_data.get("attempts", 1)
                    episode_records.append({
                        "episode": ep, "branch": branch, "seed": episode_seed,
                        "start": start_index, "end": global_step, "outcome": ep_outcome,
                        "sim_steps": ep_sim_steps, "attempted_sim_steps": ep_data["allSimSteps"],
                        "offroad_steps": ep_offroad, "attempts": ep_attempts,
                        "attempt_outcomes": ep_data["attemptOutcomes"],
                        "timings_ms": timings, "transport_sec": transport_time,
                        "decode_sec": decode_time, "duration_sec": ep_time,
                    })
                    print(
                        f"Episode {ep:02d}/{episodes:02d} | Branch: {branch.upper():5s} | "
                        f"Frames: {ep_steps:03d}/{ep_sim_steps:03d} | Off-road: {ep_offroad:02d} | "
                        f"Attempts: {ep_attempts} | Outcome: {ep_outcome:10s} | Speed: {sim_fps:5.1f} Accepted-Sim-FPS ({ep_time:.2f}s)\n"
                        f"  Rollout {timings['rollout'] / 1000:.3f}s | Capture {timings['capture'] / 1000:.3f}s | "
                        f"Encode {timings['encode'] / 1000:.3f}s | Transfer {transport_time:.3f}s | Decode {decode_time:.3f}s"
                    )

                total_time = time.perf_counter() - t_start
                overall_fps = global_step / max(0.001, total_time)
                overall_sim_fps = total_sim_steps / max(0.001, total_time)

                # Save dataset
                print(f"\n[*] Serializing {global_step} transitions to {output_path}...")
                save_start = time.perf_counter()
                dataset_payload = {
                    "images": torch.cat(collected_images, dim=0) if collected_images else torch.empty((0, 3, 64, 64), dtype=torch.uint8),
                    "actions": torch.tensor(collected_actions, dtype=torch.float32).reshape(-1, 2),
                    "executed_actions": torch.tensor(collected_executed, dtype=torch.float32).reshape(-1, 2),
                    "states": torch.tensor(collected_states, dtype=torch.float32).reshape(-1, 6),
                    "sample_steps": torch.tensor(collected_sample_steps, dtype=torch.int64),
                    "episode_offsets": torch.tensor(episode_offsets, dtype=torch.int64),
                    "episode_records": episode_records,
                    "metadata": {
                        "episodes": episodes,
                        "total_steps": global_step,
                        "total_sim_steps": total_sim_steps,
                        "total_attempted_sim_steps": total_attempted_sim_steps,
                        "total_offroad_frames": total_offroad_frames,
                        "successes": success_count,
                        "collisions": collision_count,
                        "timeouts": timeout_count,
                        "collection_fps": overall_fps,
                        "simulation_fps": overall_sim_fps,
                        "attempted_simulation_fps": total_attempted_sim_steps / max(0.001, total_time),
                        "duration_sec": total_time,
                        "phase_seconds": phase_seconds,
                        "renderer": renderer_info,
                        "seed": seed,
                        "episode_seed_rule": "(seed + (episode - 1) * 1009) mod 2^32",
                        "sample_step_convention": "1-based action tick; observation time = (sample_step - 1) * 0.05 seconds",
                        "readback": readback,
                        "frame_stride": frame_stride,
                        "dense_recovery": dense_recovery,
                        "roundabout_distribution": "alternating attempted episodes: right, left",
                        "saved_episodes_by_branch": branch_episodes,
                        "saved_frames_by_branch": branch_frames,
                        "collection_config": {
                            "max_steps": max_steps_per_episode, "max_attempts": max_attempts,
                            "ou_theta_steer": ou_theta_steer, "ou_theta_throttle": ou_theta_throttle,
                            "noise_steer": noise_steer, "noise_throttle": noise_throttle,
                            "enable_grass_bursts": enable_grass_bursts,
                        },
                    },
                }

                out_file = Path(output_path)
                out_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dataset_payload, out_file)
                save_time = time.perf_counter() - save_start

                print(f"\n[✓] Dataset successfully saved to: {out_file.resolve()}")
                print(f" -> Total Saved Frames: {global_step} (Simulated {total_sim_steps} physics ticks)")
                print(f" -> Total Off-Road (Grass Recovery) Frames: {total_offroad_frames}")
                print(f" -> Success Rate: {success_count}/{episodes} ({success_count / episodes * 100:.1f}%)")
                print(f" -> Collection: {overall_fps:.1f} saved frames/s | {overall_sim_fps:.1f} accepted simulation ticks/s")
                print(f" -> Attempted physics ticks (including rejected runs): {total_attempted_sim_steps}")
                print(f" -> Saved episodes by branch: {branch_episodes} | Saved frames: {branch_frames}")
                print(f" -> Collection {total_time:.2f}s | Serialization {save_time:.2f}s")
            finally:
                browser.close()
    finally:
        if server:
            server.shutdown()
            server.server_close()
            if server_thread:
                server_thread.join(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect DAgger data with OU smooth noise, grass recovery & unbiased roundabout paths")
    parser.add_argument("--episodes", type=int, default=30, help="Number of episodes (even number recommended)")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--frame-stride", type=int, default=4, help="Sampling stride for routine driving (default 4 = 5 Hz)")
    parser.add_argument("--no-dense-recovery", action="store_true", help="Disable dense sampling during grass recovery")
    parser.add_argument("--ou-theta-steer", type=float, default=0.20, help="OU mean-reversion rate theta for steering")
    parser.add_argument("--ou-theta-throttle", type=float, default=0.25, help="OU mean-reversion rate theta for throttle")
    parser.add_argument("--noise-steer", type=float, default=0.35, help="OU volatility sigma for steering drift")
    parser.add_argument("--noise-throttle", type=float, default=0.10, help="OU volatility sigma for throttle drift")
    parser.add_argument("--no-grass-bursts", action="store_true", help="Disable persistent off-road grass bursts")
    parser.add_argument("--output", type=str, default="dataset_dagger.pt", help="Output .pt dataset file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--port", type=int, default=8080, help="Port of running 3D server")
    parser.add_argument("--max-attempts", type=int, default=16, help="Maximum noisy rollout attempts per episode (default 16)")
    parser.add_argument("--readback", choices=("batch", "sync"), default="batch", help="Camera readback mode; sync is available for comparisons")
    args = parser.parse_args()

    run_collection(
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        frame_stride=args.frame_stride,
        dense_recovery=not args.no_dense_recovery,
        ou_theta_steer=args.ou_theta_steer,
        ou_theta_throttle=args.ou_theta_throttle,
        noise_steer=args.noise_steer,
        noise_throttle=args.noise_throttle,
        enable_grass_bursts=not args.no_grass_bursts,
        output_path=args.output,
        seed=args.seed,
        port=args.port,
        max_attempts=args.max_attempts,
        readback=args.readback,
    )
