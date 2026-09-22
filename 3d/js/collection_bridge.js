// Shared control clock and browser collection API. No substitute renderer or
// vehicle equations: real actions and oracle lookahead call Car3D.update alike.
(function (root) {
  'use strict';

  const CONTROL_DT = 0.05;
  const SUBSTEPS = 3;
  const VERSION = 'neurodrive-browser-v1';

  function normalizeAction(action) {
    const steering = Array.isArray(action) ? action[0] : action.steering;
    const throttle = Array.isArray(action) ? action[1] : action.throttle;
    if (!Number.isFinite(steering) || !Number.isFinite(throttle)) {
      throw new TypeError('Action must contain finite steering and throttle values.');
    }
    return {
      isAuto: true, isDriving: true,
      steering: Math.max(-1, Math.min(1, steering)),
      throttle: Math.max(-1, Math.min(1, throttle)),
      allowReverse: Boolean(action.allowReverse)
    };
  }

  function advance(vehicle, track, action) {
    const input = normalizeAction(action);
    let collision = false;
    let deadEnd = false;
    let offroad = false;
    let offroadDuration = 0;
    for (let index = 0; index < SUBSTEPS; index++) {
      vehicle.update(input, CONTROL_DT / SUBSTEPS, track);
      collision = collision || vehicle.hitWall || vehicle.hitDeadEnd;
      deadEnd = deadEnd || vehicle.hitDeadEnd;
      offroad = offroad || vehicle.isOffroad;
      if (vehicle.isOffroad) offroadDuration += CONTROL_DT / SUBSTEPS;
    }
    // Do not lose a collision on substep 1 when substeps 2 and 3 rebound clear.
    vehicle.hitWall = collision;
    vehicle.hitDeadEnd = deadEnd;
    return { collision, offroad, offroadDuration };
  }

  function create({ car, track, obsCanvas, dashcamRenderer, renderObservation, onReset }) {
    // The ghost has the same implementation and track, in an unrendered scene.
    // Its pose/steering are restored for every rollout; it never moves the real car.
    const ghost = new Car3D(new THREE.Scene(), track);
    let steps = 0;
    let collisions = 0;
    let offroadSteps = 0;
    let offroadDuration = 0;
    const tempRgba = new Uint8Array(64 * 64 * 4);

    function getState() { return car.getState(); }

    function getConfig() {
      return {
        version: VERSION,
        control_dt: CONTROL_DT,
        substeps: SUBSTEPS,
        physics_dt: CONTROL_DT / SUBSTEPS,
        observation_shape: [64, 64, 3],
        observation_dtype: 'uint8',
        observation_layout: 'HWC',
        renderer: 'threejs-dashcam',
        physics: 'Car3D',
        action_order: ['steering', 'throttle'],
        negative_throttle: 'brake_to_stop',
        camera: { width: dashcamRenderer ? dashcamRenderer.getContext().drawingBufferWidth : 64,
          height: dashcamRenderer ? dashcamRenderer.getContext().drawingBufferHeight : 64,
          fov_y_degrees: 65, hood_offset: 1.6, height_above_car: 0.92 },
        alignment: 'observation_and_state_before_action',
        success: 'NeuroDriveOracle.isSuccess',
        car: {
          maxSpeed: car.maxSpeed, maxReverseSpeed: car.maxReverseSpeed,
          acceleration: car.acceleration, braking: car.braking,
          friction: car.friction, offroadFriction: car.offroadFriction,
          turnSpeed: car.turnSpeed
        }
      };
    }

    function reset(state = {}, { render = true } = {}) {
      car.reset(state);
      steps = 0;
      collisions = 0;
      offroadSteps = 0;
      offroadDuration = 0;
      if (onReset) onReset();
      if (render) renderObservation();
      return getState();
    }

    function observe() {
      renderObservation();
      const rgba = obsCanvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, 64, 64).data;
      const rgb = new Uint8Array(64 * 64 * 3);
      for (let pixel = 0; pixel < 64 * 64; pixel++) {
        rgb[pixel * 3] = rgba[pixel * 4];
        rgb[pixel * 3 + 1] = rgba[pixel * 4 + 1];
        rgb[pixel * 3 + 2] = rgba[pixel * 4 + 2];
      }
      return {
        rgbBase64: btoa(String.fromCharCode(...rgb)),
        shape: [64, 64, 3], dtype: 'uint8',
        state: getState(), step: steps, simulationTime: steps * CONTROL_DT
      };
    }

    function observeRGB(targetBuffer, offset = 0) {
      renderObservation();
      const rgba = obsCanvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, 64, 64).data;
      const target = targetBuffer || new Uint8Array(64 * 64 * 3);
      let outIdx = offset;
      for (let pixel = 0; pixel < 64 * 64; pixel++) {
        const inIdx = pixel * 4;
        target[outIdx++] = rgba[inIdx];
        target[outIdx++] = rgba[inIdx + 1];
        target[outIdx++] = rgba[inIdx + 2];
      }
      return {
        rgb: target,
        state: getState(),
        step: steps,
        simulationTime: steps * CONTROL_DT
      };
    }

    function observeRGBDirect(targetBuffer, offset = 0) {
      renderObservation();
      const target = targetBuffer || new Uint8Array(64 * 64 * 3);

      if (dashcamRenderer) {
        const gl = dashcamRenderer.getContext();
        gl.readPixels(0, 0, 64, 64, gl.RGBA, gl.UNSIGNED_BYTE, tempRgba);

        // Flip vertically (WebGL origin is bottom-left)
        let outIdx = offset;
        for (let row = 63; row >= 0; row--) {
          const rowStart = row * 64 * 4;
          for (let col = 0; col < 64; col++) {
            const inIdx = rowStart + col * 4;
            target[outIdx++] = tempRgba[inIdx];
            target[outIdx++] = tempRgba[inIdx + 1];
            target[outIdx++] = tempRgba[inIdx + 2];
          }
        }
      } else {
        const rgba = obsCanvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, 64, 64).data;
        let outIdx = offset;
        for (let pixel = 0; pixel < 64 * 64; pixel++) {
          const inIdx = pixel * 4;
          target[outIdx++] = rgba[inIdx];
          target[outIdx++] = rgba[inIdx + 1];
          target[outIdx++] = rgba[inIdx + 2];
        }
      }

      return {
        rgb: target,
        state: getState(),
        step: steps,
        simulationTime: steps * CONTROL_DT
      };
    }

    async function observeRGBBatch(states, targetBuffer, batchSize = 32) {
      const FRAME_BYTES = 64 * 64 * 3;
      const RGBA_BYTES = 64 * 64 * 4;
      if (!Array.isArray(states)) throw new TypeError('States must be an array.');
      if (!Number.isInteger(batchSize) || batchSize < 1 || batchSize > 256) {
        throw new RangeError('Readback batch size must be an integer from 1 to 256.');
      }
      const target = targetBuffer || new Uint8Array(states.length * FRAME_BYTES);
      if (!(target instanceof Uint8Array) || target.length < states.length * FRAME_BYTES) {
        throw new RangeError('RGB target must hold every requested 64x64 frame.');
      }
      for (const state of states) {
        if (!state || !['x', 'y', 'z', 'heading'].every(key => Number.isFinite(state[key])) ||
            (state.speed !== undefined && !Number.isFinite(state.speed)) ||
            (state.steerAngle !== undefined && !Number.isFinite(state.steerAngle))) {
          throw new TypeError('Every capture pose must contain finite coordinates and heading.');
        }
      }
      if (!states.length) return { rgb: target, frames: 0 };

      const gl = dashcamRenderer && dashcamRenderer.getContext();
      const supportsAsyncReadback = gl && typeof gl.fenceSync === 'function' &&
        typeof gl.getBufferSubData === 'function' && gl.PIXEL_PACK_BUFFER !== undefined;
      if (!supportsAsyncReadback) {
        for (let index = 0; index < states.length; index++) {
          setPose(states[index]);
          observeRGBDirect(target, index * FRAME_BYTES);
        }
        return { rgb: target, frames: states.length };
      }

      const capacity = Math.min(batchSize, states.length);
      const rgba = new Uint8Array(capacity * RGBA_BYTES);
      const buffer = gl.createBuffer();
      if (!buffer) throw new Error('Unable to allocate the camera readback buffer.');

      // Restore raw GL bindings before yielding so Three.js and other callers
      // cannot inherit a pixel-pack buffer from this collection operation.
      function withBuffer(callback) {
        const previous = gl.getParameter(gl.PIXEL_PACK_BUFFER_BINDING);
        try {
          gl.bindBuffer(gl.PIXEL_PACK_BUFFER, buffer);
          return callback();
        } finally {
          gl.bindBuffer(gl.PIXEL_PACK_BUFFER, previous);
        }
      }

      async function waitForReadback(sync) {
        const deadline = performance.now() + 60000;
        // A new WebGL fence cannot signal until the browser's event loop runs.
        // A resolved Promise alone would only yield to the microtask queue.
        await new Promise(resolve => setTimeout(resolve, 0));
        while (true) {
          if (gl.isContextLost()) throw new Error('Camera WebGL context was lost during readback.');
          const status = gl.clientWaitSync(sync, 0, 0);
          if (status === gl.ALREADY_SIGNALED || status === gl.CONDITION_SATISFIED) return;
          if (status === gl.WAIT_FAILED) throw new Error('Camera readback fence failed.');
          if (performance.now() > deadline) throw new Error('Camera readback timed out.');
          await new Promise(resolve => setTimeout(resolve, 0));
        }
      }

      try {
        withBuffer(() => gl.bufferData(gl.PIXEL_PACK_BUFFER, rgba.byteLength, gl.STREAM_READ));
        for (let start = 0; start < states.length; start += capacity) {
          const count = Math.min(capacity, states.length - start);
          let sync = null;
          try {
            withBuffer(() => {
              const packNames = [gl.PACK_ALIGNMENT, gl.PACK_ROW_LENGTH, gl.PACK_SKIP_PIXELS, gl.PACK_SKIP_ROWS];
              const previousPack = packNames.map(name => gl.getParameter(name));
              try {
                gl.pixelStorei(gl.PACK_ALIGNMENT, 4);
                gl.pixelStorei(gl.PACK_ROW_LENGTH, 0);
                gl.pixelStorei(gl.PACK_SKIP_PIXELS, 0);
                gl.pixelStorei(gl.PACK_SKIP_ROWS, 0);
                for (let index = 0; index < count; index++) {
                  setPose(states[start + index]);
                  renderObservation();
                  // Capture this pose immediately, before rendering the next.
                  gl.readPixels(0, 0, 64, 64, gl.RGBA, gl.UNSIGNED_BYTE, index * RGBA_BYTES);
                }
                sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
                if (!sync || gl.getError() !== gl.NO_ERROR) {
                  throw new Error('Unable to submit camera readback.');
                }
                gl.flush();
              } finally {
                packNames.forEach((name, index) => gl.pixelStorei(name, previousPack[index]));
              }
            });
            await waitForReadback(sync);
            withBuffer(() => {
              gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, rgba, 0, count * RGBA_BYTES);
              if (gl.getError() !== gl.NO_ERROR) throw new Error('Unable to read captured camera frames.');
            });
            // Match observeRGBDirect exactly: flip WebGL's bottom-up rows and
            // discard alpha, retaining frame order and every RGB byte.
            let outIndex = start * FRAME_BYTES;
            for (let index = 0; index < count; index++) {
              const frameOffset = index * RGBA_BYTES;
              for (let row = 63; row >= 0; row--) {
                let inIndex = frameOffset + row * 64 * 4;
                for (let col = 0; col < 64; col++, inIndex += 4) {
                  target[outIndex++] = rgba[inIndex];
                  target[outIndex++] = rgba[inIndex + 1];
                  target[outIndex++] = rgba[inIndex + 2];
                }
              }
            }
          } finally {
            if (sync) gl.deleteSync(sync);
          }
        }
      } finally {
        gl.deleteBuffer(buffer);
      }
      return { rgb: target, frames: states.length };
    }

    function setPose(s) {
      car.position.set(s.x, s.y, s.z);
      car.heading = s.heading;
      car.speed = s.speed || 0;
      car.steerAngle = s.steerAngle || 0;
      if (car.mesh) {
        car.mesh.position.set(s.x, s.y, s.z);
        car.mesh.rotation.y = s.heading;
      }
    }

    function step(action) {
      const events = advance(car, track, action);
      if (car.mesh) {
        car.mesh.position.set(car.position.x, car.position.y, car.position.z);
        car.mesh.rotation.y = car.heading;
      }
      steps++;
      if (events.collision) collisions++;
      if (events.offroad) offroadSteps++;
      offroadDuration += events.offroadDuration;
      return {
        state: getState(), events,
        metrics: { steps, collisions, offroadSteps, offroadDuration },
        simulationTime: steps * CONTROL_DT
      };
    }

    function stepPhysics(action) {
      const events = advance(car, track, action);
      steps++;
      if (events.collision) collisions++;
      if (events.offroad) offroadSteps++;
      offroadDuration += events.offroadDuration;
      return {
        state: getState(), events,
        metrics: { steps, collisions, offroadSteps, offroadDuration },
        simulationTime: steps * CONTROL_DT
      };
    }

    function ghostStep(state, action) {
      ghost.reset(state);
      advance(ghost, track, action);
      return ghost.getState();
    }

    return Object.freeze({ version: VERSION, getConfig, getState, reset, setPose, observe, observeRGB, observeRGBDirect, observeRGBBatch, step, stepPhysics, ghostStep });
  }

  root.NeuroDriveControl = Object.freeze({ CONTROL_DT, SUBSTEPS, VERSION, normalizeAction, advance, create });
})(typeof window !== 'undefined' ? window : globalThis);
