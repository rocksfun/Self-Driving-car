// --------------------------------------------------------------------------
// NeuroDrive 3D — Main Simulation Controller, Three.js Renderer,
// Front Dashcam Feed, Dataset Collector & Autonomous AI Driver
// --------------------------------------------------------------------------

(function () {
  'use strict';

  const collectionMode = new URLSearchParams(window.location.search).get('collect') === '1';
  const CONTROL_DT = NeuroDriveControl.CONTROL_DT;
  let controlAccumulator = 0;
  let controlGeneration = 0;
  let episodeHadCollision = false;

  // Three.js Core
  let scene, camera, renderer, clock;
  let track, scenery, car;

  // Front Dashcam Camera & Dedicated Renderer
  let dashcamCamera, dashcamRenderer;

  // Camera views: 'follow' (Chase) | 'heli' (Overhead)
  let currentCameraMode = 'follow';
  let cameraFollowHeading = 0;
  let smoothLookTarget = new THREE.Vector3();
  let cameraFollowInitialized = false;

  // Visual Render Interpolation State (Fixed-Step 20 Hz Physics -> 60 FPS Render)
  let prevCarPos = new THREE.Vector3();
  let currCarPos = new THREE.Vector3();
  let prevCarHeading = 0;
  let currCarHeading = 0;
  let renderInterpolationInitialized = false;

  function shortestAngleDist(from, to) {
    let diff = (to - from) % (Math.PI * 2);
    if (diff < -Math.PI) diff += Math.PI * 2;
    if (diff > Math.PI) diff -= Math.PI * 2;
    return diff;
  }

  function syncVisualStateToPhysics(resetCamera = false) {
    if (!car) return;
    prevCarPos.copy(car.position);
    currCarPos.copy(car.position);
    prevCarHeading = car.heading;
    currCarHeading = car.heading;
    if (car.mesh) {
      car.mesh.position.copy(car.position);
      car.mesh.rotation.y = car.heading;
    }
    renderInterpolationInitialized = true;
    if (resetCamera) {
      cameraFollowInitialized = false;
    }
  }

  let sunLight, hemiLight, ambientLight;

  // Particle System (Tire smoke / grass dust)
  let particles = [];
  let particleGeo, smokeMat, dustMat;

  // Input Keys (Manual Driving)
  const keys = {
    forward: false,
    reverse: false,
    left: false,
    right: false,
    up: false,
    down: false
  };

  // Driving Mode: 'manual' | 'autonomous'
  let currentMode = 'manual';

  // Dataset Collector State (Manual Mode)
  let isRecording = false;
  let recordedEpisodes = [];
  let currentEpisodeSteps = [];
  let totalRecordedSamples = 0;

  // Autonomous Driving & ONNX Model State
  let onnxSession = null;
  let isOnnxInferring = false;
  let onnxHiddenState = null; // GRU Recurrent Memory state [1, 128]
  let isAutoDrivingActive = false; // Car starts stationary until "Start Driving" is clicked
  let predictedAction = { steering: 0.0, throttle: 0.0 };

  // Episode Lifecycle
  let isEpisodeCompleting = false;
  let completedEpisodeCount = 0;
  let currentEpisode = 1;

  // UI DOM Elements
  const canvas = document.getElementById('three-canvas');
  const radarCanvas = document.getElementById('radar-canvas');
  const radarCtx = radarCanvas ? radarCanvas.getContext('2d') : null;

  const dashcamCanvas = document.getElementById('dashcam-canvas');
  const obsCanvas = document.getElementById('obs-canvas');
  const obsCtx = obsCanvas ? obsCanvas.getContext('2d', { willReadFrequently: true }) : null;

  // HUD Metrics
  const statSpeed = document.getElementById('stat-speed');
  const statGear = document.getElementById('stat-gear');
  const statDistance = document.getElementById('stat-distance');
  const statStatus = document.getElementById('stat-status');
  const statEpisodeHud = document.getElementById('stat-episode-hud');
  const toastOverlay = document.getElementById('toast-overlay');
  const brandBadge = document.getElementById('brand-badge');

  // Top Nav Buttons
  const btnModeManual = document.getElementById('btn-mode-manual');
  const btnModeAuto = document.getElementById('btn-mode-auto');
  const btnCamFollow = document.getElementById('btn-cam-follow');
  const btnCamHeli = document.getElementById('btn-cam-heli');
  const btnResetCar = document.getElementById('btn-reset-car');

  // Panels
  const recorderPanel = document.getElementById('recorder-panel');
  const autonomousPanel = document.getElementById('autonomous-panel');

  // Dataset Collector Elements
  const badgeRecorderStatus = document.getElementById('badge-recorder-status');
  const recStatSamples = document.getElementById('rec-stat-samples');
  const recStatEpisodes = document.getElementById('rec-stat-episodes');
  const recStatAction = document.getElementById('rec-stat-action');
  const btnToggleRecord = document.getElementById('btn-toggle-record');
  const btnRecordText = document.getElementById('btn-record-text');
  const btnDownloadDataset = document.getElementById('btn-download-dataset');
  const btnClearDataset = document.getElementById('btn-clear-dataset');

  // Autonomous Panel Elements
  const badgeAutoStatus = document.getElementById('badge-auto-status');
  const autoModelName = document.getElementById('auto-model-name');
  const autoModelDesc = document.getElementById('auto-model-desc');
  const btnAutoChangeModel = document.getElementById('btn-auto-change-model');
  const btnHeroAutoDrive = document.getElementById('btn-hero-auto-drive');
  const heroBtnIcon = document.getElementById('hero-btn-icon');
  const heroBtnText = document.getElementById('hero-btn-text');
  const heroBtnSub = document.getElementById('hero-btn-sub');
  const btnAutoReset = document.getElementById('btn-auto-reset');
  const btnAutoManual = document.getElementById('btn-auto-manual');

  // Telemetry Gauges
  const gaugeSteerBar = document.getElementById('gauge-steer-bar');
  const gaugeSteerVal = document.getElementById('gauge-steer-val');
  const gaugeThrottleBar = document.getElementById('gauge-throttle-bar');
  const gaugeThrottleVal = document.getElementById('gauge-throttle-val');

  // Model Selection Modal Elements
  const modelModal = document.getElementById('model-modal');
  const btnCloseModal = document.getElementById('btn-close-modal');
  const optLoadOnnx = document.getElementById('opt-load-onnx');
  const optLoadCustom = document.getElementById('opt-load-custom');
  const inputModelFile = document.getElementById('input-model-file');

  // --------------------------------------------------------------------------
  // Initialization
  // --------------------------------------------------------------------------
  function init() {
    clock = new THREE.Clock();

    // 1. Create Main Scene with Crisp Daytime Atmosphere
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x87ceeb); // Clear daytime sky blue
    scene.fog = new THREE.FogExp2(0xcfe2f3, 0.0018); // Natural horizon haze

    // 2. Setup Perspective Main Chase/Heli Camera
    camera = new THREE.PerspectiveCamera(
      55,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );

    // 3. Setup Main WebGL Renderer
    renderer = new THREE.WebGLRenderer({
      canvas: canvas,
      antialias: true,
      powerPreference: 'high-performance',
      precision: 'mediump',
      preserveDrawingBuffer: true
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.25));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;

    // 4. Setup Lighting
    setupLighting();

    // 5. Build World Geometry: Track & Scenery
    track = new Track3D(scene);
    scenery = new Scenery3D(scene, track);
    car = new Car3D(scene, track);
    car.reset();
    syncVisualStateToPhysics(true);
    updateCamera(0.016);

    window.car = car;
    window.track = track;
    window.scenery = scenery;
    window.scene = scene;
    window.camera = camera;
    window.renderer = renderer;

    // 6. Setup Front Dashcam Camera & Renderer
    setupDashcam();

    // 7. Particle System Setup
    setupParticles();

    // 8. Event Listeners & UI Binding
    if (!collectionMode) setupEvents();

    if (collectionMode) {
      window.neuroDrive = NeuroDriveControl.create({
        car, track, obsCanvas, dashcamRenderer,
        renderObservation: () => {
          updateDashcam();
        },
        onReset: () => {
          controlGeneration++;
          onnxHiddenState = null;
          isEpisodeCompleting = false;
          episodeHadCollision = false;
        }
      });
      window.neuroDrive.reset();
    }

    // 9. Preload ONNX Model in background
    if (!collectionMode) loadOnnxModel('model.onnx', 'Vision CNN (model.onnx)');

    // 10. Start Simulation Animation Loop
    animate();
    showToast('🏎️ Track Ready! Drive into the garage to complete episodes.');
  }

  function setupLighting() {
    // Hemispherical Sky (Crisp Daylight Blue) & Ground (Slate)
    hemiLight = new THREE.HemisphereLight(0xdbeafe, 0x334155, 0.9);
    scene.add(hemiLight);

    // Clean neutral daylight ambient fill light
    ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    // Natural overhead Sun casting clean crisp shadows
    sunLight = new THREE.DirectionalLight(0xfffbeb, 1.9);
    sunLight.position.set(-90, 85, -60);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.width = 1024;
    sunLight.shadow.mapSize.height = 1024;
    sunLight.shadow.camera.near = 0.5;
    sunLight.shadow.camera.far = 280;
    sunLight.shadow.bias = -0.0003;

    const shadowDist = 95;
    sunLight.shadow.camera.left = -shadowDist;
    sunLight.shadow.camera.right = shadowDist;
    sunLight.shadow.camera.top = shadowDist;
    sunLight.shadow.camera.bottom = -shadowDist;

    scene.add(sunLight);
  }

  // --------------------------------------------------------------------------
  // Front Dashcam Feed Setup
  // --------------------------------------------------------------------------
  function setupDashcam() {
    if (!dashcamCanvas) return;

    // Both training data collection and live ONNX inference require exact 1:1 square aspect ratio
    const aspect = 1.0;
    const camWidth = collectionMode ? 64 : 128;
    const camHeight = collectionMode ? 64 : 128;

    // Perspective camera mounted on front hood / windshield looking directly ahead
    dashcamCamera = new THREE.PerspectiveCamera(65, aspect, 0.2, 350);

    // Lightweight dedicated WebGL renderer with preserveDrawingBuffer enabled
    dashcamRenderer = new THREE.WebGLRenderer({
      canvas: dashcamCanvas,
      antialias: false,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance'
    });
    dashcamRenderer.setSize(camWidth, camHeight, false);
    dashcamRenderer.toneMapping = THREE.ACESFilmicToneMapping;
    dashcamRenderer.toneMappingExposure = 1.0;

    window.dashcamRenderer = dashcamRenderer;
    window.dashcamCamera = dashcamCamera;
  }

  function updateDashcam() {
    if (!dashcamCamera || !dashcamRenderer) return;

    const heading = car && car.mesh ? car.mesh.rotation.y : (car ? car.heading : 0);
    const pos = car && car.mesh ? car.mesh.position : (car ? car.position : { x: 0, y: 0, z: 0 });

    // Car forward direction vector in world space
    const fwdX = -Math.sin(heading);
    const fwdZ = -Math.cos(heading);

    // Position camera on the hood / front bumper (1.6m ahead of car center, 0.92m height)
    dashcamCamera.position.set(
      pos.x + fwdX * 1.6,
      pos.y + 0.92,
      pos.z + fwdZ * 1.6
    );

    // Aim camera far down the road along car forward vector
    dashcamCamera.lookAt(
      pos.x + fwdX * 30.0,
      pos.y + 0.85,
      pos.z + fwdZ * 30.0
    );

    // Render live feed directly at 64x64 in collectionMode
    dashcamRenderer.render(scene, dashcamCamera);

    // Blit to 64x64 observation canvas only when not in collection mode
    if (!collectionMode && obsCtx) {
      obsCtx.drawImage(dashcamCanvas, 0, 0, 64, 64);
    }
  }

  // --------------------------------------------------------------------------
  // Camera & Particle Updates
  // --------------------------------------------------------------------------
  function setCameraMode(mode) {
    currentCameraMode = (mode === 'heli') ? 'heli' : 'follow';

    if (btnCamFollow) btnCamFollow.classList.toggle('active', currentCameraMode === 'follow');
    if (btnCamHeli) btnCamHeli.classList.toggle('active', currentCameraMode === 'heli');

    showToast(`Camera: ${currentCameraMode.toUpperCase()}`);
  }

  function setupParticles() {
    particleGeo = new THREE.DodecahedronGeometry(0.2, 0);
    smokeMat = new THREE.MeshBasicMaterial({ color: 0x94a3b8, transparent: true, opacity: 0.5 });
    dustMat = new THREE.MeshBasicMaterial({ color: 0x166534, transparent: true, opacity: 0.5 });
  }

  function emitParticle(x, y, z, isGrass = false) {
    if (particles.length > 20) return;

    const mat = isGrass ? dustMat : smokeMat;
    const pMesh = new THREE.Mesh(particleGeo, mat);
    pMesh.position.set(
      x + (Math.random() - 0.5) * 0.4,
      y + 0.1,
      z + (Math.random() - 0.5) * 0.4
    );
    scene.add(pMesh);

    particles.push({
      mesh: pMesh,
      life: 1.0,
      decay: 2.2 + Math.random() * 1.5,
      vy: 0.5 + Math.random() * 0.5
    });
  }

  function updateParticles(dt) {
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.life -= p.decay * dt;
      p.mesh.position.y += p.vy * dt;
      p.mesh.scale.multiplyScalar(1.0 + 0.8 * dt);

      if (p.life <= 0) {
        scene.remove(p.mesh);
        particles.splice(i, 1);
      }
    }
  }

  function updateCamera(dt) {
    if (!car || !car.mesh) return;
    const carPos = car.mesh.position;
    const carHeading = car.mesh.rotation.y;

    if (!cameraFollowInitialized) {
      cameraFollowHeading = carHeading;
      smoothLookTarget.set(carPos.x, carPos.y + 1.2, carPos.z);
      if (currentCameraMode === 'follow') {
        const followDist = 8.5;
        const followHeight = 3.6;
        camera.position.set(
          carPos.x + Math.sin(cameraFollowHeading) * followDist,
          carPos.y + followHeight,
          carPos.z + Math.cos(cameraFollowHeading) * followDist
        );
        camera.lookAt(smoothLookTarget);
      } else {
        camera.position.set(carPos.x, 35.0, carPos.z + 18.0);
        camera.lookAt(smoothLookTarget);
      }
      cameraFollowInitialized = true;
      return;
    }

    if (currentCameraMode === 'follow') {
      const followDist = 8.5;
      const followHeight = 3.6;

      // 1. Damped heading pivot: filter out micro-steering twitch whiplash
      const headingDiff = shortestAngleDist(cameraFollowHeading, carHeading);
      const headingSmoothing = 1.0 - Math.exp(-5.0 * dt);
      cameraFollowHeading += headingDiff * headingSmoothing;

      // 2. Camera target position behind car along smoothed heading
      const targetX = carPos.x + Math.sin(cameraFollowHeading) * followDist;
      const targetZ = carPos.z + Math.cos(cameraFollowHeading) * followDist;
      const targetY = carPos.y + followHeight;

      // Smooth camera position spring
      const posSmoothing = 1.0 - Math.exp(-8.0 * dt);
      camera.position.x += (targetX - camera.position.x) * posSmoothing;
      camera.position.z += (targetZ - camera.position.z) * posSmoothing;
      camera.position.y += (targetY - camera.position.y) * posSmoothing;

      // 3. Smooth look-at target: lags synchronously with camera to eliminate angular vibration
      const desiredLookX = carPos.x;
      const desiredLookY = carPos.y + 1.2;
      const desiredLookZ = carPos.z;

      const lookSmoothing = 1.0 - Math.exp(-10.0 * dt);
      smoothLookTarget.x += (desiredLookX - smoothLookTarget.x) * lookSmoothing;
      smoothLookTarget.y += (desiredLookY - smoothLookTarget.y) * lookSmoothing;
      smoothLookTarget.z += (desiredLookZ - smoothLookTarget.z) * lookSmoothing;

      camera.lookAt(smoothLookTarget);
    } else {
      const heliHeight = 35.0;
      const heliTarget = new THREE.Vector3(carPos.x, heliHeight, carPos.z + 18.0);
      camera.position.lerp(heliTarget, 1.0 - Math.exp(-5.0 * dt));
      smoothLookTarget.lerp(new THREE.Vector3(carPos.x, 0, carPos.z), 1.0 - Math.exp(-8.0 * dt));
      camera.lookAt(smoothLookTarget);
    }
  }

  // --------------------------------------------------------------------------
  // GPS Radar Map
  // --------------------------------------------------------------------------
  function updateRadar() {
    if (!radarCtx) return;

    const w = radarCanvas.width;
    const h = radarCanvas.height;

    radarCtx.fillStyle = '#05070a';
    radarCtx.fillRect(0, 0, w, h);

    const scaleX = w / 190.0;
    const scaleZ = h / 290.0;
    const scale = Math.min(scaleX, scaleZ);

    const toRadarX = (wx) => w / 2 + wx * scale;
    const toRadarY = (wz) => 135 + wz * scale;

    radarCtx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
    radarCtx.lineWidth = 4;

    // South Entrance
    radarCtx.beginPath();
    radarCtx.moveTo(toRadarX(0), toRadarY(85));
    radarCtx.lineTo(toRadarX(0), toRadarY(29.5));
    radarCtx.stroke();

    // Roundabout Circle
    radarCtx.beginPath();
    radarCtx.arc(toRadarX(0), toRadarY(0), (track.innerRadius + track.outerRadius) / 2 * scale, 0, Math.PI * 2);
    radarCtx.stroke();

    // Exit 1 (East Dead-End)
    radarCtx.strokeStyle = 'rgba(244, 63, 94, 0.6)';
    radarCtx.beginPath();
    radarCtx.moveTo(toRadarX(29.5), toRadarY(0));
    radarCtx.lineTo(toRadarX(80), toRadarY(0));
    radarCtx.stroke();

    // Exit 3 (West Dead-End)
    radarCtx.beginPath();
    radarCtx.moveTo(toRadarX(-29.5), toRadarY(0));
    radarCtx.lineTo(toRadarX(-80), toRadarY(0));
    radarCtx.stroke();

    // Exit 2 Curvy Road to Garage
    radarCtx.strokeStyle = 'rgba(16, 185, 129, 0.6)';
    if (track.curvyPoints) {
      radarCtx.beginPath();
      for (let i = 0; i < track.curvyPoints.length; i++) {
        const p = track.curvyPoints[i];
        if (i === 0) radarCtx.moveTo(toRadarX(p.x), toRadarY(p.z));
        else radarCtx.lineTo(toRadarX(p.x), toRadarY(p.z));
      }
      radarCtx.stroke();
    }

    // Parking Garage Icon
    radarCtx.fillStyle = 'rgba(16, 185, 129, 0.35)';
    radarCtx.fillRect(toRadarX(-15), toRadarY(-185), 30 * scale, 34 * scale);

    // Car Indicator Arrow
    const rPos = car.mesh ? car.mesh.position : car.position;
    const rHeading = car.mesh ? car.mesh.rotation.y : car.heading;
    const carX = toRadarX(rPos.x);
    const carY = toRadarY(rPos.z);

    radarCtx.save();
    radarCtx.translate(carX, carY);
    radarCtx.rotate(-rHeading);

    radarCtx.fillStyle = '#38bdf8';
    radarCtx.beginPath();
    radarCtx.moveTo(0, -6);
    radarCtx.lineTo(4, 5);
    radarCtx.lineTo(0, 3);
    radarCtx.lineTo(-4, 5);
    radarCtx.closePath();
    radarCtx.fill();

    radarCtx.restore();
  }

  // --------------------------------------------------------------------------
  // HUD Telemetry Updates
  // --------------------------------------------------------------------------
  function updateHUD() {
    const kmh = Math.round(Math.abs(car.speed) * 3.6);
    if (statSpeed) statSpeed.textContent = kmh;

    if (statGear) {
      if (car.speed < -0.5) {
        statGear.textContent = 'R';
        statGear.style.color = 'var(--accent-rose)';
      } else if (kmh < 2) {
        statGear.textContent = 'N';
        statGear.style.color = 'var(--text-muted)';
      } else if (kmh < 25) {
        statGear.textContent = '1';
        statGear.style.color = 'var(--accent-emerald)';
      } else if (kmh < 50) {
        statGear.textContent = '2';
        statGear.style.color = 'var(--accent-emerald)';
      } else if (kmh < 80) {
        statGear.textContent = '3';
        statGear.style.color = 'var(--accent-cyan)';
      } else if (kmh < 110) {
        statGear.textContent = '4';
        statGear.style.color = 'var(--accent-cyan)';
      } else {
        statGear.textContent = '5';
        statGear.style.color = 'var(--accent-amber)';
      }
    }

    if (statDistance) {
      statDistance.textContent = `${Math.round(car.distanceTraveled * 2.5)} M`;
    }

    if (statStatus) {
      if (isEpisodeCompleting) {
        statStatus.textContent = '🏆 EPISODE COMPLETE';
        statStatus.className = 'status-badge garage';
      } else if (car.inParkingGarage) {
        statStatus.textContent = '🅿️ PARKING GARAGE';
        statStatus.className = 'status-badge garage';
      } else if (car.hitDeadEnd) {
        statStatus.textContent = '⛔ CRASH BARRIER';
        statStatus.className = 'status-badge crash';
      } else if (car.isOffroad) {
        statStatus.textContent = '⚠️ OFF-ROAD (GRASS)';
        statStatus.className = 'status-badge offroad';
      } else {
        statStatus.textContent = '🟢 ON ROAD';
        statStatus.className = 'status-badge';
      }
    }
  }

  function showToast(msg) {
    if (!toastOverlay) return;
    toastOverlay.textContent = msg;
    toastOverlay.classList.add('show');
    setTimeout(() => {
      toastOverlay.classList.remove('show');
    }, 2500);
  }

  // --------------------------------------------------------------------------
  // Driving Mode Switcher (Manual vs. Autonomous)
  // --------------------------------------------------------------------------
  function setDrivingMode(mode) {
    controlGeneration++;
    controlAccumulator = 0;
    onnxHiddenState = null;
    currentMode = (mode === 'autonomous') ? 'autonomous' : 'manual';
    syncVisualStateToPhysics(false);

    if (currentMode === 'manual') {
      if (btnModeManual) btnModeManual.classList.add('active');
      if (btnModeAuto) btnModeAuto.classList.remove('active');

      if (recorderPanel) {
        recorderPanel.classList.remove('hidden');
        recorderPanel.style.display = '';
      }
      if (autonomousPanel) {
        autonomousPanel.classList.add('hidden');
        autonomousPanel.style.display = 'none';
      }

      if (brandBadge) {
        brandBadge.textContent = 'MANUAL 3D';
        brandBadge.className = 'brand-badge badge-manual';
      }

      // Halt autonomous driving if active
      if (isAutoDrivingActive) {
        pauseAutoDriving('Switched to Manual Driving');
      }

      showToast('🎮 Manual Driving Mode active (WASD / Arrows)');
    } else {
      if (btnModeAuto) btnModeAuto.classList.add('active');
      if (btnModeManual) btnModeManual.classList.remove('active');

      if (autonomousPanel) {
        autonomousPanel.classList.remove('hidden');
        autonomousPanel.style.display = '';
      }
      if (recorderPanel) {
        recorderPanel.classList.add('hidden');
        recorderPanel.style.display = 'none';
      }

      if (brandBadge) {
        brandBadge.textContent = 'AUTONOMOUS AI';
        brandBadge.className = 'brand-badge badge-auto';
      }

      // Stop manual dataset recording if active
      if (isRecording) {
        toggleRecording();
      }

      // If no model loaded, attempt loading default model.onnx
      if (!onnxSession) {
        loadOnnxModel('../model.onnx', 'Vision CNN (model.onnx)');
      }

      showToast('🤖 Autonomous AI Mode active. Click "Start Driving" to launch.');
    }
  }

  // --------------------------------------------------------------------------
  // Dataset Collector (Manual Mode)
  // --------------------------------------------------------------------------
  function toggleRecording() {
    isRecording = !isRecording;

    if (isRecording) {
      if (badgeRecorderStatus) {
        badgeRecorderStatus.className = 'recorder-badge-recording';
        badgeRecorderStatus.textContent = 'RECORDING';
      }
      if (btnToggleRecord) btnToggleRecord.classList.add('recording');
      if (btnRecordText) btnRecordText.textContent = 'Stop Recording';
      showToast('⏺️ Data Recording Started! Drive to collect training frames.');
    } else {
      if (badgeRecorderStatus) {
        badgeRecorderStatus.className = 'recorder-badge-idle';
        badgeRecorderStatus.textContent = 'IDLE';
      }
      if (btnToggleRecord) btnToggleRecord.classList.remove('recording');
      if (btnRecordText) btnRecordText.textContent = 'Start Recording';

      // Finalize episode
      if (currentEpisodeSteps.length > 0) {
        recordedEpisodes.push({
          episodeId: recordedEpisodes.length + 1,
          totalSteps: currentEpisodeSteps.length,
          completedAt: new Date().toISOString(),
          reason: 'stopped_recording',
          success: isSuccessfulParking(),
          steps: currentEpisodeSteps
        });
        currentEpisodeSteps = [];
      }
      showToast(`⏹️ Recording Stopped. ${totalRecordedSamples} frames ready to export.`);
    }

    updateRecorderUI();
  }

  function manualAction() {
    return {
      steering: Number(keys.right) - Number(keys.left),
      throttle: Number(keys.forward || keys.up) - Number(keys.reverse || keys.down),
      allowReverse: true
    };
  }

  function recordDataSample(action) {
    if (!obsCtx) return;

    // Extract current observation (64x64 RGB uint8 nested array)
    const imgData = obsCtx.getImageData(0, 0, 64, 64).data;
    const obs64 = [];
    for (let y = 0; y < 64; y++) {
      const row = [];
      for (let x = 0; x < 64; x++) {
        const idx = (y * 64 + x) * 4;
        row.push([imgData[idx], imgData[idx + 1], imgData[idx + 2]]);
      }
      obs64.push(row);
    }

    // Manual action values: steering (-1.0 to 1.0), throttle (-1.0 to 1.0)
    const { steering, throttle } = action;

    const sample = {
      step: currentEpisodeSteps.length,
      timestamp: Date.now(),
      simulationTime: currentEpisodeSteps.length * CONTROL_DT,
      action: {
        steering, throttle,
        allowReverse: action.allowReverse,
        keys: {
          forward: keys.forward || keys.up,
          reverse: keys.reverse || keys.down,
          left: keys.left,
          right: keys.right
        }
      },
      state: car.getState(),
      observation: obs64
    };

    currentEpisodeSteps.push(sample);
    totalRecordedSamples++;

    if (recStatAction) {
      recStatAction.textContent = `S: ${steering >= 0 ? '+' : ''}${steering.toFixed(2)} | T: ${throttle >= 0 ? '+' : ''}${throttle.toFixed(2)}`;
    }

    updateRecorderUI();
  }

  function updateRecorderUI() {
    if (recStatSamples) recStatSamples.textContent = totalRecordedSamples;
    if (recStatEpisodes) recStatEpisodes.textContent = recordedEpisodes.length + (currentEpisodeSteps.length > 0 ? 1 : 0);

    const hasData = totalRecordedSamples > 0;
    if (btnDownloadDataset) btnDownloadDataset.disabled = !hasData;
    if (btnClearDataset) btnClearDataset.disabled = !hasData;
  }

  function downloadDataset() {
    const episodesToSave = [...recordedEpisodes];
    if (currentEpisodeSteps.length > 0) {
      episodesToSave.push({
        episodeId: episodesToSave.length + 1,
        totalSteps: currentEpisodeSteps.length,
        completedAt: new Date().toISOString(),
        reason: 'exported_in_progress',
        success: isSuccessfulParking(),
        steps: currentEpisodeSteps
      });
    }

    if (totalRecordedSamples === 0 && episodesToSave.length === 0) {
      showToast('⚠️ No data recorded yet. Record some driving first!');
      return;
    }

    const allSamples = [];
    for (const ep of episodesToSave) {
      for (const step of ep.steps) {
        allSamples.push({ episodeId: ep.episodeId, ...step });
      }
    }

    const dataset = {
      metadata: {
        datasetName: 'NeuroDrive 3D Manual Driving Vision Dataset',
        createdAt: new Date().toISOString(),
        totalSamples: allSamples.length,
        totalEpisodes: episodesToSave.length,
        observationShape: [64, 64, 3],
        observationFormat: 'RGB uint8 [0, 255]',
        samplingRateHz: 1 / CONTROL_DT,
        controlDt: CONTROL_DT,
        physicsSubsteps: NeuroDriveControl.SUBSTEPS,
        source: 'threejs-dashcam',
        physics: 'Car3D',
        version: NeuroDriveControl.VERSION,
        alignment: 'observation_and_state_before_action',
        manualReverseEnabled: true,
        actionFormat: {
          steering: 'float in [-1.0, 1.0] (negative = left, positive = right)',
          throttle: 'float in [-1.0, 1.0] (positive = gas, negative = reverse/brake; action.allowReverse=true)'
        }
      },
      episodes: episodesToSave,
      samples: allSamples
    };

    const jsonStr = JSON.stringify(dataset);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    a.href = url;
    a.download = `neurodrive_3d_dataset_${timestamp}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);

    showToast(`💾 Exported 3D Dataset with ${allSamples.length} frames!`);
  }

  function clearDataset() {
    if (totalRecordedSamples === 0) return;
    if (!confirm(`Clear all ${totalRecordedSamples} recorded frames?`)) return;

    recordedEpisodes = [];
    currentEpisodeSteps = [];
    totalRecordedSamples = 0;

    if (recStatAction) recStatAction.textContent = 'S: 0.00 | T: 0.00';
    updateRecorderUI();
    showToast('🗑️ Recorded dataset cleared.');
  }

  // --------------------------------------------------------------------------
  // ONNX Runtime Inference & Autonomous Agent
  // --------------------------------------------------------------------------
  async function loadOnnxModel(source = 'model.onnx', displayName = 'Vision Transformer (ViT-GRU-ACT)') {
    if (typeof ort === 'undefined') {
      console.warn('ONNX Runtime Web library (ort) not loaded.');
      return false;
    }

    try {
      showToast(`⏳ Loading ${displayName}...`);
      if (ort.env && ort.env.wasm) {
        ort.env.wasm.numThreads = 1;
      }

      onnxSession = await ort.InferenceSession.create(source, {
        executionProviders: ['wasm']
      });
      controlGeneration++;
      onnxHiddenState = null;

      console.log('ONNX Model loaded successfully:', onnxSession);
      if (autoModelName) autoModelName.textContent = displayName;
      if (autoModelDesc) autoModelDesc.textContent = 'PyTorch Vision CNN Policy (Active)';

      if (currentMode !== 'autonomous') {
        setDrivingMode('autonomous');
      }

      closeModelModal();
      showToast(`🧠 ${displayName} Loaded! AI Driving Activated.`);

      // Automatically engage autonomous driving so the user watching on port 8080 sees the car drive immediately
      if (!isAutoDrivingActive) {
        toggleAutoDriving();
      }

      return true;
    } catch (err) {
      console.warn('Failed to load ONNX model from primary path:', err);
      if (source !== '../model.onnx') {
        try {
          onnxSession = await ort.InferenceSession.create('../model.onnx', { executionProviders: ['wasm'] });
          controlGeneration++;
          onnxHiddenState = null;
          if (autoModelName) autoModelName.textContent = displayName;
          if (autoModelDesc) autoModelDesc.textContent = 'PyTorch ViT + ACT + GRU Memory (Active)';
          if (!isAutoDrivingActive && badgeAutoStatus) {
            badgeAutoStatus.className = 'badge-auto-ready';
            badgeAutoStatus.textContent = 'READY';
          }
          closeModelModal();
          showToast(`🧠 ${displayName} Loaded & Ready! Click Start Driving.`);
          return true;
        } catch (e) {
          console.error('Fallback model load failed:', e);
        }
      }
      showToast(`⚠️ Could not load model: ${err.message}`);
      return false;
    }
  }

  // Convert 64x64 canvas to Float32Array in NCHW [1, 3, 64, 64] normalized to [0, 1]
  function createTensorFromCanvas(c) {
    const ctx = c.getContext('2d', { willReadFrequently: true });
    const imgData = ctx.getImageData(0, 0, 64, 64).data;
    const floatData = new Float32Array(1 * 3 * 64 * 64);
    const planeSize = 64 * 64;

    for (let i = 0; i < planeSize; i++) {
      const srcIdx = i * 4;
      floatData[0 * planeSize + i] = imgData[srcIdx] / 255.0;     // Red
      floatData[1 * planeSize + i] = imgData[srcIdx + 1] / 255.0; // Green
      floatData[2 * planeSize + i] = imgData[srcIdx + 2] / 255.0; // Blue
    }

    const inputName = (onnxSession && onnxSession.inputNames && onnxSession.inputNames[0]) || 'camera_input';
    return {
      tensor: new ort.Tensor('float32', floatData, [1, 3, 64, 64]),
      inputName
    };
  }

  async function runOnnxInference(c) {
    if (!onnxSession || isOnnxInferring || !c) return null;
    isOnnxInferring = true;
    const generation = controlGeneration;
    const session = onnxSession;

    try {
      const { tensor, inputName } = createTensorFromCanvas(c);
      const feeds = { [inputName]: tensor };

      // Pass GRU hidden state if model requires/supports 'hidden_in'
      if (session.inputNames && session.inputNames.includes('hidden_in')) {
        if (!onnxHiddenState) {
          onnxHiddenState = new ort.Tensor('float32', new Float32Array(128), [1, 128]);
        }
        feeds['hidden_in'] = onnxHiddenState;
      }

      const results = await session.run(feeds);
      // A reset, mode/model change or pause during inference invalidates both
      // its action and recurrent memory. Never apply an old frame's output.
      if (generation !== controlGeneration || session !== onnxSession) return null;

      // Update GRU hidden state if returned
      if (results['hidden_out']) {
        onnxHiddenState = results['hidden_out'];
      }

      // Action output: support 'action' tensor [1, 2] or first output [steering, throttle]
      const actionTensor = results['action'] || results[session.outputNames[0]];
      const outputData = actionTensor.data;
      if (!Number.isFinite(Number(outputData[0])) || !Number.isFinite(Number(outputData[1]))) {
        throw new Error('Model returned a non-finite action.');
      }

      const rawSteering = Math.max(-1.0, Math.min(1.0, Number(outputData[0])));
      const rawThrottle = Math.max(-1.0, Math.min(1.0, Number(outputData[1])));

      // EMA filter: blend 70% new prediction with 30% prior action to smooth WASM actuation jitter
      predictedAction.steering = 0.70 * rawSteering + 0.30 * predictedAction.steering;
      predictedAction.throttle = 0.70 * rawThrottle + 0.30 * predictedAction.throttle;

      updateActuationGauges(predictedAction.steering, predictedAction.throttle);
      return { ...predictedAction };
    } catch (err) {
      console.error('Inference error:', err);
      return null;
    } finally {
      isOnnxInferring = false;
    }
  }

  // --------------------------------------------------------------------------
  // Primary Hero "Start / Pause Driving" State Machine
  // --------------------------------------------------------------------------
  function toggleAutoDriving() {
    if (!onnxSession) {
      showToast('⚠️ No model loaded! Choose or upload an ONNX model first.');
      openModelModal();
      return;
    }

    if (currentMode !== 'autonomous') {
      setDrivingMode('autonomous');
    }

    isAutoDrivingActive = !isAutoDrivingActive;

    if (isAutoDrivingActive) {
      // Start driving
      if (btnHeroAutoDrive) btnHeroAutoDrive.classList.add('active');
      if (heroBtnIcon) heroBtnIcon.textContent = '⏸️';
      if (heroBtnText) heroBtnText.textContent = 'Pause Driving';
      if (heroBtnSub) heroBtnSub.textContent = 'AI is driving — Click to pause';

      if (badgeAutoStatus) {
        badgeAutoStatus.className = 'badge-auto-driving';
        badgeAutoStatus.textContent = 'DRIVING';
      }
      showToast('🤖 Autonomous AI Driving Started!');
    } else {
      pauseAutoDriving('AI Driving Paused');
    }
  }

  function pauseAutoDriving(reason = 'Paused') {
    controlGeneration++;
    controlAccumulator = 0;
    onnxHiddenState = null;
    isAutoDrivingActive = false;
    syncVisualStateToPhysics(false);

    if (btnHeroAutoDrive) btnHeroAutoDrive.classList.remove('active');
    if (heroBtnIcon) heroBtnIcon.textContent = '▶️';
    if (heroBtnText) heroBtnText.textContent = 'Resume Driving';
    if (heroBtnSub) heroBtnSub.textContent = 'Paused — Click to continue driving';

    if (badgeAutoStatus) {
      badgeAutoStatus.className = 'badge-auto-ready';
      badgeAutoStatus.textContent = 'PAUSED';
    }

    predictedAction.steering = 0.0;
    predictedAction.throttle = 0.0;
    updateActuationGauges(0, 0);

    showToast(`⏸️ ${reason}`);
  }

  function resetAutoDrivingState() {
    controlGeneration++;
    controlAccumulator = 0;
    isAutoDrivingActive = false;
    onnxHiddenState = null; // Reset GRU memory
    syncVisualStateToPhysics(false);

    if (btnHeroAutoDrive) btnHeroAutoDrive.classList.remove('active');
    if (heroBtnIcon) heroBtnIcon.textContent = '▶️';
    if (heroBtnText) heroBtnText.textContent = 'Start Driving';
    if (heroBtnSub) heroBtnSub.textContent = 'Click to let the AI drive the car';

    if (badgeAutoStatus) {
      badgeAutoStatus.className = 'badge-auto-ready';
      badgeAutoStatus.textContent = onnxSession ? 'READY' : 'STANDBY';
    }

    predictedAction.steering = 0.0;
    predictedAction.throttle = 0.0;
    updateActuationGauges(0, 0);
  }

  // Update Bi-directional AI Telemetry Gauges
  function updateActuationGauges(steer, throttle) {
    // 1. Steering Gauge: -1.0 (Left) to +1.0 (Right), centered at 50%
    if (gaugeSteerBar && gaugeSteerVal) {
      gaugeSteerVal.textContent = `${steer >= 0 ? '+' : ''}${steer.toFixed(2)}`;
      const sClamped = Math.max(-1.0, Math.min(1.0, steer));

      if (sClamped >= 0) {
        gaugeSteerBar.style.left = '50%';
        gaugeSteerBar.style.width = `${(sClamped * 50).toFixed(1)}%`;
        gaugeSteerBar.style.background = 'linear-gradient(90deg, #38bdf8, #818cf8)';
      } else {
        const widthPct = (-sClamped * 50).toFixed(1);
        gaugeSteerBar.style.left = `${(50 - Number(widthPct)).toFixed(1)}%`;
        gaugeSteerBar.style.width = `${widthPct}%`;
        gaugeSteerBar.style.background = 'linear-gradient(90deg, #818cf8, #38bdf8)';
      }
    }

    // 2. Throttle Gauge: -1.0 (Brake/Reverse) to +1.0 (Gas), centered at 50%
    if (gaugeThrottleBar && gaugeThrottleVal) {
      gaugeThrottleVal.textContent = `${throttle >= 0 ? '+' : ''}${throttle.toFixed(2)}`;
      const tClamped = Math.max(-1.0, Math.min(1.0, throttle));

      if (tClamped >= 0) {
        gaugeThrottleBar.style.left = '50%';
        gaugeThrottleBar.style.width = `${(tClamped * 50).toFixed(1)}%`;
        gaugeThrottleBar.style.background = 'linear-gradient(90deg, #10b981, #34d399)';
      } else {
        const widthPct = (-tClamped * 50).toFixed(1);
        gaugeThrottleBar.style.left = `${(50 - Number(widthPct)).toFixed(1)}%`;
        gaugeThrottleBar.style.width = `${widthPct}%`;
        gaugeThrottleBar.style.background = 'linear-gradient(90deg, #f43f5e, #fb7185)';
      }
    }
  }

  // --------------------------------------------------------------------------
  // Model Selection Modal
  // --------------------------------------------------------------------------
  function openModelModal() {
    if (modelModal) modelModal.classList.remove('hidden');
  }

  function closeModelModal() {
    if (modelModal) modelModal.classList.add('hidden');
  }

  // --------------------------------------------------------------------------
  // Event Listeners & Input Bindings
  // --------------------------------------------------------------------------
  function setupEvents() {
    // Keyboard input handling for Manual Mode
    window.addEventListener('keydown', (e) => {
      if (e.key === 'w' || e.key === 'W' || e.key === 'ArrowUp') keys.forward = true;
      if (e.key === 's' || e.key === 'S' || e.key === 'ArrowDown') keys.reverse = true;
      if (e.key === 'a' || e.key === 'A' || e.key === 'ArrowLeft') keys.left = true;
      if (e.key === 'd' || e.key === 'D' || e.key === 'ArrowRight') keys.right = true;

      // Shortcuts
      if (e.key === 'r' || e.key === 'R') {
        respawnCar();
      }

      if (e.key === 'c' || e.key === 'C') {
        setCameraMode(currentCameraMode === 'follow' ? 'heli' : 'follow');
      }

      // Space bar to toggle recording in Manual mode, or toggle drive in Auto mode
      if (e.key === ' ') {
        if (currentMode === 'manual') {
          toggleRecording();
        } else {
          toggleAutoDriving();
        }
      }
    });

    window.addEventListener('keyup', (e) => {
      if (e.key === 'w' || e.key === 'W' || e.key === 'ArrowUp') keys.forward = false;
      if (e.key === 's' || e.key === 'S' || e.key === 'ArrowDown') keys.reverse = false;
      if (e.key === 'a' || e.key === 'A' || e.key === 'ArrowLeft') keys.left = false;
      if (e.key === 'd' || e.key === 'D' || e.key === 'ArrowRight') keys.right = false;
    });

    // Mode Switcher Buttons
    if (btnModeManual) btnModeManual.addEventListener('click', () => setDrivingMode('manual'));
    if (btnModeAuto) btnModeAuto.addEventListener('click', () => setDrivingMode('autonomous'));
    if (btnAutoManual) btnAutoManual.addEventListener('click', () => setDrivingMode('manual'));

    // Camera buttons
    if (btnCamFollow) btnCamFollow.addEventListener('click', () => setCameraMode('follow'));
    if (btnCamHeli) btnCamHeli.addEventListener('click', () => setCameraMode('heli'));

    // Reset buttons
    if (btnResetCar) btnResetCar.addEventListener('click', respawnCar);
    if (btnAutoReset) btnAutoReset.addEventListener('click', respawnCar);

    // Dataset Collector Buttons
    if (btnToggleRecord) btnToggleRecord.addEventListener('click', toggleRecording);
    if (btnDownloadDataset) btnDownloadDataset.addEventListener('click', downloadDataset);
    if (btnClearDataset) btnClearDataset.addEventListener('click', clearDataset);

    // Autonomous Hero Drive Button
    if (btnHeroAutoDrive) btnHeroAutoDrive.addEventListener('click', toggleAutoDriving);

    // Model Modal Triggers
    if (btnAutoChangeModel) btnAutoChangeModel.addEventListener('click', openModelModal);
    if (btnCloseModal) btnCloseModal.addEventListener('click', closeModelModal);
    if (modelModal) {
      modelModal.addEventListener('click', (e) => {
        if (e.target === modelModal) closeModelModal();
      });
    }

    // Modal Option: Load Default model.onnx
    if (optLoadOnnx) {
      optLoadOnnx.addEventListener('click', async () => {
        closeModelModal();
        await loadOnnxModel('../model.onnx', 'Vision CNN (model.onnx)');
      });
    }

    // Modal Option: Browse Custom .onnx file
    if (optLoadCustom && inputModelFile) {
      optLoadCustom.addEventListener('click', () => {
        inputModelFile.click();
      });

      inputModelFile.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        closeModelModal();
        try {
          const buffer = await file.arrayBuffer();
          await loadOnnxModel(buffer, file.name);
        } catch (err) {
          showToast(`⚠️ Failed reading model file: ${err.message}`);
        }
      });
    }

    // Window Resize Handling
    window.addEventListener('resize', onWindowResize);
  }

  function respawnCar() {
    if (currentEpisodeSteps.length > 0) {
      recordedEpisodes.push({
        episodeId: recordedEpisodes.length + 1,
        totalSteps: currentEpisodeSteps.length,
        completedAt: new Date().toISOString(),
        reason: 'manual_reset', success: false,
        steps: currentEpisodeSteps
      });
      currentEpisodeSteps = [];
      updateRecorderUI();
    }
    if (episodeResetTimer !== null) clearTimeout(episodeResetTimer);
    episodeResetTimer = null;
    car.reset();
    syncVisualStateToPhysics(true);
    episodeHadCollision = false;
    isEpisodeCompleting = false;
    onnxHiddenState = null;
    resetAutoDrivingState();
    showToast('🔄 Car Respawned at Start Line');
  }

  function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }

  // --------------------------------------------------------------------------
  // Simulation Loop
  // --------------------------------------------------------------------------
  let lastRadarTime = 0;
  let episodeResetTimer = null;

  function isSuccessfulParking() {
    return Boolean(car && car.inParkingGarage);
  }

  function completeEpisodeIfParked() {
    if (isEpisodeCompleting || !isSuccessfulParking()) return;
    isEpisodeCompleting = true;
    controlGeneration++;
    onnxHiddenState = null;
    completedEpisodeCount++;
    currentEpisode++;

    // Gently slow vehicle down as it enters the garage
    if (car) car.speed *= 0.25;

    if (statEpisodeHud) statEpisodeHud.textContent = currentEpisode;
    if (isRecording && currentEpisodeSteps.length > 0) {
      recordedEpisodes.push({
        episodeId: recordedEpisodes.length + 1,
        totalSteps: currentEpisodeSteps.length,
        completedAt: new Date().toISOString(),
        reason: 'garage_success',
        success: true,
        hadCollision: episodeHadCollision,
        steps: currentEpisodeSteps
      });
      currentEpisodeSteps = [];
      updateRecorderUI();
    }
    showToast(`🏆 Episode ${completedEpisodeCount} Complete! Parking Garage reached.`);
    episodeResetTimer = setTimeout(() => {
      car.reset();
      syncVisualStateToPhysics(true);
      episodeHadCollision = false;
      isEpisodeCompleting = false;
      controlGeneration++;
      controlAccumulator = 0;
      onnxHiddenState = null;
      episodeResetTimer = null;
      showToast('🏁 Episode Reset: Ready for Next Lap!');
    }, 1500);
  }

  function advanceControl(action) {
    if (renderInterpolationInitialized) {
      prevCarPos.copy(currCarPos);
      prevCarHeading = currCarHeading;
    } else {
      prevCarPos.copy(car.position);
      prevCarHeading = car.heading;
    }
    const events = NeuroDriveControl.advance(car, track, action);
    currCarPos.copy(car.position);
    currCarHeading = car.heading;
    renderInterpolationInitialized = true;

    episodeHadCollision = episodeHadCollision || events.collision;
    completeEpisodeIfParked();
  }

  async function autonomousControlStep() {
    const generation = controlGeneration;
    // Observe -> infer once (carry GRU memory once) -> advance exactly 50 ms.
    // Ensure car mesh & dashcam represent exact physics ground truth for camera snapshot
    car.mesh.position.copy(car.position);
    car.mesh.rotation.y = car.heading;
    updateDashcam();
    const action = await runOnnxInference(obsCanvas);
    if (generation !== controlGeneration || !isAutoDrivingActive || currentMode !== 'autonomous') return;
    if (!action) {
      pauseAutoDriving('Inference failed; driving paused');
      return;
    }
    advanceControl(action);
  }

  function animate(timestamp = 0) {
    // Controlled collection renders only through bridge.observe/reset. RAF,
    // keyboard handlers, particles, inference and garage timers cannot step it.
    if (collectionMode) return;
    requestAnimationFrame(animate);
    const dt = Math.min(clock.getDelta(), 0.1);

    if (!isOnnxInferring) controlAccumulator += dt;
    if (currentMode === 'autonomous' && isAutoDrivingActive && onnxSession && !isEpisodeCompleting) {
      if (!isOnnxInferring && controlAccumulator >= CONTROL_DT) {
        controlAccumulator = 0;
        autonomousControlStep();
      }
    } else {
      while (controlAccumulator >= CONTROL_DT) {
        controlAccumulator -= CONTROL_DT;
        if (isEpisodeCompleting) {
          // Neutral lets friction finish the stop. No held negative throttle
          // can back the parked car out of its successful terminal state.
          advanceControl([0, 0]);
        } else if (currentMode === 'manual') {
          const action = manualAction();
          if (isRecording) {
            car.mesh.position.copy(car.position);
            car.mesh.rotation.y = car.heading;
            updateDashcam();
            recordDataSample(action);
          }
          advanceControl(action);
        } else {
          // A paused autonomous car brakes to rest with the same fixed steps.
          advanceControl([0, -1]);
        }
      }
    }

    // Smooth visual interpolation between 20 Hz physics sub-steps
    if (renderInterpolationInitialized) {
      const alpha = Math.max(0.0, Math.min(1.0, controlAccumulator / CONTROL_DT));
      car.mesh.position.lerpVectors(prevCarPos, currCarPos, alpha);
      const headingDiff = shortestAngleDist(prevCarHeading, currCarHeading);
      car.mesh.rotation.y = prevCarHeading + headingDiff * alpha;
    }

    // 3. Tire Smoke / Off-road Dust Particles
    if (car.isOffroad && Math.abs(car.speed) > 4.0) {
      emitParticle(car.mesh.position.x, car.mesh.position.y, car.mesh.position.z, true);
    } else if (keys.reverse && car.speed > 8.0) {
      emitParticle(car.mesh.position.x, car.mesh.position.y, car.mesh.position.z, false);
    }
    updateParticles(dt);

    // 3. Update Chase/Helicopter Camera
    updateCamera(dt);

    // 4. Update Dashcam & Observation Canvas
    updateDashcam();

    // 5. Update HUD & Mini-map Radar (~15 FPS)
    updateHUD();
    if (timestamp - lastRadarTime > 65) {
      updateRadar();
      lastRadarTime = timestamp;
    }

    // 6. Render Main 3D Viewport
    renderer.render(scene, camera);
  }

  // Auto-start simulation on DOM ready
  window.addEventListener('DOMContentLoaded', init);
})();
