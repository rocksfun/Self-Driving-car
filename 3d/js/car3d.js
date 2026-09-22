// --------------------------------------------------------------------------
// NeuroDrive 3D — Controllable Sports Car with Dynamic Lighting & Wheel Physics
// --------------------------------------------------------------------------

class Car3D {
  constructor(scene, track) {
    this.scene = scene;
    this.track = track;

    // Vehicle transform state
    this.position = track.spawnPos.clone();
    this.heading = track.spawnHeading; // Yaw angle in radians (faces -Z initially)
    this.speed = 0.0; // Units per second in 3D
    this.steerAngle = 0.0;

    // Vehicle physics tuning
    this.maxSpeed = 38.0;          // ~115 KM/H
    this.maxReverseSpeed = -10.0;
    this.acceleration = 24.0;
    this.braking = 36.0;
    this.friction = 12.0;
    this.offroadFriction = 24.0;
    this.turnSpeed = 2.4;

    // Driving metrics
    this.distanceTraveled = 0.0;
    this.isOffroad = false;
    this.inParkingGarage = false;
    this.hitDeadEnd = false;
    this.hitWall = false;

    // Build the visual 3D vehicle hierarchy
    this.wheels = [];
    this.frontWheels = [];
    this.buildModel();
  }

  buildModel() {
    this.mesh = new THREE.Group();
    this.mesh.position.copy(this.position);
    this.mesh.rotation.y = this.heading;

    // 1. Aerodynamic Main Body Chassis
    const bodyMat = new THREE.MeshStandardMaterial({
      color: 0x0284c7, // Vibrant Cyber Cyan
      metalness: 0.85,
      roughness: 0.2
    });
    this.bodyMat = bodyMat;

    // Lower Wedge Chassis
    const lowerBodyGeo = new THREE.BoxGeometry(2.1, 0.55, 4.4);
    const lowerBody = new THREE.Mesh(lowerBodyGeo, bodyMat);
    lowerBody.position.y = 0.45;
    lowerBody.castShadow = true;
    lowerBody.receiveShadow = true;
    this.mesh.add(lowerBody);

    // Aerodynamic Sloped Hood
    const hoodGeo = new THREE.BoxGeometry(1.9, 0.22, 1.6);
    const hood = new THREE.Mesh(hoodGeo, bodyMat);
    hood.position.set(0, 0.72, -1.2);
    hood.rotation.x = 0.12;
    hood.castShadow = true;
    this.mesh.add(hood);

    // Cabin / Tinted Glass Canopy
    const cabinGeo = new THREE.BoxGeometry(1.7, 0.58, 2.2);
    const glassMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      roughness: 0.1,
      metalness: 0.9,
      transparent: true,
      opacity: 0.85
    });
    const cabin = new THREE.Mesh(cabinGeo, glassMat);
    cabin.position.set(0, 0.95, 0.2);
    cabin.castShadow = true;
    this.mesh.add(cabin);

    // Roof scoop / carbon intake
    const scoopGeo = new THREE.BoxGeometry(0.6, 0.15, 0.8);
    const darkMat = new THREE.MeshStandardMaterial({ color: 0x111827, roughness: 0.5 });
    const scoop = new THREE.Mesh(scoopGeo, darkMat);
    scoop.position.set(0, 1.28, 0.3);
    this.mesh.add(scoop);

    // Rear Aero Wing / Spoiler
    const spoilerGeo = new THREE.BoxGeometry(2.2, 0.08, 0.5);
    const spoiler = new THREE.Mesh(spoilerGeo, darkMat);
    spoiler.position.set(0, 1.15, 1.9);
    spoiler.castShadow = true;
    this.mesh.add(spoiler);

    // Spoiler Struts
    const strutGeo = new THREE.BoxGeometry(0.08, 0.35, 0.15);
    for (const sx of [-0.65, 0.65]) {
      const strut = new THREE.Mesh(strutGeo, darkMat);
      strut.position.set(sx, 0.95, 1.9);
      this.mesh.add(strut);
    }

    // 2. High-Performance Alloy Wheels with Rubber Tires
    const wheelMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.9 });
    const rimMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, metalness: 0.9, roughness: 0.2 });

    const wheelGeo = new THREE.CylinderGeometry(0.38, 0.38, 0.32, 16);
    wheelGeo.rotateZ(Math.PI / 2);

    const rimGeo = new THREE.CylinderGeometry(0.24, 0.24, 0.33, 12);
    rimGeo.rotateZ(Math.PI / 2);

    const wheelOffsets = [
      { name: 'FL', x: -1.08, y: 0.38, z: -1.35, isFront: true },
      { name: 'FR', x: 1.08, y: 0.38, z: -1.35, isFront: true },
      { name: 'RL', x: -1.08, y: 0.38, z: 1.35, isFront: false },
      { name: 'RR', x: 1.08, y: 0.38, z: 1.35, isFront: false }
    ];

    for (const w of wheelOffsets) {
      const wheelAssembly = new THREE.Group();
      wheelAssembly.position.set(w.x, w.y, w.z);

      const tire = new THREE.Mesh(wheelGeo, wheelMat);
      tire.castShadow = true;
      wheelAssembly.add(tire);

      const rim = new THREE.Mesh(rimGeo, rimMat);
      wheelAssembly.add(rim);

      this.mesh.add(wheelAssembly);
      this.wheels.push(tire);

      if (w.isFront) {
        this.frontWheels.push(wheelAssembly);
      }
    }

    // 3. Dual Front Headlights with Real Light Beams
    const headlightBulbMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0x38bdf8,
      emissiveIntensity: 2.0
    });

    for (const hx of [-0.72, 0.72]) {
      const hGeo = new THREE.BoxGeometry(0.32, 0.12, 0.1);
      const hMesh = new THREE.Mesh(hGeo, headlightBulbMat);
      hMesh.position.set(hx, 0.52, -2.18);
      this.mesh.add(hMesh);

      const light = new THREE.SpotLight(0xf0f9ff, 2.5, 45, Math.PI / 6, 0.3, 1.2);
      light.position.set(hx, 0.55, -2.2);
      light.target.position.set(hx, 0, -18.0);
      this.mesh.add(light);
      this.mesh.add(light.target);
    }

    // 4. Dual Rear Taillights & Dynamic Brake Lights
    this.taillightMat = new THREE.MeshStandardMaterial({
      color: 0xef4444,
      emissive: 0xef4444,
      emissiveIntensity: 0.8
    });

    for (const tx of [-0.72, 0.72]) {
      const tGeo = new THREE.BoxGeometry(0.35, 0.12, 0.1);
      const tMesh = new THREE.Mesh(tGeo, this.taillightMat);
      tMesh.position.set(tx, 0.58, 2.18);
      this.mesh.add(tMesh);
    }

    // 5. Glowing Neon Underglow Light
    this.underglow = new THREE.PointLight(0x38bdf8, 2.0, 6);
    this.underglow.position.set(0, 0.2, 0);
    this.mesh.add(this.underglow);

    this.scene.add(this.mesh);
  }

  update(keys, dt, track) {
    // 1. Road status & friction
    this.isOffroad = !track.isPointOnRoad(this.position);
    this.inParkingGarage = track.isInsideParkingGarage(this.position);
    this.hitDeadEnd = track.isCollidingDeadEnd(this.position);

    const activeFriction = this.isOffroad ? this.offroadFriction : this.friction;
    const activeMaxSpeed = this.isOffroad ? this.maxSpeed * 0.45 : this.maxSpeed;

    // 2. Acceleration and Braking
    let isGas = false;
    let isBrake = false;
    let targetSteer = 0;

    if (keys.isAuto) {
      if (keys.isDriving) {
        const autoThrottle = Math.max(-1.0, Math.min(1.0, keys.throttle || 0));
        const autoSteering = Math.max(-1.0, Math.min(1.0, keys.steering || 0));

        // Model steering: negative = left (-1.0), positive = right (+1.0)
        // Car3D heading: left is positive turn, right is negative turn
        targetSteer = -autoSteering * 0.45;

        if (autoThrottle > 0.05) {
          isGas = true;
          this.speed += this.acceleration * autoThrottle * dt;
        } else if (autoThrottle < -0.05) {
          isBrake = true;
          const brakeAmt = Math.abs(autoThrottle);
          if (keys.allowReverse && this.speed <= 0.5) {
            // Reverse is an explicit manual-only control. Autonomous negative
            // throttle brakes to rest, so holding a parking brake cannot reverse.
            this.speed -= (this.acceleration * 0.6) * brakeAmt * dt;
          } else {
            const change = this.braking * brakeAmt * dt;
            this.speed = Math.sign(this.speed) * Math.max(0, Math.abs(this.speed) - change);
          }
        }
      } else {
        // Autonomous mode paused: rapidly damp speed to full stop
        if (Math.abs(this.speed) > 0.1) {
          isBrake = true;
          const damp = Math.sign(this.speed) * this.braking * 1.5 * dt;
          if (Math.abs(this.speed) <= Math.abs(damp)) {
            this.speed = 0;
          } else {
            this.speed -= damp;
          }
        } else {
          this.speed = 0;
        }
        targetSteer = 0;
      }
    } else {
      // Manual Keyboard Driving
      isGas = Boolean(keys.forward || keys.up);
      isBrake = Boolean(keys.reverse || keys.down);

      if (isGas) {
        this.speed += this.acceleration * dt;
      }
      if (isBrake) {
        if (this.speed > 0.5) {
          this.speed -= this.braking * dt; // Hard braking
        } else {
          this.speed -= (this.acceleration * 0.6) * dt; // Reverse
        }
      }

      if (keys.left) targetSteer += 0.45;
      if (keys.right) targetSteer -= 0.45;
    }

    // Dynamic brake light glow
    this.taillightMat.emissiveIntensity = isBrake ? 2.5 : 0.6;

    // Natural friction drag
    if (!isGas && !isBrake) {
      if (this.speed > 0) {
        this.speed = Math.max(0, this.speed - activeFriction * dt);
      } else if (this.speed < 0) {
        this.speed = Math.min(0, this.speed + activeFriction * dt);
      }
    }

    // Clamp speed limits
    this.speed = Math.max(this.maxReverseSpeed, Math.min(activeMaxSpeed, this.speed));

    // 3. Smooth steering wheel lag
    this.steerAngle += (targetSteer - this.steerAngle) * 12.0 * dt;

    // Turn front visual wheels
    for (const fw of this.frontWheels) {
      fw.rotation.y = this.steerAngle;
    }

    // Car heading turns proportionally to speed direction
    if (Math.abs(this.speed) > 0.2) {
      const speedFactor = Math.min(1.0, Math.abs(this.speed) / 10.0);
      const direction = this.speed > 0 ? 1 : -1;
      this.heading += this.steerAngle * this.turnSpeed * speedFactor * direction * dt;
    }

    // 4. Integrate Position
    const moveZ = -Math.cos(this.heading) * this.speed * dt;
    const moveX = -Math.sin(this.heading) * this.speed * dt;

    this.position.x += moveX;
    this.position.z += moveZ;

    // Solid obstacle & boundary collision resolution
    this.hitWall = false;
    if (this.track && this.track.resolveCollision) {
      const col = this.track.resolveCollision(this.position, 1.1);
      if (col && col.collided) {
        this.hitWall = true;
        if (col.type === 'dead_end') this.hitDeadEnd = true;
        // Elastic rebound bounce
        this.speed = -this.speed * 0.45;
      }
    }

    // Soft Boundary collision rebound at dead-end crash walls
    if (this.hitDeadEnd && !this.hitWall) {
      this.speed = -this.speed * 0.35; // Bounce back
      this.position.x -= moveX * 2.0;
      this.position.z -= moveZ * 2.0;
    }

    // Distance accumulator
    this.distanceTraveled += Math.abs(this.speed) * dt;

    // 5. Spin all 4 wheels
    const rollAngle = (this.speed / 0.38) * dt;
    for (const w of this.wheels) {
      w.rotation.x += rollAngle;
    }

    // 6. Update visual mesh transform
    this.mesh.position.copy(this.position);
    this.mesh.rotation.y = this.heading;

    // These flags describe the returned, post-action position. Collision flags
    // above retain events even if resolution moved the vehicle off the obstacle.
    this.isOffroad = !track.isPointOnRoad(this.position);
    this.inParkingGarage = track.isInsideParkingGarage(this.position);
    this.hitDeadEnd = this.hitDeadEnd || track.isCollidingDeadEnd(this.position);
  }

  getState() {
    return {
      x: this.position.x, y: this.position.y, z: this.position.z,
      heading: this.heading, speed: this.speed, steerAngle: this.steerAngle,
      distanceTraveled: this.distanceTraveled,
      isOffroad: this.isOffroad, inParkingGarage: this.inParkingGarage,
      hitDeadEnd: this.hitDeadEnd, hitWall: this.hitWall
    };
  }

  reset(state = {}) {
    const next = {
      x: this.track.spawnPos.x, y: this.track.spawnPos.y, z: this.track.spawnPos.z,
      heading: this.track.spawnHeading, speed: 0, steerAngle: 0, distanceTraveled: 0
    };
    for (const name of Object.keys(next)) {
      if (state[name] !== undefined) {
        if (!Number.isFinite(state[name])) throw new TypeError(`Invalid vehicle state: ${name}`);
        next[name] = state[name];
      }
    }
    this.position.set(next.x, next.y, next.z);
    this.heading = next.heading;
    this.speed = next.speed;
    this.steerAngle = next.steerAngle;
    this.distanceTraveled = next.distanceTraveled;
    this.isOffroad = !this.track.isPointOnRoad(this.position);
    this.inParkingGarage = this.track.isInsideParkingGarage(this.position);
    this.hitDeadEnd = this.track.isCollidingDeadEnd(this.position);
    this.hitWall = false;

    for (const fw of this.frontWheels) {
      fw.rotation.y = this.steerAngle;
    }
    for (const wheel of this.wheels) wheel.rotation.x = 0;
    this.taillightMat.emissiveIntensity = 0.6;

    this.mesh.position.copy(this.position);
    this.mesh.rotation.y = this.heading;
    return this.getState();
  }

  setColor(hexColor) {
    this.bodyMat.color.setHex(hexColor);
    this.underglow.color.setHex(hexColor);
  }
}
