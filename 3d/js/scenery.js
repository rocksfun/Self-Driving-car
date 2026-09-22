// --------------------------------------------------------------------------
// NeuroDrive 3D — Optimized Scenery, Nature, Buildings & Props
// Guaranteed 100% OFF-ROAD placement (all drivable lanes completely clear)
// High-performance batching: InstancedMesh for windows, zero per-frame light leaks
// --------------------------------------------------------------------------

class Scenery3D {
  constructor(scene, track) {
    this.scene = scene;
    this.track = track;
    this.group = new THREE.Group();
    this.scene.add(this.group);

    // Materials Library
    this.initMaterials();

    // Procedural Generation with Performance Optimizations
    this.buildTerrain();
    this.buildBuildings();
    this.buildTreesAndBushes();
    this.buildSideObstacles();
  }

  initMaterials() {
    // Terrain Grass
    this.grassMat = new THREE.MeshStandardMaterial({
      color: 0x166534,
      roughness: 0.95,
      metalness: 0.05
    });

    // Tree Trunks
    this.trunkMat = new THREE.MeshStandardMaterial({
      color: 0x5c3a21,
      roughness: 0.85
    });

    // Tree Foliage Greens
    this.foliageMat1 = new THREE.MeshStandardMaterial({
      color: 0x22c55e,
      roughness: 0.7
    });

    this.foliageMat2 = new THREE.MeshStandardMaterial({
      color: 0x15803d,
      roughness: 0.8
    });

    this.pineFoliageMat = new THREE.MeshStandardMaterial({
      color: 0x166534,
      roughness: 0.8
    });

    // Building Wall Materials
    this.concreteWallMat = new THREE.MeshStandardMaterial({
      color: 0x334155,
      roughness: 0.7,
      metalness: 0.2
    });

    this.glassBuildingMat = new THREE.MeshStandardMaterial({
      color: 0x0284c7,
      roughness: 0.2,
      metalness: 0.8
    });

    this.brickBuildingMat = new THREE.MeshStandardMaterial({
      color: 0x7c2d12,
      roughness: 0.85
    });

    // Emissive Window Material (glows at night)
    this.windowMat = new THREE.MeshStandardMaterial({
      color: 0xfef08a,
      emissive: 0xfde047,
      emissiveIntensity: 0.5,
      roughness: 0.3
    });

    // Prop Materials: Crates, Barrels, Rocks
    this.woodCrateMat = new THREE.MeshStandardMaterial({
      color: 0xb45309,
      roughness: 0.85
    });

    this.oilDrumMat = new THREE.MeshStandardMaterial({
      color: 0x2563eb,
      metalness: 0.7,
      roughness: 0.3
    });

    this.hazardDrumMat = new THREE.MeshStandardMaterial({
      color: 0xdc2626,
      metalness: 0.6,
      roughness: 0.4
    });

    this.boulderMat = new THREE.MeshStandardMaterial({
      color: 0x475569,
      roughness: 0.95
    });

    // Streetlamp Materials
    this.lampPostMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      metalness: 0.8,
      roughness: 0.3
    });

    this.lampBulbMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0xfde047,
      emissiveIntensity: 0.0
    });
  }

  buildTerrain() {
    // Large ground plane (450m x 450m) with gentle elevation away from roads
    const terrainGeo = new THREE.PlaneGeometry(450, 450, 48, 48);
    terrainGeo.rotateX(-Math.PI / 2);

    const posAttr = terrainGeo.attributes.position;
    for (let i = 0; i < posAttr.count; i++) {
      const x = posAttr.getX(i);
      const z = posAttr.getZ(i);

      const distToRoad = this.distanceToRoad(x, z);
      if (distToRoad > 16.0) {
        const factor = Math.min(1.0, (distToRoad - 16.0) / 40.0);
        const hill1 = Math.sin(x * 0.03) * Math.cos(z * 0.03) * 3.5;
        const hill2 = Math.sin(x * 0.06 + z * 0.04) * 1.8;
        posAttr.setY(i, (hill1 + hill2) * factor);
      } else {
        posAttr.setY(i, 0);
      }
    }
    terrainGeo.computeVertexNormals();

    const terrainMesh = new THREE.Mesh(terrainGeo, this.grassMat);
    terrainMesh.receiveShadow = true;
    this.group.add(terrainMesh);
  }

  buildBuildings() {
    // Horizon Skyline - spaced well outside the road and perimeter
    const skylinePoints = [
      // North skyline behind parking garage
      { x: -55, z: -190, w: 22, d: 24, h: 52, type: 'glass' },
      { x: 55, z: -195, w: 26, d: 22, h: 64, type: 'glass' },
      { x: 0, z: -215, w: 32, d: 28, h: 76, type: 'glass' },
      { x: -95, z: -180, w: 24, d: 20, h: 44, type: 'concrete' },
      { x: 95, z: -175, w: 28, d: 24, h: 48, type: 'concrete' },

      // East industrial park around Exit 1 Dead-End
      { x: 98, z: -25, w: 24, d: 38, h: 18, type: 'warehouse' },
      { x: 102, z: 28, w: 26, d: 34, h: 22, type: 'warehouse' },
      { x: 125, z: 0, w: 22, d: 28, h: 36, type: 'concrete' },
      { x: 78, z: 42, w: 18, d: 22, h: 16, type: 'brick' },

      // West construction / industrial zone around Exit 3 Dead-End
      { x: -100, z: -26, w: 26, d: 36, h: 20, type: 'warehouse' },
      { x: -104, z: 28, w: 28, d: 32, h: 24, type: 'warehouse' },
      { x: -128, z: 0, w: 24, d: 30, h: 38, type: 'concrete' },
      { x: -78, z: 42, w: 20, d: 22, h: 18, type: 'brick' },

      // South suburbs / office complexes along the straightaway
      { x: -38, z: 65, w: 16, d: 18, h: 26, type: 'concrete' },
      { x: 38, z: 65, w: 18, d: 18, h: 28, type: 'concrete' },
      { x: -42, z: 95, w: 20, d: 24, h: 32, type: 'glass' },
      { x: 44, z: 95, w: 22, d: 22, h: 34, type: 'glass' },
      { x: -68, z: 50, w: 20, d: 20, h: 18, type: 'brick' },
      { x: 68, z: 50, w: 20, d: 20, h: 22, type: 'brick' },

      // North-West & North-East corners of scenic route
      { x: -55, z: -85, w: 20, d: 22, h: 36, type: 'concrete' },
      { x: 55, z: -85, w: 22, d: 20, h: 40, type: 'glass' },
      { x: 65, z: -130, w: 24, d: 26, h: 48, type: 'glass' },
      { x: -52, z: -135, w: 22, d: 24, h: 42, type: 'concrete' }
    ];

    const windowMatrices = [];
    const dummy = new THREE.Object3D();

    for (const b of skylinePoints) {
      this.createBuilding(b.x, b.z, b.w, b.d, b.h, b.type, windowMatrices, dummy);
    }

    // High-performance single InstancedMesh for ALL windows in the city!
    if (windowMatrices.length > 0) {
      const winGeo = new THREE.PlaneGeometry(1.4, 1.6);
      const instancedWindows = new THREE.InstancedMesh(winGeo, this.windowMat, windowMatrices.length);
      for (let i = 0; i < windowMatrices.length; i++) {
        instancedWindows.setMatrixAt(i, windowMatrices[i]);
      }
      instancedWindows.instanceMatrix.needsUpdate = true;
      this.group.add(instancedWindows);
      this.instancedWindows = instancedWindows;
    }
  }

  createBuilding(x, z, w, d, h, type, windowMatrices, dummy) {
    const building = new THREE.Group();
    building.position.set(x, h / 2, z);

    let mainMat = this.concreteWallMat;
    if (type === 'glass') mainMat = this.glassBuildingMat;
    if (type === 'brick') mainMat = this.brickBuildingMat;
    if (type === 'warehouse') {
      mainMat = new THREE.MeshStandardMaterial({
        color: 0x475569,
        roughness: 0.6,
        metalness: 0.4
      });
    }

    // Main Structure Block
    const bodyGeo = new THREE.BoxGeometry(w, h, d);
    const bodyMesh = new THREE.Mesh(bodyGeo, mainMat);
    bodyMesh.castShadow = true;
    bodyMesh.receiveShadow = true;
    building.add(bodyMesh);

    // Rooftop HVAC Unit
    const hvacGeo = new THREE.BoxGeometry(w * 0.4, 1.8, d * 0.4);
    const hvacMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.8 });
    const hvac = new THREE.Mesh(hvacGeo, hvacMat);
    hvac.position.set(0, h / 2 + 0.9, 0);
    building.add(hvac);

    // Collect Window Positions for the Global InstancedMesh
    if (type !== 'warehouse') {
      const floors = Math.floor(h / 4.0);
      const cols = Math.floor(w / 3.5);
      for (let f = 1; f < floors; f++) {
        for (let c = 1; c < cols; c++) {
          if (Math.random() > 0.25) {
            dummy.position.set(
              x - w / 2 + c * (w / cols),
              f * 4.0,
              z + d / 2 + 0.05
            );
            dummy.rotation.set(0, 0, 0);
            dummy.scale.set(1, 1, 1);
            dummy.updateMatrix();
            windowMatrices.push(dummy.matrix.clone());
          }
        }
      }
    }

    this.group.add(building);
  }

  buildTreesAndBushes() {
    // Trees placed strictly in meadows >= 12.0m away from road
    const treeCandidates = [
      // South sector groves (well away from straight road at x=0)
      [-18, 55], [18, 55], [-20, 75], [20, 75],
      [-28, 50], [28, 50], [-30, 70], [30, 70], [-30, 90], [30, 90],

      // East Sector (North & South of Exit 1)
      [48, -24], [65, -25], [78, -24], [52, -38], [68, -40],
      [48, 24], [65, 25], [78, 24], [52, 38], [68, 40],

      // West Sector (North & South of Exit 3)
      [-48, -24], [-65, -25], [-78, -24], [-52, -38], [-68, -40],
      [-48, 24], [-65, 25], [-78, 24], [-52, 38], [-68, 40],

      // North Sector (Meadows safely outside S-curve)
      [28, -42], [-28, -42], [42, -60], [-25, -65], [45, -80],
      [38, -100], [-35, -100], [35, -125], [-35, -125], [28, -145], [-28, -145],

      // Distant woods
      [-80, 80], [80, 80], [-85, 100], [85, 100], [-90, -80], [90, -80],
      [-70, -120], [70, -120], [-60, -160], [60, -160]
    ];

    for (const [x, z] of treeCandidates) {
      if (this.distanceToRoad(x, z) >= 12.0) {
        const scale = 0.8 + Math.random() * 0.5;
        const isPine = Math.random() > 0.5;
        if (isPine) {
          this.createPineTree(x, z, scale);
        } else {
          this.createDeciduousTree(x, z, scale);
        }
      }
    }

    // Bushes ONLY when >= 12.0 meters from road
    const bushPositions = [
      [-16, 45], [16, 45], [-16, 65], [16, 65],
      [38, 16], [38, -16], [-38, 16], [-38, -16],
      [28, -50], [-26, -55], [38, -75], [-30, -85], [32, -115], [-30, -115]
    ];

    for (const [bx, bz] of bushPositions) {
      if (this.distanceToRoad(bx, bz) >= 12.0) {
        this.createBush(bx, bz, 0.6 + Math.random() * 0.4);
      }
    }
  }

  createDeciduousTree(x, z, scale = 1.0) {
    const tree = new THREE.Group();
    tree.position.set(x, 0, z);
    tree.scale.set(scale, scale, scale);

    // Trunk
    const trunkGeo = new THREE.CylinderGeometry(0.3, 0.45, 3.2, 6);
    const trunk = new THREE.Mesh(trunkGeo, this.trunkMat);
    trunk.position.y = 1.6;
    trunk.castShadow = true;
    tree.add(trunk);

    // Multi-layered canopy
    const foliageMat = Math.random() > 0.5 ? this.foliageMat1 : this.foliageMat2;
    const crownGeo = new THREE.DodecahedronGeometry(2.0, 1);
    const crown = new THREE.Mesh(crownGeo, foliageMat);
    crown.position.set(0, 4.2, 0);
    crown.castShadow = true;
    tree.add(crown);

    this.group.add(tree);
  }

  createPineTree(x, z, scale = 1.0) {
    const tree = new THREE.Group();
    tree.position.set(x, 0, z);
    tree.scale.set(scale, scale, scale);

    // Trunk
    const trunkGeo = new THREE.CylinderGeometry(0.25, 0.35, 2.5, 6);
    const trunk = new THREE.Mesh(trunkGeo, this.trunkMat);
    trunk.position.y = 1.25;
    trunk.castShadow = true;
    tree.add(trunk);

    // Pine Foliage Cone
    const coneGeo = new THREE.ConeGeometry(2.2, 4.5, 6);
    const cone = new THREE.Mesh(coneGeo, this.pineFoliageMat);
    cone.position.y = 3.6;
    cone.castShadow = true;
    tree.add(cone);

    this.group.add(tree);
  }

  createBush(x, z, scale = 1.0) {
    const bushGeo = new THREE.DodecahedronGeometry(0.9 * scale, 0);
    const bush = new THREE.Mesh(bushGeo, this.foliageMat1);
    bush.position.set(x, 0.45 * scale, z);
    this.group.add(bush);
  }

  buildRoadsideStreetlights() {
    // Streetlights disabled to maintain an open track free of pole obstacles
  }

  createStreetlamp(x, z, rotationY) {
    // No-op
  }

  buildSideObstacles() {
    // Zero obstacles on or near any roads or tracks! Completely cleared.
  }

  createDrumStack(x, z, count, type = 'oil') {
    const group = new THREE.Group();
    group.position.set(x, 0, z);

    const mat = type === 'oil' ? this.oilDrumMat : this.hazardDrumMat;
    const drumGeo = new THREE.CylinderGeometry(0.42, 0.42, 1.2, 8);

    const drum1 = new THREE.Mesh(drumGeo, mat);
    drum1.position.set(0, 0.6, 0);
    group.add(drum1);

    if (count >= 2) {
      const drum2 = new THREE.Mesh(drumGeo, mat);
      drum2.position.set(0.75, 0.6, 0.2);
      group.add(drum2);
    }

    if (count >= 3) {
      const drum3 = new THREE.Mesh(drumGeo, mat);
      drum3.position.set(0.35, 0.6, 0.7);
      group.add(drum3);
    }

    if (count >= 4) {
      const drum4 = new THREE.Mesh(drumGeo, mat);
      drum4.position.set(0.35, 1.7, 0.35);
      group.add(drum4);
    }

    this.group.add(group);
  }

  createCrateStack(x, z) {
    const group = new THREE.Group();
    group.position.set(x, 0, z);

    const crateGeo = new THREE.BoxGeometry(1.4, 1.4, 1.4);

    const crate1 = new THREE.Mesh(crateGeo, this.woodCrateMat);
    crate1.position.set(0, 0.7, 0);
    group.add(crate1);

    const crate2 = new THREE.Mesh(crateGeo, this.woodCrateMat);
    crate2.position.set(1.2, 0.7, 0.2);
    group.add(crate2);

    const crate3 = new THREE.Mesh(crateGeo, this.woodCrateMat);
    crate3.position.set(0.5, 2.0, 0.1);
    group.add(crate3);

    this.group.add(group);
  }

  createBoulder(x, z, scale = 1.0) {
    const boulderGeo = new THREE.DodecahedronGeometry(1.2 * scale, 0);
    const boulder = new THREE.Mesh(boulderGeo, this.boulderMat);
    boulder.position.set(x, 0.7 * scale, z);
    boulder.rotation.set(Math.random(), Math.random(), Math.random());
    this.group.add(boulder);
  }

  // Calculate true minimum clearance distance from (x, z) to nearest road surface
  distanceToRoad(x, z) {
    // 1. Central Roundabout (inner radius 9.0, outer radius 29.5)
    const dCenter = Math.sqrt(x * x + z * z);
    if (dCenter >= this.track.innerRadius - 1.0 && dCenter <= this.track.outerRadius + 2.0) {
      return 0.0;
    }
    const dRoundabout = dCenter < this.track.innerRadius
      ? Math.max(0, this.track.innerRadius - dCenter)
      : Math.max(0, dCenter - this.track.outerRadius);

    // 2. South Entrance Road (x=0, halfW=5.5, z in [26, 88])
    let dSouth = Infinity;
    if (z >= 26.0 && z <= 88.0) {
      dSouth = Math.max(0, Math.abs(x) - this.track.halfW);
    }

    // 3. Exit 1 East Road (z=0, halfW=5.5, x in [26, 82])
    let dEast = Infinity;
    if (x >= 26.0 && x <= 82.0) {
      dEast = Math.max(0, Math.abs(z) - this.track.halfW);
    }

    // 4. Exit 3 West Road (z=0, halfW=5.5, x in [-82, -26])
    let dWest = Infinity;
    if (x <= -26.0 && x >= -82.0) {
      dWest = Math.max(0, Math.abs(z) - this.track.halfW);
    }

    // 5. Curvy S-Road
    let dCurvy = Infinity;
    if (this.track.curvyPoints && z <= -26.0 && z >= -155.0) {
      for (const p of this.track.curvyPoints) {
        const dx = x - p.x;
        const dz = z - p.z;
        const d = Math.max(0, Math.sqrt(dx * dx + dz * dz) - this.track.halfW);
        if (d < dCurvy) dCurvy = d;
      }
    }

    // 6. Parking Garage
    let dGarage = Infinity;
    if (this.track.parkingGarageZone) {
      const g = this.track.parkingGarageZone;
      const dx = Math.max(0, Math.abs(x - g.x) - g.width / 2);
      const dz = Math.max(0, Math.abs(z - g.z) - g.depth / 2);
      dGarage = Math.sqrt(dx * dx + dz * dz);
    }

    return Math.min(dRoundabout, dSouth, dEast, dWest, dCurvy, dGarage);
  }
}
