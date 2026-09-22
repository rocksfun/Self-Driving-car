// --------------------------------------------------------------------------
// NeuroDrive 3D — Track Mesh Generator
// Accurately replicates the 3-Exit Roundabout, Straight Entrance,
// Curvy Scenic Road, Dead-End Crash Sites, and Multi-Story Parking Garage
// --------------------------------------------------------------------------

class Track3D {
  constructor(scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    this.scene.add(this.group);

    // Track dimensions (1 unit in 3D = 10 units in 2D coordinate system)
    this.roadWidth = 11.0;
    this.halfW = this.roadWidth / 2; // 5.5m
    this.innerRadius = 9.0;
    this.outerRadius = 29.5;
    this.roundaboutCenter = new THREE.Vector3(0, 0, 0);

    // Spawn parameters (South straight road, facing North towards roundabout)
    this.spawnPos = new THREE.Vector3(0, 0.4, 76.0);
    this.spawnHeading = 0.0; // Facing North (-Z towards roundabout)

    // Boundaries for collision check & road status check
    this.roadSegments = [];
    this.deadEndZones = [];
    this.parkingGarageZone = null;

    // Materials
    this.initMaterials();

    // Build the complete 3D track layout
    this.buildRoundabout();
    this.buildSouthEntrance();
    this.buildExit1EastDeadEnd();
    this.buildExit3WestDeadEnd();
    this.buildExit2NorthCurvyRoad();
    this.buildParkingGarage();
  }

  initMaterials() {
    // High-quality dark asphalt
    this.asphaltMat = new THREE.MeshStandardMaterial({
      color: 0x1e2430,
      roughness: 0.85,
      metalness: 0.1
    });

    // Curbs (Alternating red & white race curbs)
    this.curbMat = new THREE.MeshStandardMaterial({
      color: 0xe2e8f0,
      roughness: 0.6,
      metalness: 0.2
    });

    this.curbRedMat = new THREE.MeshStandardMaterial({
      color: 0xef4444,
      roughness: 0.6,
      metalness: 0.2
    });

    // Yellow road markings
    this.yellowLineMat = new THREE.MeshBasicMaterial({
      color: 0xfacc15
    });

    // White dashed road markings
    this.whiteLineMat = new THREE.MeshBasicMaterial({
      color: 0xf8fafc
    });

    // Central Island Grass
    this.islandGrassMat = new THREE.MeshStandardMaterial({
      color: 0x15803d,
      roughness: 0.9,
      metalness: 0.05
    });

    // Crash Barrier Hazard Pattern
    this.hazardMat = new THREE.MeshStandardMaterial({
      color: 0xf59e0b,
      roughness: 0.4,
      metalness: 0.3
    });

    // Concrete Garage Material
    this.concreteMat = new THREE.MeshStandardMaterial({
      color: 0x64748b,
      roughness: 0.9,
      metalness: 0.15
    });
  }

  buildRoundabout() {
    const rIn = this.innerRadius;
    const rOut = this.outerRadius;
    const segments = 64;

    // 1. Asphalt Circulating Ring
    const ringGeo = new THREE.RingGeometry(rIn, rOut, segments);
    ringGeo.rotateX(-Math.PI / 2);
    const ringMesh = new THREE.Mesh(ringGeo, this.asphaltMat);
    ringMesh.position.y = 0.02;
    ringMesh.receiveShadow = true;
    this.group.add(ringMesh);

    // 2. Dashed Middle Lane Divider Ring
    const rMid = (rIn + rOut) / 2;
    const dashCount = 32;
    for (let i = 0; i < dashCount; i++) {
      const a1 = (i / dashCount) * Math.PI * 2;
      const a2 = ((i + 0.5) / dashCount) * Math.PI * 2;
      const arcGeo = new THREE.RingGeometry(rMid - 0.12, rMid + 0.12, 4, 1, a1, a2 - a1);
      arcGeo.rotateX(-Math.PI / 2);
      const arcMesh = new THREE.Mesh(arcGeo, this.whiteLineMat);
      arcMesh.position.y = 0.03;
      this.group.add(arcMesh);
    }

    // 3. Central Island Lawn
    const islandGeo = new THREE.CylinderGeometry(rIn, rIn, 0.5, 48);
    const islandMesh = new THREE.Mesh(islandGeo, this.islandGrassMat);
    islandMesh.position.set(0, 0.25, 0);
    islandMesh.receiveShadow = true;
    islandMesh.castShadow = true;
    this.group.add(islandMesh);

    // Impassable Elevated Central Barrier Wall around island
    const barrierHeight = 0.8;
    const barrierGeo = new THREE.CylinderGeometry(rIn, rIn, barrierHeight, 48, 1, true);
    const barrierMat = new THREE.MeshStandardMaterial({
      color: 0x334155,
      roughness: 0.7,
      metalness: 0.3
    });
    const barrierMesh = new THREE.Mesh(barrierGeo, barrierMat);
    barrierMesh.position.set(0, barrierHeight / 2 + 0.2, 0);
    barrierMesh.castShadow = true;
    barrierMesh.receiveShadow = true;
    this.group.add(barrierMesh);

    // Alternating Red & White Safety Hazard Top Rim
    const rimCount = 24;
    for (let i = 0; i < rimCount; i++) {
      const a1 = (i / rimCount) * Math.PI * 2;
      const a2 = ((i + 1) / rimCount) * Math.PI * 2;
      const rimGeo = new THREE.RingGeometry(rIn - 0.5, rIn + 0.15, 3, 1, a1, a2 - a1);
      rimGeo.rotateX(-Math.PI / 2);
      const isRed = i % 2 === 0;
      const rimMesh = new THREE.Mesh(rimGeo, isRed ? this.curbRedMat : this.whiteLineMat);
      rimMesh.position.y = barrierHeight + 0.21;
      this.group.add(rimMesh);
    }

    // 4. Outer Roundabout Curbs between the 4 Road Openings
    // Arcs connect the road exits with raised circular boundary curbs
    const outerArcAngles = [
      { start: 0.22, end: 1.35 },    // Quadrant 1: East to South
      { start: 1.79, end: 2.92 },    // Quadrant 2: South to West
      { start: 3.36, end: 4.49 },    // Quadrant 3: West to North
      { start: 4.93, end: 6.06 }     // Quadrant 4: North to East
    ];
    for (const arc of outerArcAngles) {
      const curbArcGeo = new THREE.RingGeometry(rOut, rOut + 0.6, 24, 1, arc.start, arc.end - arc.start);
      curbArcGeo.rotateX(-Math.PI / 2);
      const curbArcMesh = new THREE.Mesh(curbArcGeo, this.curbMat);
      curbArcMesh.position.y = 0.16;
      curbArcMesh.receiveShadow = true;
      this.group.add(curbArcMesh);

      const lipGeo = new THREE.RingGeometry(rOut + 0.45, rOut + 0.6, 24, 1, arc.start, arc.end - arc.start);
      lipGeo.rotateX(-Math.PI / 2);
      const lipMesh = new THREE.Mesh(lipGeo, this.curbRedMat);
      lipMesh.position.y = 0.28;
      this.group.add(lipMesh);
    }

    // Central Monument / Modern Fountain Sculpture
    const monumentBaseGeo = new THREE.CylinderGeometry(3.5, 4.0, 0.8, 16);
    const monumentBaseMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.5 });
    const monumentBase = new THREE.Mesh(monumentBaseGeo, monumentBaseMat);
    monumentBase.position.set(0, 0.9, 0);
    monumentBase.castShadow = true;
    this.group.add(monumentBase);

    // Glowing futuristic spire in center
    const spireGeo = new THREE.ConeGeometry(1.2, 7.0, 6);
    const spireMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      emissive: 0x0284c7,
      emissiveIntensity: 0.6,
      roughness: 0.2,
      metalness: 0.8
    });
    const spire = new THREE.Mesh(spireGeo, spireMat);
    spire.position.set(0, 4.8, 0);
    spire.castShadow = true;
    this.group.add(spire);

    // Light emitting from central spire
    const monumentLight = new THREE.PointLight(0x38bdf8, 1.5, 30);
    monumentLight.position.set(0, 5, 0);
    this.group.add(monumentLight);
  }

  buildSouthEntrance() {
    const halfW = this.halfW;
    const startZ = 85.0;
    // Overlap 2.0m inside roundabout ring for seamless junction without gaps
    const endZ = 27.5;
    const length = startZ - endZ;

    // Asphalt Road
    const roadGeo = new THREE.PlaneGeometry(this.roadWidth, length);
    roadGeo.rotateX(-Math.PI / 2);
    const roadMesh = new THREE.Mesh(roadGeo, this.asphaltMat);
    roadMesh.position.set(0, 0.02, (startZ + endZ) / 2);
    roadMesh.receiveShadow = true;
    this.group.add(roadMesh);

    // Double Yellow Centerline
    for (const offset of [-0.15, 0.15]) {
      const lineGeo = new THREE.PlaneGeometry(0.18, length);
      lineGeo.rotateX(-Math.PI / 2);
      const lineMesh = new THREE.Mesh(lineGeo, this.yellowLineMat);
      lineMesh.position.set(offset, 0.03, (startZ + endZ) / 2);
      this.group.add(lineMesh);
    }

    // White Edge Lines
    for (const sign of [-1, 1]) {
      const edgeGeo = new THREE.PlaneGeometry(0.2, length);
      edgeGeo.rotateX(-Math.PI / 2);
      const edgeMesh = new THREE.Mesh(edgeGeo, this.whiteLineMat);
      edgeMesh.position.set(sign * (halfW - 0.4), 0.03, (startZ + endZ) / 2);
      this.group.add(edgeMesh);

      // Sidewalk Curbs
      const curbGeo = new THREE.BoxGeometry(0.6, 0.3, length);
      const curbMesh = new THREE.Mesh(curbGeo, this.curbMat);
      curbMesh.position.set(sign * (halfW + 0.3), 0.15, (startZ + endZ) / 2);
      curbMesh.castShadow = true;
      curbMesh.receiveShadow = true;
      this.group.add(curbMesh);
    }

    // Start / Finish Checkered Line at Z = 70
    const checkerW = this.roadWidth;
    const checkerL = 1.6;
    const squares = 14;
    const sqW = checkerW / squares;
    for (let i = 0; i < squares; i++) {
      for (let j = 0; j < 2; j++) {
        const isWhite = (i + j) % 2 === 0;
        const sqGeo = new THREE.PlaneGeometry(sqW, checkerL / 2);
        sqGeo.rotateX(-Math.PI / 2);
        const sqMat = isWhite ? this.whiteLineMat : this.asphaltMat;
        const sqMesh = new THREE.Mesh(sqGeo, sqMat);
        sqMesh.position.set(-halfW + (i + 0.5) * sqW, 0.035, 70 - (j - 0.5) * (checkerL / 2));
        this.group.add(sqMesh);
      }
    }
  }

  buildExit1EastDeadEnd() {
    const halfW = this.halfW;
    // Overlap 2.0m inside roundabout ring for seamless junction
    const startX = 27.5;
    const endX = 80.0;
    const length = endX - startX;

    // Asphalt Road
    const roadGeo = new THREE.PlaneGeometry(length, this.roadWidth);
    roadGeo.rotateX(-Math.PI / 2);
    const roadMesh = new THREE.Mesh(roadGeo, this.asphaltMat);
    roadMesh.position.set((startX + endX) / 2, 0.02, 0);
    roadMesh.receiveShadow = true;
    this.group.add(roadMesh);

    // Dashed White Centerline
    const dashLen = 2.0;
    const gapLen = 2.0;
    for (let x = startX + 2; x < endX - 4; x += (dashLen + gapLen)) {
      const lineGeo = new THREE.PlaneGeometry(dashLen, 0.25);
      lineGeo.rotateX(-Math.PI / 2);
      const lineMesh = new THREE.Mesh(lineGeo, this.whiteLineMat);
      lineMesh.position.set(x + dashLen / 2, 0.03, 0);
      this.group.add(lineMesh);
    }

    // Curbs along edges
    for (const sign of [-1, 1]) {
      const curbGeo = new THREE.BoxGeometry(length, 0.3, 0.6);
      const curbMesh = new THREE.Mesh(curbGeo, this.curbMat);
      curbMesh.position.set((startX + endX) / 2, 0.15, sign * (halfW + 0.3));
      curbMesh.castShadow = true;
      this.group.add(curbMesh);
    }

    // Dead-End Crash Wall at endX (80m)
    this.buildCrashBarrier(endX, 0, Math.PI / 2, "EXIT 1: DEAD END CRASH SITE");
    this.deadEndZones.push({ x: endX, z: 0, radius: 8 });
  }

  buildExit3WestDeadEnd() {
    const halfW = this.halfW;
    // Overlap 2.0m inside roundabout ring for seamless junction
    const startX = -27.5;
    const endX = -80.0;
    const length = Math.abs(endX - startX);

    // Asphalt Road
    const roadGeo = new THREE.PlaneGeometry(length, this.roadWidth);
    roadGeo.rotateX(-Math.PI / 2);
    const roadMesh = new THREE.Mesh(roadGeo, this.asphaltMat);
    roadMesh.position.set((startX + endX) / 2, 0.02, 0);
    roadMesh.receiveShadow = true;
    this.group.add(roadMesh);

    // Dashed White Centerline
    const dashLen = 2.0;
    const gapLen = 2.0;
    for (let x = startX - 2; x > endX + 4; x -= (dashLen + gapLen)) {
      const lineGeo = new THREE.PlaneGeometry(dashLen, 0.25);
      lineGeo.rotateX(-Math.PI / 2);
      const lineMesh = new THREE.Mesh(lineGeo, this.whiteLineMat);
      lineMesh.position.set(x - dashLen / 2, 0.03, 0);
      this.group.add(lineMesh);
    }

    // Curbs along edges
    for (const sign of [-1, 1]) {
      const curbGeo = new THREE.BoxGeometry(length, 0.3, 0.6);
      const curbMesh = new THREE.Mesh(curbGeo, this.curbMat);
      curbMesh.position.set((startX + endX) / 2, 0.15, sign * (halfW + 0.3));
      curbMesh.castShadow = true;
      this.group.add(curbMesh);
    }

    // Dead-End Crash Wall at endX (-80m)
    this.buildCrashBarrier(endX, 0, -Math.PI / 2, "EXIT 3: DEAD END CRASH SITE");
    this.deadEndZones.push({ x: endX, z: 0, radius: 8 });
  }

  buildExit2NorthCurvyRoad() {
    const halfW = this.halfW;

    // Spline curve keypoints scaled from 2D coordinates
    // (starts at Z = -27.5 inside roundabout ring for seamless connection)
    const keypoints = [
      new THREE.Vector3(0, 0, -27.5),
      new THREE.Vector3(0, 0, -38.0),
      new THREE.Vector3(4.0, 0, -50.0),
      new THREE.Vector3(17.0, 0, -64.0),   // Sweeping curve right
      new THREE.Vector3(23.0, 0, -80.0),
      new THREE.Vector3(12.0, 0, -96.0),   // S-curve back left
      new THREE.Vector3(-8.0, 0, -110.0),  // Sweeping curve left
      new THREE.Vector3(-15.0, 0, -124.0),
      new THREE.Vector3(-4.0, 0, -136.0),  // Straightening towards garage
      new THREE.Vector3(2.0, 0, -146.0),
      new THREE.Vector3(2.0, 0, -150.0)    // Garage entrance threshold
    ];

    this.curvyCurve = new THREE.CatmullRomCurve3(keypoints);
    this.curvyCurve.curveType = 'centripetal';

    const segments = 96;
    const points = this.curvyCurve.getPoints(segments);
    this.curvyPoints = points;

    // Build curved ribbon mesh for asphalt road
    const roadGeo = new THREE.BufferGeometry();
    const vertices = [];
    const uvs = [];
    const indices = [];

    for (let i = 0; i <= segments; i++) {
      const p = points[i];
      const t = i / segments;
      const tangent = this.curvyCurve.getTangent(t).normalize();
      // Normal vector in XZ plane: (-tangent.z, 0, tangent.x)
      const normal = new THREE.Vector3(-tangent.z, 0, tangent.x).normalize();

      const pLeft = p.clone().add(normal.clone().multiplyScalar(-halfW));
      const pRight = p.clone().add(normal.clone().multiplyScalar(halfW));

      vertices.push(pLeft.x, 0.02, pLeft.z);
      vertices.push(pRight.x, 0.02, pRight.z);

      uvs.push(0, t * 10);
      uvs.push(1, t * 10);

      if (i < segments) {
        const base = i * 2;
        indices.push(base, base + 1, base + 2);
        indices.push(base + 1, base + 3, base + 2);
      }
    }

    roadGeo.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    roadGeo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    roadGeo.setIndex(indices);
    roadGeo.computeVertexNormals();

    const curvyRoadMesh = new THREE.Mesh(roadGeo, this.asphaltMat);
    curvyRoadMesh.receiveShadow = true;
    this.group.add(curvyRoadMesh);

    // Dashed Centerline on Curvy Road
    for (let i = 0; i < segments; i += 2) {
      const p1 = points[i];
      const p2 = points[Math.min(i + 1, segments)];
      const dashGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(p1.x, 0.03, p1.z),
        new THREE.Vector3(p2.x, 0.03, p2.z)
      ]);
      const dashLine = new THREE.Line(dashGeo, new THREE.LineBasicMaterial({ color: 0xf8fafc, linewidth: 3 }));
      this.group.add(dashLine);
    }

    // High-performance InstancedMesh for red & white alternating race curbs
    const curbBox = new THREE.BoxGeometry(0.7, 0.25, 1.2);
    const redInstanced = new THREE.InstancedMesh(curbBox, this.curbRedMat, segments);
    const whiteInstanced = new THREE.InstancedMesh(curbBox, this.curbMat, segments);
    let redIdx = 0;
    let whiteIdx = 0;
    const dummyCurb = new THREE.Object3D();

    for (let i = 0; i < segments; i++) {
      const p = points[i];
      const t = i / segments;
      const tangent = this.curvyCurve.getTangent(t).normalize();
      const normal = new THREE.Vector3(-tangent.z, 0, tangent.x).normalize();
      const isRed = (i % 2 === 0);

      for (const sign of [-1, 1]) {
        const curbPos = p.clone().add(normal.clone().multiplyScalar(sign * (halfW + 0.35)));
        dummyCurb.position.set(curbPos.x, 0.12, curbPos.z);
        dummyCurb.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), tangent);
        dummyCurb.updateMatrix();

        if (isRed) {
          redInstanced.setMatrixAt(redIdx++, dummyCurb.matrix);
        } else {
          whiteInstanced.setMatrixAt(whiteIdx++, dummyCurb.matrix);
        }
      }
    }
    redInstanced.instanceMatrix.needsUpdate = true;
    whiteInstanced.instanceMatrix.needsUpdate = true;
    this.group.add(redInstanced);
    this.group.add(whiteInstanced);
  }

  buildParkingGarage() {
    const cx = 2.0;
    const cz = -170.0;
    const width = 34.0;
    const depth = 38.0;
    const height = 10.0;

    // Ground Floor Slab
    const floorGeo = new THREE.BoxGeometry(width, 0.4, depth);
    const floorMesh = new THREE.Mesh(floorGeo, this.concreteMat);
    floorMesh.position.set(cx, 0.2, cz);
    floorMesh.receiveShadow = true;
    this.group.add(floorMesh);

    // Second Floor Ceiling / Upper Deck
    const deckGeo = new THREE.BoxGeometry(width, 0.6, depth);
    const deckMesh = new THREE.Mesh(deckGeo, this.concreteMat);
    deckMesh.position.set(cx, height / 2, cz);
    deckMesh.castShadow = true;
    deckMesh.receiveShadow = true;
    this.group.add(deckMesh);

    // Roof Slab
    const roofGeo = new THREE.BoxGeometry(width, 0.6, depth);
    const roofMesh = new THREE.Mesh(roofGeo, this.concreteMat);
    roofMesh.position.set(cx, height, cz);
    roofMesh.castShadow = true;
    roofMesh.receiveShadow = true;
    this.group.add(roofMesh);

    // Support Pillars (perimeter and rear walls only - entrance completely open and clear)
    const pillarGeo = new THREE.CylinderGeometry(0.6, 0.6, height, 12);
    const pillarMat = new THREE.MeshStandardMaterial({ color: 0x475569, roughness: 0.8 });
    const pillarOffsets = [
      [-width / 2 + 2, -depth / 2 + 2],
      [width / 2 - 2, -depth / 2 + 2],
      [-width / 2 + 2, depth / 2 - 2],
      [width / 2 - 2, depth / 2 - 2],
      [-width / 2 + 2, 0],
      [width / 2 - 2, 0],
      [0, -depth / 2 + 2]
    ];

    for (const [ox, oz] of pillarOffsets) {
      const pillar = new THREE.Mesh(pillarGeo, pillarMat);
      pillar.position.set(cx + ox, height / 2, cz + oz);
      pillar.castShadow = true;
      this.group.add(pillar);
    }

    // Glowing Neon "PARKING GARAGE" Entrance Sign
    const signGeo = new THREE.BoxGeometry(14.0, 2.2, 0.6);
    const signMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      roughness: 0.2,
      metalness: 0.8
    });
    const signMesh = new THREE.Mesh(signGeo, signMat);
    signMesh.position.set(cx, height / 2 + 0.6, cz + depth / 2 + 0.3);
    this.group.add(signMesh);

    // Glowing Cyan Neon text plate
    const textPlateGeo = new THREE.PlaneGeometry(13.2, 1.6);
    const textPlateMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      emissive: 0x0284c7,
      emissiveIntensity: 0.8,
      roughness: 0.2
    });
    const textPlate = new THREE.Mesh(textPlateGeo, textPlateMat);
    textPlate.position.set(cx, height / 2 + 0.6, cz + depth / 2 + 0.65);
    this.group.add(textPlate);

    // Green Entry Arrow Lanterns
    const greenLight = new THREE.PointLight(0x10b981, 2.0, 25);
    greenLight.position.set(cx, height / 2 - 0.5, cz + depth / 2);
    this.group.add(greenLight);

    // Yellow Parking Bay Stripes on Ground Deck
    const bayCount = 6;
    for (let i = 0; i < bayCount; i++) {
      for (const side of [-1, 1]) {
        const stripeGeo = new THREE.PlaneGeometry(0.15, 6.0);
        stripeGeo.rotateX(-Math.PI / 2);
        const stripeMesh = new THREE.Mesh(stripeGeo, this.yellowLineMat);
        stripeMesh.position.set(cx + side * (width / 2 - 5), 0.42, cz - depth / 2 + 6 + i * 5);
        this.group.add(stripeMesh);
      }
    }

    // Parked Low-Poly Cars inside the garage for visual life
    this.buildParkedCar(cx - 10, 0.42, cz - 8, 0xef4444);
    this.buildParkedCar(cx - 10, 0.42, cz + 6, 0x38bdf8);
    this.buildParkedCar(cx + 10, 0.42, cz - 4, 0xf59e0b);
    this.buildParkedCar(cx + 10, 0.42, cz + 10, 0x10b981);

    // Record Parking Garage Zone for destination trigger
    this.parkingGarageZone = {
      x: cx,
      z: cz,
      width,
      depth,
      entryZ: cz + depth / 2
    };
  }

  buildParkedCar(x, y, z, colorHex) {
    const carGroup = new THREE.Group();
    carGroup.position.set(x, y, z);

    // Body
    const bodyGeo = new THREE.BoxGeometry(2.4, 0.9, 4.4);
    const bodyMat = new THREE.MeshStandardMaterial({ color: colorHex, metalness: 0.6, roughness: 0.3 });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.position.y = 0.6;
    body.castShadow = true;
    carGroup.add(body);

    // Cabin
    const cabinGeo = new THREE.BoxGeometry(2.0, 0.7, 2.4);
    const cabinMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.1 });
    const cabin = new THREE.Mesh(cabinGeo, cabinMat);
    cabin.position.set(0, 1.25, -0.2);
    carGroup.add(cabin);

    // Wheels
    const wheelGeo = new THREE.CylinderGeometry(0.4, 0.4, 0.3, 12);
    wheelGeo.rotateZ(Math.PI / 2);
    const wheelMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.9 });
    for (const wx of [-1.25, 1.25]) {
      for (const wz of [-1.3, 1.3]) {
        const wheel = new THREE.Mesh(wheelGeo, wheelMat);
        wheel.position.set(wx, 0.4, wz);
        carGroup.add(wheel);
      }
    }

    this.group.add(carGroup);
  }

  buildCrashBarrier(x, z, angle, titleText) {
    const barrierGroup = new THREE.Group();
    barrierGroup.position.set(x, 0, z);
    barrierGroup.rotation.y = angle;

    const w = this.roadWidth + 4;

    // Heavy Concrete Barrier Block
    const blockGeo = new THREE.BoxGeometry(w, 2.0, 2.4);
    const blockMat = new THREE.MeshStandardMaterial({
      color: 0x334155,
      roughness: 0.9,
      metalness: 0.1
    });
    const block = new THREE.Mesh(blockGeo, blockMat);
    block.position.y = 1.0;
    block.castShadow = true;
    barrierGroup.add(block);

    // Red & White Safety Stripes on Face
    const stripeCount = 10;
    const stripeW = w / stripeCount;
    for (let i = 0; i < stripeCount; i++) {
      const isRed = i % 2 === 0;
      const sGeo = new THREE.PlaneGeometry(stripeW, 1.6);
      const sMat = isRed ? this.curbRedMat : this.whiteLineMat;
      const sMesh = new THREE.Mesh(sGeo, sMat);
      sMesh.position.set(-w / 2 + (i + 0.5) * stripeW, 1.0, 1.22);
      barrierGroup.add(sMesh);
    }

    // Overhead Danger Billboard
    const signGeo = new THREE.BoxGeometry(w * 0.9, 1.8, 0.3);
    const signMat = new THREE.MeshStandardMaterial({
      color: 0xef4444,
      emissive: 0x991b1b,
      emissiveIntensity: 0.5
    });
    const signMesh = new THREE.Mesh(signGeo, signMat);
    signMesh.position.set(0, 3.2, 1.0);
    barrierGroup.add(signMesh);

    // Blinking Red Warning Beacons on top of barrier
    for (const bx of [-w / 3, 0, w / 3]) {
      const beaconGeo = new THREE.CylinderGeometry(0.3, 0.3, 0.6, 12);
      const beaconMat = new THREE.MeshStandardMaterial({
        color: 0xef4444,
        emissive: 0xf87171,
        emissiveIntensity: 1.5
      });
      const beacon = new THREE.Mesh(beaconGeo, beaconMat);
      beacon.position.set(bx, 4.3, 1.0);
      barrierGroup.add(beacon);

      const beaconLight = new THREE.PointLight(0xef4444, 2.0, 18);
      beaconLight.position.set(bx, 4.5, 1.2);
      barrierGroup.add(beaconLight);
    }

    this.group.add(barrierGroup);
  }

  buildOverheadGantry(x, z, label, colorHex = 0x38bdf8) {
    // Disabled to keep track completely clear of any roadside pillars or poles
  }

  // Helper: Query if position is on drivable road vs grass
  isPointOnRoad(pos) {
    const x = pos.x;
    const z = pos.z;
    const distCenter = Math.sqrt(x * x + z * z);

    // 1. In circulating roundabout
    if (distCenter >= this.innerRadius && distCenter <= this.outerRadius) {
      return true;
    }

    // 2. On South Entrance road
    if (z >= this.outerRadius - 2 && z <= 85.0 && Math.abs(x) <= this.halfW + 0.5) {
      return true;
    }

    // 3. On Exit 1 (East Dead-End)
    if (x >= this.outerRadius - 2 && x <= 80.0 && Math.abs(z) <= this.halfW + 0.5) {
      return true;
    }

    // 4. On Exit 3 (West Dead-End)
    if (x <= -this.outerRadius + 2 && x >= -80.0 && Math.abs(z) <= this.halfW + 0.5) {
      return true;
    }

    // 5. In Parking Garage
    if (this.parkingGarageZone) {
      const g = this.parkingGarageZone;
      if (Math.abs(x - g.x) <= g.width / 2 && Math.abs(z - g.z) <= g.depth / 2) {
        return true;
      }
    }

    // 6. On Exit 2 Curvy Road: check distance to nearest curve point
    if (z < -this.outerRadius + 2 && z > -152.0) {
      if (this.curvyPoints) {
        let minDistSq = Infinity;
        for (const p of this.curvyPoints) {
          const dx = x - p.x;
          const dz = z - p.z;
          const d2 = dx * dx + dz * dz;
          if (d2 < minDistSq) minDistSq = d2;
        }
        if (Math.sqrt(minDistSq) <= this.halfW + 0.8) {
          return true;
        }
      }
    }

    return false;
  }

  // Helper: Check if inside parking garage
  isInsideParkingGarage(pos) {
    if (!this.parkingGarageZone) return false;
    const g = this.parkingGarageZone;
    return (
      Math.abs(pos.x - g.x) <= g.width / 2 &&
      pos.z <= g.entryZ &&
      pos.z >= g.z - g.depth / 2
    );
  }

  // Helper: Check if colliding with dead-end barrier
  isCollidingDeadEnd(pos) {
    for (const d of this.deadEndZones) {
      const dx = pos.x - d.x;
      const dz = pos.z - d.z;
      if (Math.sqrt(dx * dx + dz * dz) <= d.radius) {
        return true;
      }
    }
    return false;
  }

  // --------------------------------------------------------------------------
  // Solid Collision Resolution Engine
  // Resolves penetrations and returns collision normals for bounce physics
  // --------------------------------------------------------------------------
  resolveCollision(pos, radius = 1.1) {
    let normalX = 0;
    let normalZ = 0;

    // 1. Central Roundabout Island Impassable Block
    const dCenter = Math.sqrt(pos.x * pos.x + pos.z * pos.z);
    const minCenterDist = this.innerRadius + radius; // 16.5 + 1.1 = 17.6m
    if (dCenter < minCenterDist) {
      const nx = pos.x / (dCenter || 1);
      const nz = pos.z / (dCenter || 1);
      pos.x = nx * minCenterDist;
      pos.z = nz * minCenterDist;
      return { collided: true, type: 'central_island', normal: { x: nx, z: nz } };
    }

    // 2. Dead-End Crash Sites (Exit 1 East & Exit 3 West)
    if (pos.x > 79.0 - radius && Math.abs(pos.z) <= this.halfW + 2.0) {
      pos.x = 79.0 - radius;
      return { collided: true, type: 'dead_end', normal: { x: -1, z: 0 } };
    }
    if (pos.x < -79.0 + radius && Math.abs(pos.z) <= this.halfW + 2.0) {
      pos.x = -79.0 + radius;
      return { collided: true, type: 'dead_end', normal: { x: 1, z: 0 } };
    }

    // 3. Parking Garage Obstacles & Walls (cz = -170, depth = 38, width = 34, cx = 2.0)
    if (this.parkingGarageZone) {
      const gz = this.parkingGarageZone;
      const xMin = gz.x - gz.width / 2; // -15.0
      const xMax = gz.x + gz.width / 2; // +19.0
      const zMin = gz.z - gz.depth / 2; // -189.0
      const zMax = gz.z + gz.depth / 2; // -151.0
      const doorLeft = -5.0;
      const doorRight = 9.0;

      // Inside or entering garage zone
      if (pos.z <= zMax + 3 && pos.z >= zMin - 4 && pos.x >= xMin - 4 && pos.x <= xMax + 4) {
        // Rear Wall
        if (pos.z < zMin + radius + 0.4) {
          pos.z = zMin + radius + 0.4;
          return { collided: true, type: 'garage_wall', normal: { x: 0, z: 1 } };
        }
        // Left Wall
        if (pos.x < xMin + radius + 0.4 && pos.z <= zMax) {
          pos.x = xMin + radius + 0.4;
          return { collided: true, type: 'garage_wall', normal: { x: 1, z: 0 } };
        }
        // Right Wall
        if (pos.x > xMax - radius - 0.4 && pos.z <= zMax) {
          pos.x = xMax - radius - 0.4;
          return { collided: true, type: 'garage_wall', normal: { x: -1, z: 0 } };
        }
        // Front doorway wings (outside the 14m entrance doorway)
        if (Math.abs(pos.z - zMax) < radius + 0.4) {
          if (pos.x < doorLeft) {
            pos.z = zMax + radius + 0.4;
            return { collided: true, type: 'garage_wall', normal: { x: 0, z: 1 } };
          } else if (pos.x > doorRight) {
            pos.z = zMax + radius + 0.4;
            return { collided: true, type: 'garage_wall', normal: { x: 0, z: 1 } };
          }
        }
        // Parked cars inside garage
        const parkedCarBoxes = [
          { minX: gz.x - 12.0, maxX: gz.x - 8.0, minZ: gz.z - 11.0, maxZ: gz.z - 5.0 },
          { minX: gz.x - 12.0, maxX: gz.x - 8.0, minZ: gz.z + 3.0, maxZ: gz.z + 9.0 },
          { minX: gz.x + 8.0, maxX: gz.x + 12.0, minZ: gz.z - 7.0, maxZ: gz.z - 1.0 },
          { minX: gz.x + 8.0, maxX: gz.x + 12.0, minZ: gz.z + 7.0, maxZ: gz.z + 13.0 }
        ];
        for (const pb of parkedCarBoxes) {
          if (pos.x > pb.minX - radius && pos.x < pb.maxX + radius &&
              pos.z > pb.minZ - radius && pos.z < pb.maxZ + radius) {
            const dLeft = Math.abs(pos.x - (pb.minX - radius));
            const dRight = Math.abs(pos.x - (pb.maxX + radius));
            const dBottom = Math.abs(pos.z - (pb.minZ - radius));
            const dTop = Math.abs(pos.z - (pb.maxZ + radius));
            const minD = Math.min(dLeft, dRight, dBottom, dTop);
            if (minD === dLeft) { pos.x = pb.minX - radius; normalX = -1; }
            else if (minD === dRight) { pos.x = pb.maxX + radius; normalX = 1; }
            else if (minD === dBottom) { pos.z = pb.minZ - radius; normalZ = -1; }
            else { pos.z = pb.maxZ + radius; normalZ = 1; }
            return { collided: true, type: 'parked_car', normal: { x: normalX, z: normalZ } };
          }
        }
      }
    }

    // 4. Outer World Boundary (prevents driving off into infinity beyond terrain)
    const maxWorldX = 140.0;
    const maxWorldZ = 110.0;
    const minWorldZ = -220.0;
    if (Math.abs(pos.x) > maxWorldX) {
      pos.x = Math.sign(pos.x) * maxWorldX;
      return { collided: true, type: 'world_edge', normal: { x: -Math.sign(pos.x), z: 0 } };
    }
    if (pos.z > maxWorldZ) {
      pos.z = maxWorldZ;
      return { collided: true, type: 'world_edge', normal: { x: 0, z: -1 } };
    }
    if (pos.z < minWorldZ) {
      pos.z = minWorldZ;
      return { collided: true, type: 'world_edge', normal: { x: 0, z: 1 } };
    }

    return { collided: false };
  }
}
