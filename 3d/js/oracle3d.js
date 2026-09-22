// Expert policy for the rendered track. All action previews use Car3D via the bridge.
(function (root) {
  'use strict';
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));

  class NeuroDriveOracle {
    constructor(track, branch = 'right') {
      const T = root.THREE;
      this.branch = branch;
      const ingressRight = [[0,76],[0,40],[0,31],[3,24],[10,19],[18,12],
        [22,3],[21,-6],[16,-15],[8,-21],[2,-25],[0,-27.5]];
      const ingressLeft = [[0,76],[0,40],[0,31],[-3,24],[-10,19],[-18,12],
        [-22,3],[-21,-6],[-16,-15],[-8,-21],[-2,-25],[0,-27.5]];
      const ingress = branch === 'left' ? ingressLeft : ingressRight;
      const curve = new T.CatmullRomCurve3(ingress.map(([x,z]) => new T.Vector3(x,0,z)));
      curve.curveType = 'centripetal';
      this.points = curve.getSpacedPoints(550).map(p => ({x:p.x,z:p.z}));
      this.points.push(...track.curvyCurve.getSpacedPoints(850).slice(1).map(p=>({x:p.x,z:p.z})));
      for (let z=-150.2; z>=-164; z-=.2) this.points.push({x:2,z});
      this.distances = [0];
      for (let i=1;i<this.points.length;i++) {
        const a=this.points[i-1], b=this.points[i];
        this.distances.push(this.distances[i-1]+Math.hypot(b.x-a.x,b.z-a.z));
      }
      this.headings=this.points.map((p,i)=>{
        const a=this.points[Math.max(0,i-2)],b=this.points[Math.min(this.points.length-1,i+2)];
        return Math.atan2(-(b.x-a.x),-(b.z-a.z));
      });
      this._nearestX=NaN;
      this._nearestZ=NaN;
      this._nearestIndex=0;
    }

    nearest(state) {
      const x=state.x, z=state.z;
      // The path is fixed for this oracle. CTE, action and terminal checks often
      // query the same position, even when getState() returns a new object.
      if(x===this._nearestX && z===this._nearestZ) return this._nearestIndex;
      let best=0, distance=Infinity;
      for (let i=0;i<this.points.length;i++) {
        const p=this.points[i],d=(p.x-x)**2+(p.z-z)**2;
        if(d<distance){distance=d;best=i;}
      }
      this._nearestX=x;
      this._nearestZ=z;
      this._nearestIndex=best;
      return best;
    }

    cte(state) {
      const i=this.nearest(state), p=this.points[i];
      return Math.abs((state.x-p.x)*Math.cos(this.headings[i])-(state.z-p.z)*Math.sin(this.headings[i]));
    }

    static isSuccess(s) {
      return Boolean(s.inParkingGarage && !s.hitWall && !s.hitDeadEnd &&
        Math.abs(s.speed)<=.5 && Math.abs(wrap(s.heading))<=.2 &&
        Math.abs(s.x-2)<=2.5 && s.z<=-158);
    }

    isSuccess(s) { return NeuroDriveOracle.isSuccess(s); }

    terminalOutcome(s) {
      if(s.hitWall || s.hitDeadEnd) return {success:false,crashed:true,reason:'collision'};
      // Allow car to be pushed onto the grass (up to 14m CTE) to capture recovery maneuvers
      if(this.cte(s)>14.0) return {success:false,crashed:true,reason:'off_track'};
      if(this.isSuccess(s)) return {success:true,crashed:false,reason:'garage_success'};
      return {success:false,crashed:false,reason:null};
    }

    action(state) {
      if(this.isSuccess(state)) return [0,0];
      const speed = Math.max(0, state.speed);

      // Actuation latency compensation: project car position forward by 40ms transport delay
      const latencyTau = 0.040;
      const projX = state.x - Math.sin(state.heading) * speed * latencyTau;
      const projZ = state.z - Math.cos(state.heading) * speed * latencyTau;
      const projState = { x: projX, z: projZ, heading: state.heading, speed: speed };

      const i = this.nearest(projState);
      const lookahead = clamp(3.5 + speed * 0.45, 3.5, 12.0);
      let j = i;
      while(j < this.points.length - 1 && this.distances[j] - this.distances[i] < lookahead) j++;
      const target = this.points[j];
      const dx = target.x - projX, dz = target.z - projZ;
      const lateral = dx * Math.cos(state.heading) - dz * Math.sin(state.heading);
      const curvature = 2 * lateral / Math.max(4, dx * dx + dz * dz);
      const steering = clamp(curvature * Math.max(10, speed) / (0.45 * 2.4), -1, 1);

      const bend = Math.abs(wrap(this.headings[j] - this.headings[i])) / Math.max(1, this.distances[j] - this.distances[i]);
      const remaining = this.distances[this.distances.length - 1] - this.distances[i];

      // Open road speed: 12 m/s on straights, slowed by road curvature
      let desired = Math.min(12.5, Math.sqrt(3.2 / Math.max(0.015, bend)));

      // In the garage or garage approach (remaining < 25m), decelerate toward parking stop
      const isGarageZone = Boolean(state.inParkingGarage || remaining < 25.0);
      if (isGarageZone) {
        desired = Math.min(desired, Math.sqrt(2 * 3.2 * Math.max(0, remaining - 1.5)));
        if (remaining < 2.0) desired = 0;
      }

      const error = desired - speed;
      let throttle;

      if (isGarageZone && (desired === 0 || error < -1.0)) {
        // Controlled deceleration and final stop in garage
        if (desired === 0) {
          throttle = speed > 0.05 ? -clamp(speed * 3 / 36, 0.06, 0.7) : 0;
        } else {
          throttle = -clamp(-error * 3 / 36, 0.06, 0.5);
        }
      } else {
        // Open road / track: NEVER command negative throttle or hard braking.
        if (speed < 2.5) {
          throttle = 0.55; // Strong launch
        } else if (state.isOffroad) {
          throttle = 0.45; // Overcome grass friction
        } else if (error < 0) {
          // Coast: engine friction (12 m/s^2) naturally trims speed without braking
          throttle = 0.25;
        } else {
          // Accelerate to target speed
          throttle = clamp(Math.max(0.25, error * 3 / 24), 0.25, 0.70);
        }
      }

      return [steering, throttle];
    }

    spawnStart(rng = Math.random) {
      const offset = (rng() - 0.5) * 1.6;
      const headingNoise = (rng() - 0.5) * 0.04;
      return {
        state: {
          x: offset,
          y: 0.4,
          z: 76.0,
          heading: headingNoise,
          speed: 0.0,
          steerAngle: 0.0
        },
        sector: 'Start Line'
      };
    }

    spawn(episodeIndex, rng) {
      let fraction=0, sector='Start Straight';
      if(episodeIndex%3!==0) {
        const options=[[.20,'Roundabout Ingress'],[.35,'Roundabout Apex'],[.55,'S-Curves Entry'],
          [.72,'S-Curves Midpoint'],[.91,'Garage Approach']];
        [fraction,sector]=options[episodeIndex%options.length];
      }
      const distance=fraction*this.distances[this.distances.length-1];
      let i=this.distances.findIndex(d=>d>=distance); if(i<0)i=this.points.length-1;
      const p=this.points[i], h=this.headings[i], offset=(rng()-.5)*1.4;
      return {state:{x:p.x+Math.cos(h)*offset,y:.4,z:p.z-Math.sin(h)*offset,
        heading:h+(rng()-.5)*.16,speed:fraction===0?rng()*2:3+rng()*3,steerAngle:0},sector};
    }
  }
  root.NeuroDriveOracle=NeuroDriveOracle;
  if(typeof module!=='undefined') module.exports=NeuroDriveOracle;
})(typeof window!=='undefined'?window:globalThis);
