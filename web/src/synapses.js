/* The synapses a recall actually travelled along.
 *
 * When the server returns a wavefront it gives the neurons that fired at each
 * hop, not the connections between them. Those are recovered here from the CSR
 * adjacency the browser already holds: for every neuron that fired at hop t,
 * look at its real postsynaptic partners and keep the ones that fired at hop
 * t+1. The arcs on screen are therefore measured synapses, not a decorative
 * graph drawn between lit points.
 *
 * Excitatory and inhibitory connections are drawn differently because the sign
 * is real: it comes from the presynaptic neuron's predicted transmitter. */

import * as THREE from 'three';

import { densityFactor } from './pack.js';

const SEGMENTS = 14;
const MAX_ARCS = 2600;

const vertexShader = /* glsl */`
  attribute float aT;         // 0..1 along the arc
  attribute float aDelay;     // hop index, so arcs fire in sequence
  attribute vec3 aColor;

  uniform float uTime;
  uniform float uSpeed;
  uniform float uDensity;

  varying vec3 vColor;
  varying float vGlow;

  void main() {
    vColor = aColor;
    float age = (uTime - aDelay) * uSpeed;
    // A pulse runs from the presynaptic end to the postsynaptic end, then the
    // whole arc fades; before its hop arrives the arc is invisible.
    float head = clamp(age, 0.0, 1.6);
    float pulse = exp(-28.0 * pow(aT - head, 2.0));
    float trail = smoothstep(head + 0.06, head - 0.55, aT);
    float fade = exp(-max(0.0, age - 1.0) * 1.7) * step(0.0, age);
    vGlow = (0.16 * trail + 0.85 * pulse) * fade * uDensity;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const fragmentShader = /* glsl */`
  varying vec3 vColor;
  varying float vGlow;

  void main() {
    if (vGlow < 0.004) discard;
    gl_FragColor = vec4(vColor * (0.5 + vGlow), vGlow * 0.55);
  }
`;

const EXCITATORY = new THREE.Color(0.42, 0.74, 1.0);
const INHIBITORY = new THREE.Color(1.0, 0.36, 0.40);
const UP = new THREE.Vector3(0, 1, 0);
const SIDE = new THREE.Vector3(1, 0, 0);

export class SynapseArcs {
  constructor(pack, centre = new THREE.Vector3()) {
    this.pack = pack;
    this.centre = centre.clone();
    const max = MAX_ARCS * SEGMENTS * 2;

    this.geometry = new THREE.BufferGeometry();
    for (const [name, size] of [['position', 3], ['aColor', 3], ['aT', 1], ['aDelay', 1]]) {
      const attr = new THREE.BufferAttribute(new Float32Array(max * size), size);
      attr.setUsage(THREE.DynamicDrawUsage);
      this.geometry.setAttribute(name, attr);
    }
    this.geometry.setDrawRange(0, 0);

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uSpeed: { value: 2.6 },
        // Arcs crowd into the same small volume as the neurons do, so they
        // need the same density correction or the central brain clips.
        uDensity: { value: Math.min(1.15, densityFactor(pack.n)) },
      },
      vertexShader,
      fragmentShader,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    this.lines = new THREE.LineSegments(this.geometry, this.material);
    this.lines.frustumCulled = false;
    this.clock = 0;
  }

  set visible(v) { this.lines.visible = v; }

  clear() { this.geometry.setDrawRange(0, 0); }

  /** Rebuild the arc set from a list of wavefronts. */
  show(wavefronts) {
    const { pack } = this;
    const pos = this.geometry.getAttribute('position').array;
    const col = this.geometry.getAttribute('aColor').array;
    const tt = this.geometry.getAttribute('aT').array;
    const dd = this.geometry.getAttribute('aDelay').array;

    let vertex = 0;
    let arcs = 0;
    const a = new THREE.Vector3(), b = new THREE.Vector3();
    const mid = new THREE.Vector3(), ctrl = new THREE.Vector3();
    const p = new THREE.Vector3(), out = new THREE.Vector3();

    for (let hop = 0; hop + 1 < wavefronts.length && arcs < MAX_ARCS; hop++) {
      const from = wavefronts[hop];
      const next = new Map();
      wavefronts[hop + 1].index.forEach((idx, i) => next.set(idx, wavefronts[hop + 1].level[i]));

      for (let s = 0; s < from.index.length && arcs < MAX_ARCS; s++) {
        const pre = from.index[s];
        const targets = pack.targets(pre);
        const weights = pack.weights(pre);
        for (let e = 0; e < targets.length && arcs < MAX_ARCS; e++) {
          const post = targets[e];
          const level = next.get(post);
          if (level === undefined || level < 0.2) continue;

          a.fromArray(pack.pos, pre * 3);
          b.fromArray(pack.pos, post * 3);
          mid.addVectors(a, b).multiplyScalar(0.5);
          // Bow the arc so overlapping connections stay individually readable.
          // Bowing away from the centre of the brain looks best, but a
          // connection between the two hemispheres has its midpoint *at* the
          // centre, where that direction is undefined and would send the arc
          // off in an arbitrary direction. Fall back to an axis perpendicular
          // to the connection whenever the midpoint is too close to the middle.
          const span = a.distanceTo(b);
          out.subVectors(mid, this.centre);
          if (out.lengthSq() < 1e-4 || out.length() < span * 0.25) {
            out.subVectors(b, a).cross(UP);
            if (out.lengthSq() < 1e-6) out.subVectors(b, a).cross(SIDE);
          }
          ctrl.copy(mid).add(out.normalize().multiplyScalar(span * 0.16));

          const colour = weights[e] < 0 ? INHIBITORY : EXCITATORY;
          const bright = Math.min(1, 0.40 + level);

          for (let seg = 0; seg < SEGMENTS; seg++) {
            for (const end of [seg / SEGMENTS, (seg + 1) / SEGMENTS]) {
              quadratic(a, ctrl, b, end, p);
              pos[vertex * 3] = p.x; pos[vertex * 3 + 1] = p.y; pos[vertex * 3 + 2] = p.z;
              col[vertex * 3] = colour.r * bright;
              col[vertex * 3 + 1] = colour.g * bright;
              col[vertex * 3 + 2] = colour.b * bright;
              tt[vertex] = end;
              dd[vertex] = hop * 0.42;
              vertex++;
            }
          }
          arcs++;
        }
      }
    }

    for (const name of ['position', 'aColor', 'aT', 'aDelay']) {
      this.geometry.getAttribute(name).needsUpdate = true;
    }
    this.geometry.setDrawRange(0, vertex);
    this.clock = 0;
    this.material.uniforms.uTime.value = 0;
    return arcs;
  }

  update(dt) {
    this.clock += dt;
    this.material.uniforms.uTime.value = this.clock;
  }
}

function quadratic(p0, p1, p2, t, out) {
  const u = 1 - t;
  return out.set(
    u * u * p0.x + 2 * u * t * p1.x + t * t * p2.x,
    u * u * p0.y + 2 * u * t * p1.y + t * t * p2.y,
    u * u * p0.z + 2 * u * t * p1.z + t * t * p2.z,
  );
}
