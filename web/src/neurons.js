/* The neuron cloud.
 *
 * One draw call for the whole brain. Each neuron is a point sprite whose size
 * and brightness are driven by a per-vertex activation attribute that the
 * application writes every frame; additive blending makes overlapping neurons
 * accumulate into the denser structures — the optic lobes, the mushroom body
 * calyx — the way they do in the real volume. */

import * as THREE from 'three';

import { densityFactor } from './pack.js';

const vertexShader = /* glsl */`
  attribute vec3 aColor;
  attribute float aSize;
  attribute float aAct;

  uniform float uScale;
  uniform float uTime;
  uniform vec3 uHot;

  varying vec3 vColor;
  varying float vAct;

  void main() {
    vAct = aAct;
    // Resting hue gives way to the ignition colour as activation rises, so a
    // firing neuron reads as the same cell that is now hot rather than as a
    // different category of thing.
    vColor = mix(aColor, uHot, smoothstep(0.30, 1.20, aAct));

    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    // A slow shimmer keeps the cloud from looking like a still image without
    // moving any neuron away from where it actually is.
    float breathe = 1.0 + 0.04 * sin(uTime * 0.6 + position.x * 0.05);
    float size = aSize * breathe * (1.0 + 5.0 * aAct);
    gl_PointSize = size * uScale / max(-mv.z, 1.0);
    gl_Position = projectionMatrix * mv;
  }
`;

const fragmentShader = /* glsl */`
  uniform float uOpacity;
  uniform float uDensity;

  varying vec3 vColor;
  varying float vAct;

  void main() {
    vec2 d = gl_PointCoord - 0.5;
    float r = dot(d, d) * 4.0;          // 0 at centre, 1 at the sprite edge
    if (r > 1.0) discard;

    // A tight core with a short halo. Tens of thousands of these blend
    // additively, so a wide falloff turns the whole frame into fog; the
    // brightness has to live in the middle few pixels instead.
    float halo = 1.0 - r;
    float core = pow(halo, 12.0);
    float glow = pow(halo, 3.0);
    // Resting brightness is scaled down for larger packs. Additive blending
    // means the summed alpha of a dense structure grows with the number of
    // neurons in it, so a level that reads well for six thousand neurons
    // clips the central brain to pure white at a hundred and forty thousand.
    float rest = uDensity * (uOpacity * glow + core * 0.34);
    float fired = vAct * (core * 1.9 + glow * 0.5) * clamp(mix(uDensity, 1.0, 0.5), 0.3, 1.3);
    float alpha = rest + fired;
    gl_FragColor = vec4(vColor * (1.0 + 2.6 * vAct), alpha);
  }
`;

export class NeuronCloud {
  constructor(pack) {
    this.pack = pack;
    this.n = pack.n;
    this.activation = new Float32Array(this.n);

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(pack.pos, 3));
    geometry.setAttribute('aColor', new THREE.BufferAttribute(new Float32Array(this.n * 3), 3));
    geometry.setAttribute('aSize', new THREE.BufferAttribute(this.sizes(), 1));
    const act = new THREE.BufferAttribute(this.activation, 1);
    act.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('aAct', act);
    geometry.computeBoundingSphere();

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uScale:   { value: 260 },
        uOpacity: { value: 0.20 },
        uDensity: { value: densityFactor(pack.n) },
        uTime:    { value: 0 },
        uHot:     { value: new THREE.Color(1.0, 0.92, 0.72) },
      },
      vertexShader,
      fragmentShader,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    this.points = new THREE.Points(geometry, this.material);
    this.points.frustumCulled = false;
    this.geometry = geometry;
  }

  /** Bigger points for neurons that drive more of the brain. */
  sizes() {
    const { indptr } = this.pack;
    const sizes = new Float32Array(this.n);
    for (let i = 0; i < this.n; i++) {
      const degree = indptr[i + 1] - indptr[i];
      sizes[i] = 0.55 + Math.min(1.9, Math.log1p(degree) * 0.34);
    }
    return sizes;
  }

  setColors(colors) {
    this.geometry.getAttribute('aColor').array.set(colors);
    this.geometry.getAttribute('aColor').needsUpdate = true;
  }

  setScale(v) { this.material.uniforms.uScale.value = 260 * v; }

  /** Light a set of neurons; levels are added to whatever is already there. */
  ignite(indices, levels, gain = 1) {
    const a = this.activation;
    for (let i = 0; i < indices.length; i++) {
      const at = indices[i];
      a[at] = Math.min(1.6, a[at] + levels[i] * gain);
    }
    this.dirty = true;
  }

  clear() {
    this.activation.fill(0);
    this.dirty = true;
  }

  /** Exponential decay, so a wavefront leaves a trail rather than a flash. */
  update(dt, time, halfLife = 1.15) {
    this.material.uniforms.uTime.value = time;
    const k = Math.pow(0.5, dt / halfLife);
    const a = this.activation;
    let live = 0;
    for (let i = 0; i < a.length; i++) {
      if (a[i] > 0.0015) { a[i] *= k; live++; }
      else if (a[i] !== 0) a[i] = 0;
    }
    if (this.dirty || live) {
      this.geometry.getAttribute('aAct').needsUpdate = true;
      this.dirty = false;
    }
    return live;
  }
}
