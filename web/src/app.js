/* Fluctfly — the viewer.
 *
 * Holds the scene, replays activation wavefronts hop by hop, and wires the
 * console to the memory API. */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

import { loadPack, extent } from './pack.js';
import { colorize, css } from './palette.js';
import { NeuronCloud } from './neurons.js';
import { SynapseArcs } from './synapses.js';

const $ = (id) => document.getElementById(id);

const state = {
  pack: null, cloud: null, arcs: null,
  replay: null,          // { wavefronts, hop, clock }
  hopInterval: 0.42,
};

boot();

async function boot() {
  const bar = document.querySelector('.boot-bar i');
  const pack = await loadPack((p) => { bar.style.width = `${Math.round(p * 100)}%`; });
  state.pack = pack;

  setupScene(pack);
  setupConsole(pack);

  $('stat-neurons').textContent = pack.n.toLocaleString();
  $('stat-edges').textContent = pack.edges.toLocaleString();
  $('stat-memories').textContent = pack.manifest.memories ?? 0;
  $('prov-dataset').textContent = pack.manifest.dataset ?? '';
  $('prov-mode').textContent = [
    `pack: ${pack.manifest.subset}`,
    `≥${pack.manifest.min_synapses} synapses`,
    `cue: ${pack.manifest.cue_encoding}`,
    pack.manifest.fused_with_fluctlightdb ? 'fused with FluctlightDB' : 'connectome only',
  ].join(' · ');

  $('boot').classList.add('done');
  setTimeout(() => $('boot').remove(), 800);
}

/* ---- scene --------------------------------------------------------------- */

let renderer, scene, camera, controls, composer, bloom, raycaster;

function setupScene(pack) {
  const canvas = $('stage');
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 1);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;

  scene = new THREE.Scene();

  state.cloud = new NeuronCloud(pack);
  const sphere = state.cloud.geometry.boundingSphere;
  const centre = sphere.center.clone();
  const radius = sphere.radius || extent(pack.pos);

  camera = new THREE.PerspectiveCamera(38, 1, radius * 0.01, radius * 24);
  // Back off far enough that the bounding sphere fits the vertical field of
  // view with a margin, instead of guessing a multiple of the radius.
  const fit = (radius / Math.sin((camera.fov * Math.PI) / 360)) * 1.12;
  camera.position.copy(centre).add(
    new THREE.Vector3(0.52, 0.26, 0.81).normalize().multiplyScalar(fit),
  );

  controls = new OrbitControls(camera, canvas);
  controls.target.copy(centre);
  controls.enableDamping = true;
  controls.dampingFactor = 0.055;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.36;
  controls.minDistance = radius * 0.12;
  controls.maxDistance = radius * 6;

  state.arcs = new SynapseArcs(pack, centre);
  scene.add(state.cloud.points, state.arcs.lines);
  scene.add(backdrop(radius));
  applyScheme('class');

  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.7, 0.32, 0.30);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  raycaster = new THREE.Raycaster();
  raycaster.params.Points.threshold = Math.max(1.2, radius * 0.006);

  addEventListener('resize', resize);
  resize();
  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointerup', onPointerUp);

  renderer.setAnimationLoop(frame);
}

function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false);
  composer.setSize(w, h);
  bloom.resolution.set(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

/** A very dark gradient shell, so the scene reads as space rather than a flat fill. */
function backdrop(radius) {
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(radius * 9, 24, 16),
    new THREE.ShaderMaterial({
      side: THREE.BackSide,
      depthWrite: false,
      uniforms: { uTop: { value: new THREE.Color(0x0a1220) }, uBottom: { value: new THREE.Color(0x020306) } },
      vertexShader: `varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
      fragmentShader: `
        uniform vec3 uTop; uniform vec3 uBottom; varying vec3 vP;
        void main(){ gl_FragColor = vec4(mix(uBottom, uTop, smoothstep(-0.55, 0.85, normalize(vP).y)), 1.0); }`,
    }),
  );
  mesh.frustumCulled = false;
  return mesh;
}

const clock = new THREE.Clock();

function frame() {
  const dt = Math.min(clock.getDelta(), 0.1);
  const t = clock.elapsedTime;

  advanceReplay(dt);
  const live = state.cloud.update(dt, t);
  state.arcs.update(dt);
  controls.update();
  composer.render();

  $('stat-lit').textContent = live.toLocaleString();
}

/* ---- wavefront replay ---------------------------------------------------- */

function playActivation(activation) {
  state.cloud.clear();
  state.replay = { wavefronts: activation.wavefronts, hop: 0, clock: 0 };
  if ($('arcs').checked) state.arcs.show(activation.wavefronts);
  else state.arcs.clear();
}

function advanceReplay(dt) {
  const replay = state.replay;
  if (!replay) return;
  replay.clock += dt;
  while (replay.hop < replay.wavefronts.length &&
         replay.clock >= replay.hop * state.hopInterval) {
    const front = replay.wavefronts[replay.hop];
    // Later hops are dimmer, matching the decay the engine applies to the field.
    state.cloud.ignite(front.index, front.level, Math.pow(0.82, replay.hop));
    replay.hop++;
  }
  if (replay.hop >= replay.wavefronts.length) state.replay = null;
}

/* ---- picking ------------------------------------------------------------- */

let downAt = null;

function onPointerDown(e) { downAt = { x: e.clientX, y: e.clientY }; }

function onPointerUp(e) {
  if (!downAt) return;
  const moved = Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y);
  downAt = null;
  if (moved > 4) return;                       // a drag, not a click

  const ndc = new THREE.Vector2(
    (e.clientX / innerWidth) * 2 - 1,
    -(e.clientY / innerHeight) * 2 + 1,
  );
  raycaster.setFromCamera(ndc, camera);
  const hits = raycaster.intersectObject(state.cloud.points, false);
  if (!hits.length) return;
  // Points nearest the ray, not merely nearest the camera, is what a user means.
  hits.sort((a, b) => a.distanceToRay - b.distanceToRay);
  inspect(hits[0].index);
}

async function inspect(index) {
  const detail = await post('/api/neuron', { index });
  if (detail.error) return;

  $('ins-type').textContent = detail.cell_type || detail.cell_class || detail.super_class || 'neuron';
  $('ins-body').innerHTML = '';
  const rows = [
    ['class', detail.cell_class && detail.cell_class !== 'unknown' ? detail.cell_class : detail.super_class],
    ['transmitter', detail.neurotransmitter],
    ['side', detail.side],
    ['outputs', `${detail.out_degree} partners`],
    ['position', `${detail.position_um.map((v) => v.toFixed(0)).join(', ')} µm`],
    ['root id', detail.root_id],
  ];
  for (const [k, v] of rows) {
    if (!v) continue;
    const dt = document.createElement('dt'); dt.textContent = k;
    const dd = document.createElement('dd'); dd.textContent = v;
    $('ins-body').append(dt, dd);
  }
  $('ins-link').href = `https://codex.flywire.ai/app/cell_details?root_id=${detail.root_id}`;
  $('inspector').hidden = false;

  // Flash the neuron and the synapses leaving it.
  state.cloud.ignite([detail.index], [1.2]);
  if (detail.targets?.length && $('arcs').checked) {
    state.arcs.show([
      { index: [detail.index], level: [1] },
      { index: detail.targets.map((t) => t.index), level: detail.targets.map(() => 1) },
    ]);
  }
}

/* ---- console ------------------------------------------------------------- */

function setupConsole(pack) {
  document.querySelectorAll('.tabs button').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tabs button').forEach((b) =>
        b.setAttribute('aria-selected', String(b === btn)));
      document.querySelectorAll('.panel').forEach((p) =>
        { p.hidden = p.dataset.panel !== btn.dataset.tab; });
    });
  });

  $('go').addEventListener('click', recall);
  $('cue').addEventListener('keydown', (e) => { if (e.key === 'Enter') recall(); });
  $('write').addEventListener('click', write);
  $('inspector-close').addEventListener('click', () => { $('inspector').hidden = true; });

  $('colour').addEventListener('change', (e) => applyScheme(e.target.value));
  bindSlider('bloom', (v) => { bloom.strength = v; });
  bindSlider('size', (v) => state.cloud.setScale(v));
  bindSlider('sal', () => {});
  $('spin').addEventListener('change', (e) => { controls.autoRotate = e.target.checked; });
  $('arcs').addEventListener('change', (e) => { state.arcs.visible = e.target.checked; });

  // A first cue so the brain is doing something when the viewer opens.
  if (pack.manifest.memories) {
    $('cue').value = 'what does the user prefer?';
    recall();
  }
}

function bindSlider(id, apply) {
  const input = $(id), out = $(`${id}-out`);
  const run = () => {
    const v = parseFloat(input.value);
    if (out) out.textContent = v.toFixed(2);
    apply(v);
  };
  input.addEventListener('input', run);
  run();
}

function applyScheme(scheme) {
  const { colors, legend } = colorize(state.pack, scheme);
  state.cloud.setColors(colors);
  $('legend').innerHTML = '';
  for (const item of legend) {
    const span = document.createElement('span');
    const swatch = document.createElement('i');
    swatch.style.background = css(item.rgb);
    span.append(swatch, document.createTextNode(item.label));
    $('legend').append(span);
  }
}

async function recall() {
  const cue = $('cue').value.trim();
  if (!cue) return;
  const button = $('go');
  button.disabled = true;
  try {
    const data = await post('/api/activate', { cue, k: 6 });
    if (data.error) return;
    playActivation(data.activation);
    renderHits(data.hits);
  } finally {
    button.disabled = false;
  }
}

async function write() {
  const text = $('mem').value.trim();
  if (!text) return;
  const button = $('write');
  button.disabled = true;
  try {
    const data = await post('/api/experience', {
      text,
      context: $('ctx').value.trim(),
      provenance: $('prov').value.trim() || 'chat',
      salience: parseFloat($('sal').value),
      verified: $('ver').checked,
    });
    if (data.error) return;
    playActivation(data.activation);
    $('mem').value = '';
    $('stat-memories').textContent = Number($('stat-memories').textContent || 0) + 1;
  } finally {
    button.disabled = false;
  }
}

function renderHits(hits) {
  const list = $('hits');
  list.innerHTML = '';
  if (!hits.length) {
    const li = document.createElement('li');
    li.innerHTML = '<span class="score">—</span><span class="text">nothing recalled</span>';
    list.append(li);
    return;
  }
  for (const hit of hits) {
    const m = hit.memory;
    const li = document.createElement('li');

    const score = document.createElement('span');
    score.className = 'score';
    score.textContent = hit.score.toFixed(3);

    const text = document.createElement('span');
    text.className = 'text';
    text.textContent = m.text;

    const meta = document.createElement('span');
    meta.className = 'meta';
    meta.innerHTML =
      `<b>${m.tag_size} neurons</b> · overlap ${hit.connectome.toFixed(3)}` +
      (hit.fluctlight !== null ? ` · fluctlight ${hit.fluctlight.toFixed(3)}` : '') +
      (m.context ? ` · ${escape(m.context)}` : '') +
      (m.verified ? ` · <span class="verified">${escape(m.provenance)}</span>` : '');

    li.append(score, text, meta);
    list.append(li);
  }
}

function escape(s) {
  return String(s).replace(/[<>&]/g, (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;' })[c]);
}

async function post(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return res.json();
}
