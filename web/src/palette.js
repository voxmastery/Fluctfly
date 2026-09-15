/* Colour schemes for the neuron cloud.
 *
 * Every scheme returns a Float32Array of RGB triples, one per neuron. Hues are
 * chosen so that the activation colour — a hot near-white gold — never collides
 * with a resting colour, which is what lets a recall read as ignition rather
 * than as a change of category. */

const CELL_CLASS = {
  Kenyon_Cell: [0.24, 0.72, 0.86],   // the sparse coding layer: cool cyan
  ALPN:        [0.36, 0.84, 0.52],   // olfactory input: green
  MBON:        [0.92, 0.36, 0.68],   // readout: magenta
  DAN:         [1.00, 0.62, 0.22],   // reinforcement: amber
  MBIN:        [0.68, 0.48, 0.96],   // modulatory input: violet
};

const SUPER_CLASS = {
  central:            [0.34, 0.62, 0.92],
  optic:              [0.22, 0.46, 0.70],
  sensory:            [0.38, 0.86, 0.56],
  sensory_ascending:  [0.52, 0.88, 0.44],
  ascending:          [0.72, 0.84, 0.40],
  visual_projection:  [0.40, 0.78, 0.86],
  visual_centrifugal: [0.30, 0.66, 0.78],
  descending:         [0.94, 0.52, 0.34],
  motor:              [0.96, 0.34, 0.30],
  endocrine:          [0.88, 0.44, 0.84],
  unknown:            [0.42, 0.46, 0.54],
};

const NT = {
  acetylcholine: [0.36, 0.76, 0.94],  // excitatory
  glutamate:     [0.96, 0.54, 0.30],  // inhibitory in the fly, via GluCl
  gaba:          [0.92, 0.34, 0.42],  // inhibitory
  dopamine:      [1.00, 0.78, 0.30],
  serotonin:     [0.72, 0.52, 0.96],
  octopamine:    [0.42, 0.90, 0.70],
  unknown:       [0.40, 0.44, 0.52],
};

const SIDE = {
  left:   [0.34, 0.68, 0.94],
  right:  [0.96, 0.58, 0.32],
  center: [0.86, 0.86, 0.42],
  na:     [0.45, 0.48, 0.55],
};

const FALLBACK = [0.45, 0.50, 0.60];

/** Colour a neuron cloud; returns {colors, legend}. */
export function colorize(pack, scheme) {
  const n = pack.n;
  const colors = new Float32Array(n * 3);
  const used = new Map();

  const paint = (i, rgb) => {
    colors[i * 3] = rgb[0];
    colors[i * 3 + 1] = rgb[1];
    colors[i * 3 + 2] = rgb[2];
  };

  if (scheme === 'depth') {
    // Anterior to posterior along the y axis of the FAFB volume.
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < n; i++) {
      const y = pack.pos[i * 3 + 1];
      if (y < lo) lo = y;
      if (y > hi) hi = y;
    }
    const span = hi - lo || 1;
    for (let i = 0; i < n; i++) paint(i, ramp((pack.pos[i * 3 + 1] - lo) / span));
    return { colors, legend: [
      { label: 'anterior', rgb: ramp(0) },
      { label: 'mid', rgb: ramp(0.5) },
      { label: 'posterior', rgb: ramp(1) },
    ] };
  }

  const { table, vocab, index } = pick(pack, scheme);
  for (let i = 0; i < n; i++) {
    const name = vocab[index[i]] ?? 'unknown';
    const rgb = table[name] ?? FALLBACK;
    paint(i, rgb);
    if (!used.has(name)) used.set(name, rgb);
  }

  const legend = [...used.entries()]
    .map(([label, rgb]) => ({ label: label.replace(/_/g, ' '), rgb }))
    .sort((a, b) => a.label.localeCompare(b.label));
  return { colors, legend };
}

function pick(pack, scheme) {
  if (scheme === 'nt')   return { table: NT,   vocab: pack.ntOrder,   index: pack.nt };
  if (scheme === 'side') return { table: SIDE, vocab: pack.sideOrder, index: pack.side };
  // 'class': prefer the fine-grained cell class, but only when the palette
  // actually covers the pack. The whole-brain pack has fifty cell classes,
  // most of them optic-lobe types the mushroom-body palette knows nothing
  // about, and colouring 80% of the brain in the fallback grey is worse than
  // using the coarse division that is fully covered.
  const cellVocab = pack.vocab.cell_class || [];
  if (cellVocab.length) {
    let covered = 0;
    for (let i = 0; i < pack.n; i++) if ((cellVocab[pack.cellClass[i]] ?? '') in CELL_CLASS) covered++;
    if (covered > pack.n * 0.6) return { table: CELL_CLASS, vocab: cellVocab, index: pack.cellClass };
  }
  return { table: SUPER_CLASS, vocab: pack.vocab.super_class || [], index: pack.superClass };
}

/** Cool-to-warm ramp used for the depth scheme. */
function ramp(t) {
  const c = Math.min(1, Math.max(0, t));
  return [0.18 + 0.78 * c, 0.42 + 0.22 * Math.sin(Math.PI * c), 0.92 - 0.62 * c];
}

export function css(rgb) {
  const b = (v) => Math.round(Math.min(1, Math.max(0, v)) * 255);
  return `rgb(${b(rgb[0])},${b(rgb[1])},${b(rgb[2])})`;
}
