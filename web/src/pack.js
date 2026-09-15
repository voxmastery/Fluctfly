/* Loading the connectome into typed arrays.
 *
 * The server hands back raw little-endian buffers in exactly the layout the GPU
 * wants, so there is no parsing step: fetch, wrap in a typed array, done. The
 * CSR adjacency comes along too, because the viewer needs it to work out which
 * synapses carried a wavefront from one hop to the next. */

const ARRAYS = {
  pos:     Float32Array,
  attr:    Uint8Array,
  indptr:  Uint32Array,
  indices: Uint32Array,
  weight:  Float32Array,
  syn:     Uint16Array,
};

export async function loadPack(onProgress = () => {}) {
  const manifest = await (await fetch('/api/pack')).json();

  const names = Object.keys(ARRAYS);
  const data = {};
  let done = 0;
  await Promise.all(names.map(async (name) => {
    const buffer = await (await fetch(`/api/pack/${name}`)).arrayBuffer();
    data[name] = new ARRAYS[name](buffer);
    onProgress(++done / names.length);
  }));

  const n = manifest.n_neurons;
  return {
    manifest,
    n,
    edges: manifest.n_edges,
    pos: data.pos,
    indptr: data.indptr,
    indices: data.indices,
    weight: data.weight,
    syn: data.syn,
    // attr is interleaved [super_class, cell_class, nt, side] per neuron.
    superClass: strided(data.attr, n, 4, 0),
    cellClass:  strided(data.attr, n, 4, 1),
    nt:         strided(data.attr, n, 4, 2),
    side:       strided(data.attr, n, 4, 3),
    vocab: manifest.vocab || {},
    ntOrder: manifest.nt_order || [],
    sideOrder: manifest.side_order || [],
    /** Postsynaptic partners of one neuron, as a subarray view. */
    targets(i) { return this.indices.subarray(this.indptr[i], this.indptr[i + 1]); },
    weights(i) { return this.weight.subarray(this.indptr[i], this.indptr[i + 1]); },
  };
}

function strided(src, n, stride, offset) {
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = src[i * stride + offset];
  return out;
}

/** Brightness correction for a pack's size.
 *
 * Everything in the scene is additively blended, so the light a structure emits
 * grows with the number of neurons packed into it. The mushroom-body pack has
 * six thousand neurons spread thinly and needs lifting; the whole-brain pack
 * has a hundred and forty thousand, and the central brain clips to white
 * without a substantial cut. The exponent is fitted to those two ends. */
export function densityFactor(n) {
  return Math.max(0.15, Math.min(2.0, 1.8 * Math.pow(7000 / Math.max(1, n), 0.68)));
}

/** Radius of the smallest sphere containing every neuron, for framing the camera. */
export function extent(pos) {
  let max = 0;
  for (let i = 0; i < pos.length; i += 3) {
    const r = pos[i] * pos[i] + pos[i + 1] * pos[i + 1] + pos[i + 2] * pos[i + 2];
    if (r > max) max = r;
  }
  return Math.sqrt(max);
}
