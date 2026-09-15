"""Activation spreading across a real Drosophila connectome.

One `activate()` in FluctlightDB fans a cue out through a graph.  Here that
graph is not an index built from the stored text - it is the wiring of an actual
fly brain, with the real synapse counts as weights and the predicted
neurotransmitter of each presynaptic neuron deciding whether a connection drives
or suppresses its target.

The dynamics are deliberately simple and are not a claim to simulate a fly:

    drive_{t+1} = W^T . a_t                 signed, input-normalised
    a_{t+1}     = sparsify(relu(drive) - inhibition)

`sparsify` keeps the strongest few hundred neurons, which is both what the
mushroom body does to an odour (a few percent of Kenyon cells fire) and what
keeps a hop cheap enough to animate at sixty frames a second.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pack import BrainPack


@dataclass
class Wavefront:
    """One hop of spreading, kept so the front end can replay the propagation."""

    hop: int
    index: np.ndarray        # neuron indices that fired
    level: np.ndarray        # their activation, descending

    def as_json(self, limit: int = 600) -> dict:
        n = min(limit, self.index.size)
        return {
            "hop": self.hop,
            "index": self.index[:n].astype(int).tolist(),
            "level": [round(float(v), 5) for v in self.level[:n]],
        }


@dataclass
class Activation:
    """The result of spreading a cue through the connectome."""

    field: np.ndarray            # (N,) accumulated activation per neuron
    wavefronts: list[Wavefront]
    ignition: np.ndarray         # indices the cue entered through

    def top(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, int((self.field > 0).sum()))
        if k == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        idx = np.argpartition(self.field, -k)[-k:]
        idx = idx[np.argsort(self.field[idx])[::-1]]
        return idx, self.field[idx]

    def tag(self, k: int = 64) -> np.ndarray:
        """Sparse identity of this cue: the k most-activated neurons, sorted."""
        idx, _ = self.top(k)
        return np.sort(idx).astype(np.uint32)

    def sparse(self, k: int = 128) -> tuple[np.ndarray, np.ndarray]:
        """The engram as a sparse vector: neuron indices and their activation.

        Keeping the levels rather than a plain set of indices matters for
        ranking - two cues often light the same hub neurons weakly, and only
        the magnitudes separate a real match from that shared background.
        """
        idx, val = self.top(k)
        order = np.argsort(idx)
        return idx[order].astype(np.uint32), val[order].astype(np.float32)


class Spreader:
    """Propagates activation over a pack's CSR adjacency."""

    def __init__(
        self,
        pack: BrainPack,
        hops: int = 4,
        width: int = 1024,
        decay: float = 0.65,
        inhibition: float = 0.25,
    ):
        self.pack = pack
        self.hops = hops
        self.width = width          # neurons kept per hop
        self.decay = decay          # contribution of later hops to the field
        self.inhibition = inhibition  # strength of the winner-take-all threshold
        # Materialise CSR views once; memmapped arrays are fine to index into.
        self._indptr = np.asarray(pack.indptr, dtype=np.int64)
        self._indices = np.asarray(pack.indices, dtype=np.int64)
        self._weight = np.asarray(pack.weight, dtype=np.float32)

    # -- one hop -------------------------------------------------------------

    def _drive(self, active: np.ndarray, level: np.ndarray) -> np.ndarray:
        """Scatter activation from `active` along their outgoing synapses."""
        lo = self._indptr[active]
        hi = self._indptr[active + 1]
        count = hi - lo
        total = int(count.sum())
        if total == 0:
            return np.zeros(self.pack.n_neurons, dtype=np.float64)
        # Expand each active neuron's [lo, hi) range into a flat edge list
        # without a Python loop: repeat the starts, then subtract the running
        # offset so each run counts 0, 1, 2, ...
        starts = np.repeat(lo, count)
        run_base = np.repeat(np.cumsum(count) - count, count)
        edges = starts + (np.arange(total, dtype=np.int64) - run_base)
        contrib = self._weight[edges] * np.repeat(level, count)
        return np.bincount(self._indices[edges], weights=contrib, minlength=self.pack.n_neurons)

    def _sparsify(self, drive: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Keep the strongest excited neurons; everything else is silenced."""
        excited = np.flatnonzero(drive > 0)
        if excited.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        values = drive[excited]
        if excited.size > self.width:
            keep = np.argpartition(values, -self.width)[-self.width :]
            excited, values = excited[keep], values[keep]
        # Global inhibition, as the APL neuron provides for Kenyon cells: hold
        # a threshold proportional to the strongest response, so a cue that
        # drives the brain hard does not simply recruit everything.
        peak = values.max()
        values = values - self.inhibition * peak
        alive = values > 0
        excited, values = excited[alive], values[alive]
        if values.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        values = values / values.max()
        order = np.argsort(values)[::-1]
        return excited[order], values[order].astype(np.float32)

    # -- full spread ---------------------------------------------------------

    def spread(self, seed: np.ndarray, hops: int | None = None) -> Activation:
        """Propagate a seed activation vector over the connectome.

        `seed` is a dense (N,) array; only its non-zero entries matter.
        """
        hops = self.hops if hops is None else hops
        field = np.zeros(self.pack.n_neurons, dtype=np.float64)

        active = np.flatnonzero(seed > 0)
        if active.size > self.width:
            vals = seed[active]
            keep = np.argpartition(vals, -self.width)[-self.width :]
            active = active[keep]
        level = seed[active].astype(np.float32)
        if level.size and level.max() > 0:
            level = level / level.max()

        ignition = active.copy()
        fronts = [Wavefront(0, active, level)]
        field[active] += level

        for hop in range(1, hops + 1):
            if active.size == 0:
                break
            drive = self._drive(active, level)
            active, level = self._sparsify(drive)
            if active.size == 0:
                break
            field[active] += (self.decay ** hop) * level
            fronts.append(Wavefront(hop, active, level))

        peak = field.max()
        if peak > 0:
            field /= peak
        return Activation(field=field, wavefronts=fronts, ignition=ignition)


def overlap(tag_a: np.ndarray, tag_b: np.ndarray) -> float:
    """Jaccard overlap between two sparse engram tags."""
    if tag_a.size == 0 or tag_b.size == 0:
        return 0.0
    shared = np.intersect1d(tag_a, tag_b, assume_unique=True).size
    union = tag_a.size + tag_b.size - shared
    return float(shared) / float(union) if union else 0.0


def cosine(
    idx_a: np.ndarray, val_a: np.ndarray, idx_b: np.ndarray, val_b: np.ndarray
) -> float:
    """Cosine similarity between two sparse engrams, both index-sorted."""
    if idx_a.size == 0 or idx_b.size == 0:
        return 0.0
    shared, pos_a, pos_b = np.intersect1d(idx_a, idx_b, assume_unique=True, return_indices=True)
    if shared.size == 0:
        return 0.0
    dot = float(np.dot(val_a[pos_a], val_b[pos_b]))
    norm = float(np.linalg.norm(val_a) * np.linalg.norm(val_b))
    return dot / norm if norm > 0 else 0.0
