"""Binary container for a FlyWire connectome, readable by Python and the browser.

A *brain pack* is a directory of flat little-endian arrays plus a JSON manifest.
Flat arrays are used (rather than Parquet/HDF5) so the Three.js front end can
`fetch()` a buffer and hand it straight to a GPU attribute with no parsing.

Layout
------
    manifest.json    shapes, dtypes, category vocabularies, provenance
    pos.f32          [N*3]   soma//centroid position in micrometres, centred
    attr.u8          [N*4]   super_class, cell_class, neurotransmitter, side
    root_ids.u64     [N]     FlyWire root ids (v783)
    indptr.u32       [N+1]   CSR row pointer over outgoing synaptic edges
    indices.u32      [E]     CSR column indices (postsynaptic neuron)
    weight.f32       [E]     signed, input-normalised synaptic weight
    syn.u16          [E]     raw synapse count (clipped at 65535)
    types.json       cell_type / hemilineage strings, indexed by neuron

All edges are directed pre -> post, as in the FlyWire release.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_VERSION = 1

# Fly neurotransmitter -> sign used when building synaptic weights.
#
# Acetylcholine is the principal excitatory transmitter in the adult fly brain;
# GABA is inhibitory; glutamate is treated as inhibitory because the dominant
# postsynaptic receptor in Drosophila central brain is the glutamate-gated
# chloride channel GluCl. Aminergic transmitters are modulatory and are given a
# small positive weight rather than being dropped. This is the same sign
# convention used by published leaky-integrate-and-fire models of this
# connectome (Shiu et al., Nature 2024).
NT_SIGN: dict[str, float] = {
    "acetylcholine": 1.0,
    "glutamate": -1.0,
    "gaba": -1.0,
    "dopamine": 0.25,
    "serotonin": 0.25,
    "octopamine": 0.25,
    "unknown": 0.0,
}

NT_ORDER = [
    "acetylcholine",
    "glutamate",
    "gaba",
    "dopamine",
    "serotonin",
    "octopamine",
    "unknown",
]

SIDE_ORDER = ["left", "right", "center", "na"]

ARRAYS = {
    "pos": ("pos.f32", np.float32),
    "attr": ("attr.u8", np.uint8),
    "root_ids": ("root_ids.u64", np.uint64),
    "indptr": ("indptr.u32", np.uint32),
    "indices": ("indices.u32", np.uint32),
    "weight": ("weight.f32", np.float32),
    "syn": ("syn.u16", np.uint16),
}


def default_pack_dir(name: str = "mushroom_body") -> Path:
    """Where to look for a pack when none was named.

    An editable install or a clone has ``data/packs/`` beside the package; a
    wheel does not, so the working directory is checked too. Returns the first
    candidate that exists, else the repository-relative one so the error
    message points somewhere useful.
    """
    candidates = [
        Path(__file__).resolve().parent.parent / "data" / "packs" / name,
        Path.cwd() / "data" / "packs" / name,
    ]
    for candidate in candidates:
        if (candidate / "manifest.json").exists():
            return candidate
    return candidates[0]


@dataclass
class BrainPack:
    """An in-memory FlyWire connectome ready for activation spreading."""

    pos: np.ndarray           # (N, 3) float32, micrometres
    attr: np.ndarray          # (N, 4) uint8  -> super_class, cell_class, nt, side
    root_ids: np.ndarray      # (N,)  uint64
    indptr: np.ndarray        # (N+1,) uint32
    indices: np.ndarray       # (E,)  uint32
    weight: np.ndarray        # (E,)  float32, signed + input-normalised
    syn: np.ndarray           # (E,)  uint16
    manifest: dict[str, Any] = field(default_factory=dict)
    cell_types: list[str] = field(default_factory=list)

    # ---- derived -----------------------------------------------------------

    @property
    def n_neurons(self) -> int:
        return int(self.pos.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.indices.shape[0])

    @property
    def super_class(self) -> np.ndarray:
        return self.attr[:, 0]

    @property
    def cell_class(self) -> np.ndarray:
        return self.attr[:, 1]

    @property
    def nt(self) -> np.ndarray:
        return self.attr[:, 2]

    @property
    def side(self) -> np.ndarray:
        return self.attr[:, 3]

    def vocab(self, name: str) -> list[str]:
        return list(self.manifest.get("vocab", {}).get(name, []))

    def label(self, index: int) -> dict[str, Any]:
        """Human-readable description of one neuron, for the UI heads-up display."""
        sc = self.vocab("super_class")
        cc = self.vocab("cell_class")
        return {
            "index": int(index),
            "root_id": str(self.root_ids[index]),
            "super_class": sc[self.attr[index, 0]] if sc else "",
            "cell_class": cc[self.attr[index, 1]] if cc else "",
            "neurotransmitter": NT_ORDER[self.attr[index, 2]],
            "side": SIDE_ORDER[self.attr[index, 3]],
            "cell_type": self.cell_types[index] if self.cell_types else "",
            "position_um": [round(float(v), 2) for v in self.pos[index]],
            "out_degree": int(self.indptr[index + 1] - self.indptr[index]),
        }

    def out_edges(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """(postsynaptic indices, signed weights) for one neuron."""
        lo, hi = int(self.indptr[index]), int(self.indptr[index + 1])
        return self.indices[lo:hi], self.weight[lo:hi]

    # ---- io ----------------------------------------------------------------

    def save(self, path: str | os.PathLike[str]) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        for name, (fname, dtype) in ARRAYS.items():
            arr = np.ascontiguousarray(getattr(self, name), dtype=dtype)
            arr.tofile(out / fname)
        manifest = dict(self.manifest)
        manifest.update(
            format_version=FORMAT_VERSION,
            n_neurons=self.n_neurons,
            n_edges=self.n_edges,
            arrays={k: v[0] for k, v in ARRAYS.items()},
            nt_order=NT_ORDER,
            side_order=SIDE_ORDER,
        )
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (out / "types.json").write_text(json.dumps(self.cell_types))
        return out

    @classmethod
    def load(cls, path: str | os.PathLike[str], mmap: bool = True) -> "BrainPack":
        src = Path(path)
        manifest_file = src / "manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(
                f"no brain pack at {src} - build one with `make pack` "
                f"(or `python -m fluctfly.build --help`)"
            )
        manifest = json.loads(manifest_file.read_text())
        version = manifest.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"brain pack at {src} is format v{version}, this build reads "
                f"v{FORMAT_VERSION}; rebuild it with `make pack`"
            )
        n, e = manifest["n_neurons"], manifest["n_edges"]
        shapes = {
            "pos": (n, 3),
            "attr": (n, 4),
            "root_ids": (n,),
            "indptr": (n + 1,),
            "indices": (e,),
            "weight": (e,),
            "syn": (e,),
        }
        mode = "r" if mmap else None
        data = {}
        for name, (fname, dtype) in ARRAYS.items():
            f = src / fname
            if mode:
                data[name] = np.memmap(f, dtype=dtype, mode=mode, shape=shapes[name])
            else:
                data[name] = np.fromfile(f, dtype=dtype).reshape(shapes[name])

        types_file = src / "types.json"
        cell_types = json.loads(types_file.read_text()) if types_file.exists() else []
        return cls(manifest=manifest, cell_types=cell_types, **data)
