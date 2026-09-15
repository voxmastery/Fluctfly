"""Turn the raw FlyWire release into a brain pack.

    python -m fluctfly.build --subset core --out data/packs/core

The expensive part is the connection table: the v783 release lists every
(pre, post, neuropil) triple with at least one detected synapse, which is tens
of millions of rows.  We read only the three columns we need, aggregate across
neuropils, threshold, and emit a CSR adjacency.

Thresholding matters.  Single-synapse connections in an EM reconstruction are
dominated by false positives; the FlyWire papers analyse connections of five or
more synapses, and that is the default here.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from .pack import NT_ORDER, NT_SIGN, SIDE_ORDER, BrainPack
from .fetch import RAW, fetch

# FAFB v14.1 is stored in voxels; these are the voxel dimensions in nanometres.
VOXEL_NM = np.array([4.0, 4.0, 40.0], dtype=np.float64)

# A subset is a predicate over the annotation row.  ``super_class`` is the
# coarse division (optic / central / sensory / ...); ``cell_class`` is the
# finer one that names actual circuits.
SUBSETS: dict[str, dict[str, set[str]] | None] = {
    # The whole brain: 139,248 neurons, ~2.7M connections.
    "all": None,
    # Everything that is not optic-lobe intrinsic.  The optic lobes are 77k of
    # the 139k neurons and are highly repetitive columnar circuitry; dropping
    # them leaves the central brain plus all sensory input and motor output,
    # which is the interesting substrate for associative recall and small
    # enough to ship.
    "core": {
        "super_class": {
            "central",
            "sensory",
            "visual_projection",
            "ascending",
            "descending",
            "sensory_ascending",
            "visual_centrifugal",
            "motor",
            "endocrine",
        }
    },
    # The mushroom body: the fly's associative learning and memory centre, and
    # the obvious place to put an agent's memories.  Olfactory projection
    # neurons carry odour identity to ~5,200 Kenyon cells, whose sparse
    # combinatorial code is read out by ~100 output neurons; dopaminergic
    # neurons signal reinforcement and gate which synapses change.  Roughly
    # 6,300 neurons - small enough to ship in the repository.
    "mushroom_body": {
        "cell_class": {"Kenyon_Cell", "MBON", "DAN", "MBIN", "ALPN"}
    },
}


def _matches(row: dict, subset: dict[str, set[str]] | None) -> bool:
    if subset is None:
        return True
    return all((row.get(field) or "unknown") in allowed for field, allowed in subset.items())


def _log(msg: str) -> None:
    print(f"[build] {msg}", file=sys.stderr, flush=True)


def _read_annotations(path: Path, subset: dict[str, set[str]] | None) -> dict:
    """Load the per-neuron annotation table, optionally restricted to a subset."""
    root_ids: list[int] = []
    pos: list[tuple[float, float, float]] = []
    super_class: list[str] = []
    cell_class: list[str] = []
    nt: list[str] = []
    side: list[str] = []
    cell_types: list[str] = []

    with path.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if not _matches(row, subset):
                continue
            sc = row["super_class"] or "unknown"
            # Prefer the soma when the nucleus was identified, otherwise the
            # annotator's representative point inside the neuron.
            try:
                p = (float(row["soma_x"]), float(row["soma_y"]), float(row["soma_z"]))
            except (ValueError, KeyError):
                p = (float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"]))
            root_ids.append(int(row["root_id"]))
            pos.append(p)
            super_class.append(sc)
            cell_class.append(row["cell_class"] or "unknown")
            nt.append((row["top_nt"] or "unknown").lower())
            side.append((row["side"] or "na").lower())
            cell_types.append(row["cell_type"] or row["hemibrain_type"] or "")

    n = len(root_ids)
    if n == 0:
        raise ValueError("annotation table produced zero neurons - wrong subset?")
    _log(f"annotations: {n:,} neurons")

    # Sort by root id so edge lookup can use searchsorted.
    order = np.argsort(np.asarray(root_ids, dtype=np.uint64), kind="stable")
    take = lambda seq: [seq[i] for i in order]  # noqa: E731

    xyz_nm = np.asarray(take(pos), dtype=np.float64) * VOXEL_NM
    xyz_um = (xyz_nm / 1000.0).astype(np.float32)
    xyz_um -= xyz_um.mean(axis=0)  # centre on the brain so the camera orbits it

    sc_vocab = sorted(set(super_class))
    cc_vocab = sorted(set(cell_class))
    sc_index = {v: i for i, v in enumerate(sc_vocab)}
    cc_index = {v: i for i, v in enumerate(cc_vocab)}
    nt_index = {v: i for i, v in enumerate(NT_ORDER)}
    side_index = {v: i for i, v in enumerate(SIDE_ORDER)}

    attr = np.zeros((n, 4), dtype=np.uint8)
    attr[:, 0] = [sc_index[v] for v in take(super_class)]
    attr[:, 1] = [cc_index[v] for v in take(cell_class)]
    attr[:, 2] = [nt_index.get(v, nt_index["unknown"]) for v in take(nt)]
    attr[:, 3] = [side_index.get(v, side_index["na"]) for v in take(side)]

    return {
        "root_ids": np.asarray(take(root_ids), dtype=np.uint64),
        "pos": xyz_um,
        "attr": attr,
        "cell_types": take(cell_types),
        "vocab": {"super_class": sc_vocab, "cell_class": cc_vocab},
    }


def _connection_columns(path: Path) -> tuple[str, str, str]:
    """Find the pre / post / synapse-count columns, whatever the release names them."""
    import pyarrow as pa

    # Feather v2 is the Arrow IPC file format, so the schema can be read from
    # the footer without touching the several hundred megabytes of column data.
    with pa.memory_map(str(path), "r") as source:
        names = pa.ipc.open_file(source).schema.names
    lowered = {c.lower(): c for c in names}

    def pick(*candidates: str) -> str:
        for c in candidates:
            if c in lowered:
                return lowered[c]
        raise KeyError(f"none of {candidates} in connection table columns {names}")

    return (
        pick("pre_root_id", "pre_pt_root_id", "pre"),
        pick("post_root_id", "post_pt_root_id", "post"),
        pick("syn_count", "synapse_count", "count", "n_syn"),
    )


def _read_edges(path: Path, root_ids: np.ndarray, min_syn: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate the connection table into unique (pre, post, syn_count) edges."""
    import pyarrow.feather as feather

    pre_col, post_col, syn_col = _connection_columns(path)
    _log(f"connections: reading columns {pre_col}, {post_col}, {syn_col}")
    t0 = time.time()
    table = feather.read_table(path, columns=[pre_col, post_col, syn_col], memory_map=True)
    pre_ids = table.column(pre_col).to_numpy().astype(np.uint64, copy=False)
    post_ids = table.column(post_col).to_numpy().astype(np.uint64, copy=False)
    syn = table.column(syn_col).to_numpy().astype(np.int64, copy=False)
    del table
    _log(f"connections: {len(syn):,} raw rows in {time.time()-t0:.1f}s")

    n = len(root_ids)
    # root_ids is sorted, so searchsorted maps ids -> indices; anything that
    # lands outside the table (e.g. an optic neuron in the `core` subset) is
    # dropped along with its edges.
    pre = np.searchsorted(root_ids, pre_ids).astype(np.int64)
    np.clip(pre, 0, n - 1, out=pre)
    keep = root_ids[pre] == pre_ids
    del pre_ids

    post = np.searchsorted(root_ids, post_ids).astype(np.int64)
    np.clip(post, 0, n - 1, out=post)
    keep &= root_ids[post] == post_ids
    del post_ids

    keep &= pre != post  # autapses are reconstruction artefacts here
    pre, post, syn = pre[keep], post[keep], syn[keep]
    del keep
    _log(f"connections: {len(syn):,} rows inside the selected neuron set")

    # Collapse the per-neuropil rows into one edge per ordered pair.
    key = pre * np.int64(n) + post
    del pre, post
    order = np.argsort(key, kind="stable")
    key = key[order]
    syn = syn[order]
    del order
    boundary = np.empty(len(key), dtype=bool)
    boundary[0] = True
    np.not_equal(key[1:], key[:-1], out=boundary[1:])
    starts = np.flatnonzero(boundary)
    totals = np.add.reduceat(syn, starts)
    uniq = key[starts]
    del key, syn, boundary, starts

    strong = totals >= min_syn
    uniq, totals = uniq[strong], totals[strong]
    _log(f"connections: {len(totals):,} edges with >={min_syn} synapses")

    return (uniq // n).astype(np.uint32), (uniq % n).astype(np.uint32), totals


def build(
    annotations: Path,
    connections: Path,
    out: Path,
    subset: str = "core",
    min_syn: int = 5,
) -> BrainPack:
    meta = _read_annotations(annotations, SUBSETS[subset])
    root_ids = meta["root_ids"]
    n = len(root_ids)

    pre, post, syn = _read_edges(connections, root_ids, min_syn)
    e = len(syn)

    # CSR over outgoing edges: rows are presynaptic neurons.
    order = np.argsort(pre, kind="stable")
    pre, post, syn = pre[order], post[order], syn[order]
    indptr = np.zeros(n + 1, dtype=np.uint32)
    counts = np.bincount(pre, minlength=n)
    indptr[1:] = np.cumsum(counts)

    # Signed, input-normalised weights.  Dividing by the postsynaptic neuron's
    # total input turns raw synapse counts into "fraction of this neuron's
    # drive", which keeps hub neurons from swamping the spread.
    sign = np.array([NT_SIGN[t] for t in NT_ORDER], dtype=np.float32)[meta["attr"][:, 2]]
    in_strength = np.bincount(post, weights=syn.astype(np.float64), minlength=n)
    in_strength[in_strength == 0] = 1.0
    weight = (sign[pre] * syn / in_strength[post]).astype(np.float32)

    pack = BrainPack(
        pos=meta["pos"],
        attr=meta["attr"],
        root_ids=root_ids,
        indptr=indptr,
        indices=post.astype(np.uint32),
        weight=weight,
        syn=np.clip(syn, 0, 65535).astype(np.uint16),
        cell_types=meta["cell_types"],
        manifest={
            "dataset": "FlyWire FAFB v783 (adult female Drosophila melanogaster)",
            "subset": subset,
            "min_synapses": min_syn,
            "vocab": meta["vocab"],
            "bounds_um": {
                "min": [round(float(v), 2) for v in meta["pos"].min(axis=0)],
                "max": [round(float(v), 2) for v in meta["pos"].max(axis=0)],
            },
            "sources": [
                "Dorkenwald et al., Nature 634, 124-138 (2024) - connectome",
                "Schlegel et al., Nature 634, 139-152 (2024) - annotations",
                "Eckstein, Bates et al., Cell 187, 2574-2594 (2024) - neurotransmitters",
            ],
            "license": "FlyWire data released under CC BY 4.0",
        },
    )
    _log(f"pack: {n:,} neurons, {e:,} edges -> {out}")
    pack.save(out)
    return pack


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", choices=sorted(SUBSETS), default="core")
    ap.add_argument("--min-syn", type=int, default=5, help="minimum synapses per connection (default 5)")
    ap.add_argument("--out", type=Path, default=None, help="output pack directory")
    ap.add_argument("--annotations", type=Path, default=None)
    ap.add_argument("--connections", type=Path, default=None)
    ap.add_argument("--no-fetch", action="store_true", help="fail instead of downloading missing sources")
    args = ap.parse_args(argv)

    def resolve(explicit: Path | None, source: str) -> Path:
        if explicit:
            return explicit
        path = RAW / __import__("fluctfly.fetch", fromlist=["SOURCES"]).SOURCES[source]["filename"]
        if path.exists():
            return path
        if args.no_fetch:
            raise SystemExit(f"missing {path}; run `python -m fluctfly.fetch {source}`")
        return fetch(source)

    out = args.out or (Path(__file__).resolve().parent.parent / "data" / "packs" / args.subset)
    build(
        annotations=resolve(args.annotations, "annotations"),
        connections=resolve(args.connections, "connections"),
        out=out,
        subset=args.subset,
        min_syn=args.min_syn,
    )
    print(f"brain pack written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
