"""Download the public FlyWire sources that a brain pack is built from.

Nothing here is redistributed with this repository: the files are fetched from
their canonical homes (GitHub and Zenodo) into ``data/raw/`` on first use.

Sources
-------
annotations
    Supplemental file 1 of Schlegel et al., *Nature* 2024 - one row per
    proofread neuron with position, side, super class, cell type and predicted
    neurotransmitter.  ~31 MB.
connections
    ``proofread_connections_783.feather`` from the FlyWire connectivity record
    on Zenodo - one row per (pre, post, neuropil) with a synapse count.
    ~852 MB, and the reason `make pack` is a separate step from `make run`.
"""

from __future__ import annotations

import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
RAW = DATA_ROOT / "raw"

SOURCES = {
    "annotations": {
        "url": (
            "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
            "main/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
        ),
        "filename": "neuron_annotations_783.tsv",
        "approx_mb": 31,
        "cite": "Schlegel et al., Nature 634, 139-152 (2024)",
    },
    "connections": {
        "url": (
            "https://zenodo.org/api/records/10676866/files/"
            "proofread_connections_783.feather/content"
        ),
        "filename": "proofread_connections_783.feather",
        "approx_mb": 852,
        "cite": "Dorkenwald et al., Nature 634, 124-138 (2024); Zenodo 10676866",
    },
}


def _progress(done: int, total: int, label: str) -> None:
    if not sys.stderr.isatty():
        return
    if total > 0:
        pct = 100.0 * done / total
        bar = "#" * int(pct / 2.5)
        sys.stderr.write(f"\r  {label:14s} [{bar:<40}] {pct:5.1f}%  {done/1e6:7.1f} MB")
    else:
        sys.stderr.write(f"\r  {label:14s} {done/1e6:7.1f} MB")
    sys.stderr.flush()


def fetch(name: str, force: bool = False) -> Path:
    """Download one source if it is not already present; return its path."""
    if name not in SOURCES:
        raise KeyError(f"unknown source {name!r}; expected one of {sorted(SOURCES)}")
    spec = SOURCES[name]
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / spec["filename"]
    if dest.exists() and not force:
        print(f"  {name:14s} already present ({dest.stat().st_size/1e6:.1f} MB)")
        return dest

    print(f"  {name:14s} downloading ~{spec['approx_mb']} MB from {spec['cite']}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(spec["url"], headers={"User-Agent": "fluctfly/0.1"})
        with urllib.request.urlopen(req) as resp, tmp.open("wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                _progress(done, total, name)
        if sys.stderr.isatty():
            sys.stderr.write("\n")
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"could not download {name} from {spec['url']}: {exc}") from exc
    shutil.move(tmp, dest)
    return dest


def fetch_all(force: bool = False) -> dict[str, Path]:
    print("Fetching FlyWire v783 sources into", RAW)
    return {name: fetch(name, force=force) for name in SOURCES}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", nargs="?", choices=sorted(SOURCES), help="one source; default all")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()
    if args.source:
        fetch(args.source, force=args.force)
    else:
        fetch_all(force=args.force)
