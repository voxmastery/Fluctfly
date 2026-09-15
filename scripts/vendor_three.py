"""Download three.js and the addons the viewer needs into web/vendor/.

Vendoring rather than hot-linking a CDN keeps `make run` working with no
network and pins exactly one version of three.  Relative imports are followed
recursively so the addon graph arrives complete.
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path
from posixpath import normpath

VERSION = "0.169.0"
BASE = f"https://cdn.jsdelivr.net/npm/three@{VERSION}"
OUT = Path(__file__).resolve().parent.parent / "web" / "vendor" / "three"

ENTRIES = [
    "build/three.module.js",
    "examples/jsm/controls/OrbitControls.js",
    "examples/jsm/postprocessing/EffectComposer.js",
    "examples/jsm/postprocessing/RenderPass.js",
    "examples/jsm/postprocessing/UnrealBloomPass.js",
    "examples/jsm/postprocessing/OutputPass.js",
]

IMPORT = re.compile(r"""\bfrom\s+['"](\.[^'"]+)['"]""")


def fetch(rel: str, seen: set[str]) -> None:
    if rel in seen:
        return
    seen.add(rel)
    url = f"{BASE}/{rel}"
    dest = OUT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {rel}")
    with urllib.request.urlopen(url) as resp:
        source = resp.read().decode()
    dest.write_text(source)

    here = str(Path(rel).parent)
    for spec in IMPORT.findall(source):
        # 'three' itself is remapped by the import map, only follow relatives
        target = normpath(f"{here}/{spec}")
        if target.startswith(".."):
            continue
        fetch(target, seen)


def main() -> int:
    print(f"vendoring three@{VERSION} into {OUT}")
    seen: set[str] = set()
    for entry in ENTRIES:
        fetch(entry, seen)
    total = sum(f.stat().st_size for f in OUT.rglob("*.js"))
    print(f"{len(seen)} files, {total/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
