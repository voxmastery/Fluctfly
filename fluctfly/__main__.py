"""`fluctfly` command line: fetch, build, serve, or query from a terminal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pack import default_pack_dir

DEFAULT_PACK = default_pack_dir()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fluctfly", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", help="download the FlyWire sources")

    build = sub.add_parser("build", help="build a brain pack from the sources")
    build.add_argument("--subset", default="core")
    build.add_argument("--min-syn", type=int, default=5)
    build.add_argument("--out", type=Path, default=None)

    serve = sub.add_parser("serve", help="run the viewer")
    serve.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    serve.add_argument("--store", type=Path, default=None)
    serve.add_argument("--fluctlight", type=Path, default=None)
    serve.add_argument("--embed", action="store_true")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8799)
    serve.add_argument("--seed", action="store_true")

    recall = sub.add_parser("recall", help="recall from a cue without the browser")
    recall.add_argument("cue")
    recall.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    recall.add_argument("--store", type=Path, default=None)
    recall.add_argument("-k", type=int, default=5)
    recall.add_argument("--seed", action="store_true", help="use the demo memories")

    args = ap.parse_args(argv)

    if args.command == "fetch":
        from .fetch import fetch_all

        fetch_all()
        return 0

    if args.command == "build":
        from .build import main as build_main

        argv2 = ["--subset", args.subset, "--min-syn", str(args.min_syn)]
        if args.out:
            argv2 += ["--out", str(args.out)]
        return build_main(argv2)

    if args.command == "serve":
        from .server import main as serve_main

        argv2 = ["--pack", str(args.pack), "--host", args.host, "--port", str(args.port)]
        if args.store:
            argv2 += ["--store", str(args.store)]
        if args.fluctlight:
            argv2 += ["--fluctlight", str(args.fluctlight)]
        if args.embed:
            argv2.append("--embed")
        if args.seed:
            argv2.append("--seed")
        return serve_main(argv2)

    if args.command == "recall":
        from .engine import Fluctfly
        from .pack import BrainPack

        store = Fluctfly(BrainPack.load(args.pack), path=args.store)
        if args.seed and not store.memories:
            from .demo import seed

            seed(store)
        if not store.memories:
            print("no memories; pass --seed or --store", file=sys.stderr)
            return 1
        hits, activation = store.activate(args.cue, k=args.k)
        lit = int((activation.field > 0).sum())
        print(f"{lit} neurons lit over {len(activation.wavefronts)} hops\n")
        for hit in hits:
            flag = "verified" if hit.memory.verified else "chat"
            print(f"  {hit.score:.3f}  {hit.memory.text}")
            print(f"         overlap {hit.connectome:.3f} · {hit.memory.context or '—'} · {flag}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
