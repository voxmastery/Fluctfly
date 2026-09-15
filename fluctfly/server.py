"""HTTP surface for the browser front end.

Three kinds of endpoint:

* ``/api/pack/*``  - the connectome itself, served as raw little-endian buffers
  that go straight into GPU attributes with no parsing on the client.
* ``/api/experience`` and ``/api/activate`` - the memory verbs, which return the
  full hop-by-hop wavefront so the front end can replay the propagation instead
  of only showing its result.
* ``/`` - the static viewer.

Deliberately dependency-light: the standard library's HTTP server is enough for
a single-user instrument, and it means `make run` needs nothing but numpy.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from .engine import Fluctfly
from .pack import BrainPack, default_pack_dir

WEB_ROOT = Path(__file__).resolve().parent.parent / "web"
ARRAY_ROUTE = re.compile(r"^/api/pack/(pos|attr|indptr|indices|weight|syn)$")


class Handler(BaseHTTPRequestHandler):
    server_version = "fluctfly"

    def __init__(self, *args, store: Fluctfly, **kwargs):
        self.store = store
        super().__init__(*args, **kwargs)

    # -- plumbing ------------------------------------------------------------

    def log_message(self, fmt: str, *args) -> None:  # quieter than the default
        if self.path.startswith("/api/") and not self.path.startswith("/api/pack/"):
            super().log_message(fmt, *args)

    def _send(self, code: int, body: bytes, content_type: str, cache: bool = False) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            return {}

    # -- routes --------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/pack":
            return self._json(self.pack_manifest())
        if match := ARRAY_ROUTE.match(path):
            array = np.ascontiguousarray(getattr(self.store.pack, match.group(1)))
            return self._send(200, array.tobytes(), "application/octet-stream", cache=True)
        if path == "/api/memories":
            return self._json({"memories": [m.as_json() for m in self.store.memories.values()]})
        return self.static(path)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        body = self._body()
        if path == "/api/experience":
            text = (body.get("text") or "").strip()
            if not text:
                return self._json({"error": "text is required"}, 400)
            memory, activation = self.store.experience(
                text,
                context=body.get("context", ""),
                salience=float(body.get("salience", 0.5)),
                verified=bool(body.get("verified", False)),
                provenance=body.get("provenance", "chat"),
            )
            return self._json(
                {"memory": memory.as_json(), "activation": self.activation_json(activation)}
            )
        if path == "/api/activate":
            cue = (body.get("cue") or "").strip()
            if not cue:
                return self._json({"error": "cue is required"}, 400)
            hits, activation = self.store.activate(cue, k=int(body.get("k", 5)))
            return self._json(
                {
                    "cue": cue,
                    "hits": [h.as_json() for h in hits],
                    "activation": self.activation_json(activation),
                }
            )
        if path == "/api/checkpoint":
            target = self.store.checkpoint(body.get("path"))
            return self._json({"checkpointed": str(target), "count": len(self.store.memories)})
        if path == "/api/neuron":
            index = int(body.get("index", -1))
            if not 0 <= index < self.store.pack.n_neurons:
                return self._json({"error": "index out of range"}, 400)
            return self._json(self.neuron_detail(index))
        return self._json({"error": "not found"}, 404)

    # -- payload builders ----------------------------------------------------

    def pack_manifest(self) -> dict:
        pack = self.store.pack
        manifest = dict(pack.manifest)
        manifest["memories"] = len(self.store.memories)
        manifest["fused_with_fluctlightdb"] = self.store.fused
        manifest["cue_encoding"] = self.store.encoder.mode
        manifest["input_population"] = int(self.store.encoder.inputs.size)
        return manifest

    @staticmethod
    def activation_json(activation) -> dict:
        idx, val = activation.sparse(4096)
        return {
            "wavefronts": [w.as_json() for w in activation.wavefronts],
            "field": {
                "index": idx.astype(int).tolist(),
                "level": [round(float(v), 4) for v in val],
            },
            "lit": int((activation.field > 0).sum()),
        }

    def neuron_detail(self, index: int) -> dict:
        pack = self.store.pack
        detail = pack.label(index)
        targets, weights = pack.out_edges(index)
        if targets.size:
            order = np.argsort(np.abs(weights))[::-1][:24]
            detail["targets"] = [
                {
                    "index": int(targets[i]),
                    "weight": round(float(weights[i]), 4),
                    "position_um": [round(float(v), 2) for v in pack.pos[targets[i]]],
                }
                for i in order
            ]
        else:
            detail["targets"] = []
        return detail

    # -- static files --------------------------------------------------------

    def static(self, path: str) -> None:
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        target = (WEB_ROOT / rel).resolve()
        if not str(target).startswith(str(WEB_ROOT.resolve())) or not target.is_file():
            return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self._send(200, target.read_bytes(), ctype)


def serve(store: Fluctfly, host: str = "127.0.0.1", port: int = 8799) -> None:
    handler = partial(Handler, store=store)
    httpd = ThreadingHTTPServer((host, port), handler)
    pack = store.pack
    print(f"Fluctfly  {pack.n_neurons:,} neurons  {pack.n_edges:,} synaptic connections")
    print(f"          pack '{pack.manifest.get('subset')}' from {pack.manifest.get('dataset')}")
    print(f"          cue encoding: {store.encoder.mode}")
    print(f"          http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Serve the Fluctfly viewer.")
    ap.add_argument("--pack", type=Path, default=default_pack_dir())
    ap.add_argument("--store", type=Path, default=None, help="directory to persist memories in")
    ap.add_argument("--fluctlight", type=Path, default=None, help="FluctlightDB brain to fuse with")
    ap.add_argument("--embed", action="store_true", help="use sentence-transformers for cues")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--seed", action="store_true", help="load the demo memories on start")
    args = ap.parse_args(argv)

    embed = None
    if args.embed:
        from .encode import sentence_encoder

        embed = sentence_encoder()

    store = Fluctfly(
        BrainPack.load(args.pack),
        path=args.store,
        fluctlight_path=args.fluctlight,
        embed=embed,
    )
    if args.seed and not store.memories:
        from .demo import seed

        seed(store)
    serve(store, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
