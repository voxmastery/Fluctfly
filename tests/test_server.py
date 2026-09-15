"""The HTTP surface the viewer talks to."""

import json
import threading
import urllib.error
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from fluctfly.server import Handler


@pytest.fixture(scope="module")
def live(request):
    from fluctfly.demo import seed
    from fluctfly.engine import Fluctfly
    from fluctfly.pack import BrainPack

    from conftest import PACK_DIR

    if not (PACK_DIR / "manifest.json").exists():
        pytest.skip("no brain pack; run `make pack`")
    store = Fluctfly(BrainPack.load(PACK_DIR))
    seed(store)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, store=store))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", store
    httpd.shutdown()


def get(base, path):
    with urllib.request.urlopen(f"{base}{path}") as r:
        return r.read(), r.headers


def post(base, path, payload):
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return json.loads(exc.read())


def test_manifest_describes_the_pack(live):
    base, store = live
    body, _ = get(base, "/api/pack")
    manifest = json.loads(body)
    assert manifest["n_neurons"] == store.pack.n_neurons
    assert manifest["n_edges"] == store.pack.n_edges
    assert manifest["cue_encoding"] in {"lexical", "embedding"}
    assert "FlyWire" in manifest["dataset"]


@pytest.mark.parametrize(
    "name,itemsize,columns",
    [("pos", 4, 3), ("attr", 1, 4), ("indices", 4, 1), ("weight", 4, 1), ("syn", 2, 1)],
)
def test_arrays_are_raw_buffers_of_the_right_length(live, name, itemsize, columns):
    base, store = live
    body, headers = get(base, f"/api/pack/{name}")
    rows = store.pack.n_neurons if name in {"pos", "attr"} else store.pack.n_edges
    assert len(body) == rows * columns * itemsize
    assert headers["Content-Type"] == "application/octet-stream"


def test_indptr_has_one_extra_entry(live):
    base, store = live
    body, _ = get(base, "/api/pack/indptr")
    assert len(body) == (store.pack.n_neurons + 1) * 4
    assert np.frombuffer(body, dtype=np.uint32)[-1] == store.pack.n_edges


def test_activate_returns_hits_and_a_replayable_wavefront(live):
    base, _ = live
    data = post(base, "/api/activate", {"cue": "crates.io registry", "k": 3})
    assert data["hits"], "expected at least one recalled memory"
    assert "crates.io" in data["hits"][0]["memory"]["text"]

    fronts = data["activation"]["wavefronts"]
    assert fronts and fronts[0]["hop"] == 0
    for front in fronts:
        assert len(front["index"]) == len(front["level"])
    assert data["activation"]["lit"] > 0


def test_experience_adds_a_memory(live):
    base, store = live
    before = len(store.memories)
    data = post(base, "/api/experience", {"text": "a fact written over http", "salience": 0.9})
    assert data["memory"]["tag_size"] > 0
    assert len(store.memories) == before + 1


def test_neuron_detail_reports_real_annotations(live):
    base, store = live
    data = post(base, "/api/neuron", {"index": 0})
    assert data["root_id"] == str(store.pack.root_ids[0])
    assert isinstance(data["targets"], list)
    assert len(data["position_um"]) == 3


def test_bad_input_is_rejected_not_crashed(live):
    base, store = live
    assert "error" in post(base, "/api/activate", {"cue": "  "})
    assert "error" in post(base, "/api/experience", {"text": ""})
    assert "error" in post(base, "/api/neuron", {"index": store.pack.n_neurons + 10})


def test_viewer_is_served(live):
    base, _ = live
    index, _ = get(base, "/")
    assert b"FLUCTFLY" in index
    module, _ = get(base, "/src/app.js")
    assert b"NeuronCloud" in module


def test_static_paths_cannot_escape_the_web_root(live):
    base, _ = live
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(base, "/../pyproject.toml")
    assert exc.value.code == 404
