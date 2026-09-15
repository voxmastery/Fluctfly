"""Encoding, spreading, and the memory contract."""

import numpy as np
import pytest

from fluctfly.encode import CueEncoder
from fluctfly.spread import Spreader, cosine


def test_encoding_is_deterministic(pack):
    enc = CueEncoder(pack)
    a, b = enc.encode("the user prefers dark mode"), enc.encode("the user prefers dark mode")
    assert np.array_equal(a, b)


def test_cue_only_enters_through_afferents(pack):
    enc = CueEncoder(pack)
    activity = enc.encode("an arbitrary cue")
    lit = np.flatnonzero(activity)
    assert lit.size > 0
    assert np.all(np.isin(lit, enc.inputs))


def test_empty_cue_lights_nothing(pack):
    enc = CueEncoder(pack)
    assert not enc.encode("   ").any()


def test_similar_cues_land_on_similar_afferents(pack):
    enc = CueEncoder(pack)
    near = float(enc.encode("deploy the rust crate") @ enc.encode("deploy the rust crates"))
    far = float(enc.encode("deploy the rust crate") @ enc.encode("mitochondria powerhouse cell"))
    assert near > far * 2


def test_spread_walks_the_connectome(pack):
    enc, spreader = CueEncoder(pack), Spreader(pack)
    activation = spreader.spread(enc.encode("olfactory projection neurons"))
    assert len(activation.wavefronts) > 1, "activation should leave the afferents"
    assert activation.field.max() == pytest.approx(1.0)

    # Every neuron in hop t+1 must be postsynaptic to something in hop t.
    for before, after in zip(activation.wavefronts, activation.wavefronts[1:]):
        reachable = set()
        for pre in before.index:
            reachable.update(pack.out_edges(int(pre))[0].tolist())
        assert set(after.index.tolist()) <= reachable


def test_engram_tags_separate_topics(pack):
    enc, spreader = CueEncoder(pack), Spreader(pack)
    tag = lambda s: spreader.spread(enc.encode(s)).sparse(64)  # noqa: E731
    a, b, c = tag("deploy the rust crate"), tag("publish the rust crate"), tag("standup at 9:30am")
    assert cosine(*a, *b) > cosine(*a, *c)


def test_recall_returns_the_right_memory(store):
    from fluctfly.demo import SEED

    for text, *_ in SEED:
        cue = " ".join(text.split()[:5])
        hits, _ = store.activate(cue, k=1)
        assert hits, f"nothing recalled for {cue!r}"
        assert hits[0].memory.text == text


def test_experience_binds_a_memory_to_neurons(store, pack):
    memory, activation = store.experience("a brand new observation", context="test")
    assert 0 < len(memory.tag) <= store.tag_size
    assert len(memory.tag) == len(memory.level)
    assert max(memory.tag) < pack.n_neurons
    assert activation.field.max() > 0


def test_verified_evidence_breaks_a_tie_but_does_not_invent_one(store):
    """Provenance should order plausible answers, not outrank the right one."""
    from fluctfly.engine import Memory

    chat = Memory(id="a", text="x", salience=0.5, verified=False)
    doc = Memory(id="b", text="y", salience=0.5, verified=True)
    assert store._fuse(0.40, None, doc) > store._fuse(0.40, None, chat)
    assert store._fuse(0.40, None, chat) > store._fuse(0.10, None, doc)


def test_checkpoint_round_trips(store, pack, tmp_path):
    from fluctfly.engine import Fluctfly

    before = {m.id: m.tag for m in store.memories.values()}
    store.checkpoint(tmp_path)

    reloaded = Fluctfly(pack, path=tmp_path)
    assert {m.id: m.tag for m in reloaded.memories.values()} == before

    hits, _ = reloaded.activate("crates.io registry", k=1)
    assert hits and "crates.io" in hits[0].memory.text


def test_runs_without_fluctlightdb(store):
    """The bridge is optional; the connectome store must work on its own."""
    assert store.fused is False
    hits, _ = store.activate("dark mode", k=1)
    assert hits and hits[0].fluctlight is None


def test_writing_the_same_thing_twice_rehearses_it(store):
    before = len(store.memories)
    first, _ = store.experience("a repeated observation", context="test", salience=0.4)
    again, _ = store.experience("  A Repeated   Observation ", context="test", salience=0.4)
    assert len(store.memories) == before + 1
    assert again.id == first.id
    assert again.rehearsals == 1
    assert again.salience > 0.4


def test_rehearsal_can_upgrade_provenance(store):
    store.experience("the port is 8799", context="ops", verified=False)
    memory, _ = store.experience("the port is 8799", context="ops", verified=True,
                                 provenance="file:Makefile")
    assert memory.verified and memory.provenance == "file:Makefile"


def test_same_text_in_a_different_context_is_a_different_memory(store):
    before = len(store.memories)
    store.experience("the default is five", context="threshold")
    store.experience("the default is five", context="retries")
    assert len(store.memories) == before + 2
