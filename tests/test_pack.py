"""The pack has to stay a valid CSR graph, because everything else assumes it."""

import numpy as np


def test_shapes_agree(pack):
    n, e = pack.n_neurons, pack.n_edges
    assert pack.pos.shape == (n, 3)
    assert pack.attr.shape == (n, 4)
    assert pack.root_ids.shape == (n,)
    assert pack.indptr.shape == (n + 1,)
    assert pack.indices.shape == (e,)
    assert pack.weight.shape == (e,)
    assert len(pack.cell_types) == n


def test_csr_is_well_formed(pack):
    indptr = pack.indptr.astype(np.int64)
    assert indptr[0] == 0
    assert indptr[-1] == pack.n_edges
    assert np.all(np.diff(indptr) >= 0), "row pointers must be non-decreasing"
    assert pack.indices.max(initial=0) < pack.n_neurons


def test_no_self_synapses(pack):
    row = np.repeat(np.arange(pack.n_neurons), np.diff(pack.indptr.astype(np.int64)))
    assert not np.any(row == pack.indices), "autapses should have been dropped"


def test_weights_are_input_normalised(pack):
    """Total absolute drive onto a neuron should not exceed one."""
    incoming = np.bincount(
        pack.indices.astype(np.int64),
        weights=np.abs(pack.weight).astype(np.float64),
        minlength=pack.n_neurons,
    )
    assert incoming.max() <= 1.0 + 1e-5
    # Most neurons that receive anything should be close to saturated: the
    # deficit is only the sub-threshold edges the build step dropped.
    receiving = incoming[incoming > 0]
    assert np.median(receiving) > 0.8


def test_signs_follow_the_presynaptic_transmitter(pack):
    from fluctfly.pack import NT_ORDER, NT_SIGN

    row = np.repeat(np.arange(pack.n_neurons), np.diff(pack.indptr.astype(np.int64)))
    expected = np.array([NT_SIGN[t] for t in NT_ORDER])[pack.nt[row]]
    nonzero = expected != 0
    assert np.all(np.sign(pack.weight[nonzero]) == np.sign(expected[nonzero]))


def test_positions_are_a_fly_brain(pack):
    """The FAFB volume is roughly 800 x 400 x 250 micrometres."""
    span = pack.pos.max(axis=0) - pack.pos.min(axis=0)
    assert 100 < span[0] < 1000
    assert np.all(span > 50)


def test_labels_are_readable(pack):
    label = pack.label(0)
    assert label["root_id"].isdigit()
    assert label["neurotransmitter"] in {
        "acetylcholine", "glutamate", "gaba", "dopamine",
        "serotonin", "octopamine", "unknown",
    }
    assert label["side"] in {"left", "right", "center", "na"}
