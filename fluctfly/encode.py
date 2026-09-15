"""Turn text into activity on the brain's afferent neurons.

Nothing here tries to be a language model.  The job is only to get a cue into
the brain in a way that is deterministic, offline, and stable across restarts:
similar strings must land on overlapping input neurons, because everything
downstream (the real connectome) does the actual work of turning that into a
distributed code.

The input population is the set of neurons that carry signal *into* the brain -
olfactory projection neurons in the mushroom-body pack, the full sensory and
ascending complement in the larger ones.  A hashed bag of character n-grams and
word tokens is sprayed across that population, which plays the part the antennal
lobe plays for an odour: a dense, low-dimensional description that the mushroom
body is about to expand and sparsify.
"""

from __future__ import annotations

import hashlib
import re
from typing import Callable, Sequence

import numpy as np

from .pack import BrainPack

_TOKEN = re.compile(r"[a-z0-9]+")

# Afferent classes, in order of preference.  The first one that is actually
# present in the pack becomes the input population.
INPUT_CELL_CLASSES = ("ALPN",)
INPUT_SUPER_CLASSES = ("sensory", "sensory_ascending", "ascending", "visual_projection")


def _hash(token: str, salt: str = "") -> int:
    return int.from_bytes(hashlib.blake2b((salt + token).encode(), digest_size=8).digest(), "little")


def input_population(pack: BrainPack) -> np.ndarray:
    """Indices of the neurons a cue is allowed to enter through."""
    cc_vocab = pack.vocab("cell_class")
    for name in INPUT_CELL_CLASSES:
        if name in cc_vocab:
            idx = np.flatnonzero(pack.cell_class == cc_vocab.index(name))
            if idx.size:
                return idx
    sc_vocab = pack.vocab("super_class")
    wanted = [sc_vocab.index(n) for n in INPUT_SUPER_CLASSES if n in sc_vocab]
    if wanted:
        idx = np.flatnonzero(np.isin(pack.super_class, wanted))
        if idx.size:
            return idx
    # Degenerate pack with no afferents annotated: fall back to the neurons with
    # the most outgoing drive, which is where signal would enter anyway.
    deg = np.diff(pack.indptr.astype(np.int64))
    return np.argsort(deg)[::-1][: max(1, pack.n_neurons // 20)]


class CueEncoder:
    """Deterministic text -> afferent activity map for one brain pack.

    Two ways in.  By default a cue is hashed: word tokens plus character
    4-grams are sprayed across the afferent population, which costs nothing and
    works offline, but only matches text that shares surface form.  Pass
    ``embed`` - any callable returning a fixed-length vector, such as a
    sentence-transformer's ``encode`` - and the cue is instead projected onto
    the afferents through a fixed random matrix.  A random projection preserves
    cosine similarity (Johnson-Lindenstrauss), so paraphrases arrive at
    overlapping afferents and the connectome can carry the semantics the rest
    of the way.
    """

    def __init__(
        self,
        pack: BrainPack,
        fan: int = 12,
        salt: str = "fluctfly/v1",
        embed: Callable[[str], Sequence[float]] | None = None,
    ):
        self.pack = pack
        self.inputs = input_population(pack)
        self.n_inputs = int(self.inputs.size)
        self.fan = fan          # afferents recruited per token
        self.salt = salt
        self.embed = embed
        self._projection: np.ndarray | None = None

    @property
    def mode(self) -> str:
        return "embedding" if self.embed is not None else "lexical"

    def tokens(self, text: str) -> list[str]:
        """Word tokens plus 4-grams, so near-misses and typos still overlap."""
        low = text.lower()
        words = _TOKEN.findall(low)
        grams = [low[i : i + 4] for i in range(max(0, len(low) - 3))]
        return words + [w[:3] for w in words if len(w) > 3] + grams

    # -- the two routes in ---------------------------------------------------

    def _lexical(self, text: str) -> np.ndarray:
        activity = np.zeros(self.n_inputs, dtype=np.float32)
        for tok in self.tokens(text):
            h = _hash(tok, self.salt)
            for k in range(self.fan):
                slot = (h >> (k * 5)) ^ (h * (k + 1) & 0xFFFFFFFF)
                activity[slot % self.n_inputs] += 1.0
        # Odour intensity should not decide identity, so compress the counts the
        # way antennal-lobe gain control compresses concentration.
        return np.sqrt(activity)

    def _projected(self, text: str) -> np.ndarray:
        vector = np.asarray(self.embed(text), dtype=np.float32).ravel()  # type: ignore[misc]
        if self._projection is None or self._projection.shape[0] != vector.size:
            rng = np.random.default_rng(_hash("projection", self.salt) % (2**32))
            self._projection = rng.standard_normal(
                (vector.size, self.n_inputs), dtype=np.float32
            ) / np.sqrt(vector.size)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        # Half-wave rectify: afferents cannot fire negatively, and discarding
        # the negative half of the projection is what makes the code sparse.
        return np.maximum(vector @ self._projection, 0.0)

    def encode(self, text: str) -> np.ndarray:
        """Activation over the *whole* neuron population, non-zero only on afferents."""
        empty = np.zeros(self.pack.n_neurons, dtype=np.float32)
        if not text or not text.strip():
            return empty
        activity = self._projected(text) if self.embed is not None else self._lexical(text)
        norm = np.linalg.norm(activity)
        if norm == 0:
            return empty
        activity = activity / norm
        full = empty
        full[self.inputs] = activity
        return full


def sentence_encoder(model_name: str = "all-MiniLM-L6-v2") -> Callable[[str], Sequence[float]]:
    """An ``embed`` callable backed by sentence-transformers, if it is installed."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    model = SentenceTransformer(model_name)
    return lambda text: model.encode(text, normalize_embeddings=True)
