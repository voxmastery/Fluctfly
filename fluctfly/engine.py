"""Fluctfly: FluctlightDB's memory contract, served by a real fly connectome.

FluctlightDB's premise is that a database for agents should have memory
operations as its native verbs - `experience()` to write, `activate()` to recall
from a cue, `checkpoint()` to survive a restart - and that recall should fuse
lexical, semantic and *graph* evidence rather than returning nearest neighbours.

Fluctfly keeps that contract and replaces the graph with the wiring diagram of
an adult female *Drosophila melanogaster*, as reconstructed by FlyWire from
electron microscopy.  Writing a memory ignites a cue on the brain's afferent
neurons, lets it propagate across real chemical synapses, and stores the sparse
set of neurons that ended up firing.  Recalling runs the same propagation and
ranks memories by how much their engrams overlap the new activation.

That is the fly's own algorithm.  The mushroom body expands olfactory input
onto a large population of Kenyon cells through sparse, effectively random
connectivity, then silences all but the few percent that respond most strongly;
the resulting tag is a locality-sensitive hash of the input, which is why
similar odours stay similar after the expansion (Dasgupta, Stevens & Navlakha,
*Science* 358, 793-796, 2017).  Fluctfly runs that scheme on the measured
connectivity instead of on a random matrix.

If the `fluctlightdb` package is installed, every write is mirrored into a real
FluctlightDB brain and its recall score is fused with the connectome's, so the
two systems rank together.  Without it, Fluctfly runs standalone.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .encode import CueEncoder
from .pack import BrainPack
from .spread import Activation, Spreader, cosine


@dataclass
class Memory:
    """One stored experience and the neurons it lives in."""

    id: str
    text: str
    context: str = ""
    salience: float = 0.5
    verified: bool = False
    provenance: str = "chat"
    created: float = field(default_factory=time.time)
    rehearsals: int = 0
    tag: list[int] = field(default_factory=list)      # engram: neuron indices
    level: list[float] = field(default_factory=list)  # activation of those neurons
    ignition: list[int] = field(default_factory=list)  # afferents the cue entered through

    def as_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["tag_size"] = len(self.tag)
        return d


@dataclass
class Recall:
    memory: Memory
    score: float
    connectome: float
    fluctlight: float | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "memory": self.memory.as_json(),
            "score": round(self.score, 5),
            "connectome": round(self.connectome, 5),
            "fluctlight": None if self.fluctlight is None else round(self.fluctlight, 5),
        }


class Fluctfly:
    """A memory store whose index is a fly brain."""

    def __init__(
        self,
        pack: BrainPack,
        path: str | Path | None = None,
        tag_size: int = 64,
        hops: int = 4,
        width: int = 1024,
        fluctlight_path: str | Path | None = None,
        embed: Callable[[str], Sequence[float]] | None = None,
    ):
        self.pack = pack
        self.path = Path(path) if path else None
        self.tag_size = tag_size
        self.encoder = CueEncoder(pack, embed=embed)
        self.spreader = Spreader(pack, hops=hops, width=width)
        self.memories: dict[str, Memory] = {}
        self._brain = self._open_fluctlight(fluctlight_path)
        if self.path and (self.path / "memories.jsonl").exists():
            self.load()

    # -- optional FluctlightDB bridge ----------------------------------------

    def _open_fluctlight(self, path: str | Path | None):
        """Attach a real FluctlightDB brain if the package is installed."""
        if path is None:
            return None
        try:
            from fluctlightdb import connect_embedded  # type: ignore
        except ImportError:
            print("[fluctfly] fluctlightdb not installed - running connectome-only")
            return None
        brain = connect_embedded(str(path))
        print(f"[fluctfly] mirroring writes into FluctlightDB brain at {path}")
        return brain

    @property
    def fused(self) -> bool:
        """True when a FluctlightDB brain is backing this store."""
        return self._brain is not None

    # -- write ---------------------------------------------------------------

    def experience(
        self,
        text: str,
        context: str = "",
        salience: float = 0.5,
        verified: bool = False,
        provenance: str = "chat",
    ) -> tuple[Memory, Activation]:
        """Write a memory, binding it to the neurons its cue ignites.

        Writing something the store already holds rehearses it instead of
        storing it twice: the engram is unchanged, so a duplicate would only
        split the same evidence across two rows and crowd the recall list.
        Repetition raises salience, which is what it does in a brain.
        """
        existing = self._find(text, context)
        if existing is not None:
            existing.rehearsals += 1
            existing.salience = float(min(1.0, existing.salience + 0.1))
            if verified and not existing.verified:
                existing.verified, existing.provenance = True, provenance
            return existing, self.spreader.spread(
                self.encoder.encode(f"{context} {text}".strip())
            )

        activation = self.spreader.spread(self.encoder.encode(f"{context} {text}".strip()))
        tag, level = activation.sparse(self.tag_size)
        memory = Memory(
            id=uuid.uuid4().hex[:12],
            text=text,
            context=context,
            salience=float(np.clip(salience, 0.0, 1.0)),
            verified=verified,
            provenance=provenance,
            tag=tag.tolist(),
            level=[round(float(v), 5) for v in level],
            ignition=activation.ignition[:64].astype(int).tolist(),
        )
        self.memories[memory.id] = memory

        if self._brain is not None:
            try:
                self._brain.turn_begin()
                self._brain.wm_push(text, context=context or "fluctfly", salience=memory.salience)
                self._brain.turn_end(flush=True)
            except Exception as exc:  # the connectome store must survive a bridge failure
                print(f"[fluctfly] FluctlightDB write skipped: {exc}")

        return memory, activation

    def _find(self, text: str, context: str) -> Memory | None:
        key = (" ".join(text.split()).casefold(), context.strip().casefold())
        for memory in self.memories.values():
            if (" ".join(memory.text.split()).casefold(), memory.context.strip().casefold()) == key:
                return memory
        return None

    # -- read ----------------------------------------------------------------

    def activate(self, cue: str, k: int = 5) -> tuple[list[Recall], Activation]:
        """Recall from a cue by propagating it through the connectome."""
        activation = self.spreader.spread(self.encoder.encode(cue))
        probe_idx, probe_val = activation.sparse(self.tag_size)

        lexical = self._fluctlight_scores(cue)
        results: list[Recall] = []
        for memory in self.memories.values():
            conn = cosine(
                probe_idx,
                probe_val,
                np.asarray(memory.tag, dtype=np.uint32),
                np.asarray(memory.level, dtype=np.float32),
            )
            lex = lexical.get(memory.text)
            results.append(
                Recall(
                    memory=memory,
                    score=self._fuse(conn, lex, memory),
                    connectome=conn,
                    fluctlight=lex,
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return [r for r in results[:k] if r.score > 0], activation

    def _fuse(self, connectome: float, lexical: float | None, memory: Memory) -> float:
        """Blend connectome overlap with provenance, salience and FluctlightDB.

        Provenance dominance is FluctlightDB's rule, kept here: an observation
        backed by a file or a tool result outranks the same claim made in chat.
        It is applied as a tie-break rather than a multiplier large enough to
        drag an irrelevant memory above a relevant one - evidence quality should
        decide between plausible answers, not manufacture one.
        """
        score = connectome
        if lexical is not None:
            score = 0.6 * connectome + 0.4 * lexical
        score *= 0.9 + 0.2 * memory.salience
        if memory.verified:
            score *= 1.15
        return float(score)

    def _fluctlight_scores(self, cue: str) -> dict[str, float]:
        if self._brain is None:
            return {}
        try:
            hits = self._brain.recall(cue)
        except Exception as exc:
            print(f"[fluctfly] FluctlightDB recall skipped: {exc}")
            return {}
        scores: dict[str, float] = {}
        for rank, hit in enumerate(self._normalise_hits(hits)):
            text, score = hit
            scores[text] = score if score is not None else 1.0 / (1.0 + rank)
        return scores

    @staticmethod
    def _normalise_hits(hits: Any) -> Iterable[tuple[str, float | None]]:
        """FluctlightDB versions differ in what `recall` returns; accept them all."""
        if hits is None:
            return []
        out: list[tuple[str, float | None]] = []
        for hit in hits:
            if isinstance(hit, str):
                out.append((hit, None))
            elif isinstance(hit, dict):
                text = hit.get("text") or hit.get("content") or hit.get("summary")
                if text:
                    out.append((text, hit.get("score")))
            elif isinstance(hit, (tuple, list)) and hit:
                out.append((str(hit[0]), float(hit[1]) if len(hit) > 1 else None))
            else:
                text = getattr(hit, "text", None) or getattr(hit, "content", None)
                if text:
                    out.append((text, getattr(hit, "score", None)))
        return out

    # -- persistence ---------------------------------------------------------

    def checkpoint(self, path: str | Path | None = None) -> Path:
        """Persist every memory and its engram."""
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("no path given to checkpoint() and none set on the store")
        target.mkdir(parents=True, exist_ok=True)
        with (target / "memories.jsonl").open("w") as fh:
            for memory in self.memories.values():
                fh.write(json.dumps(asdict(memory)) + "\n")
        (target / "store.json").write_text(
            json.dumps(
                {
                    "pack": self.pack.manifest.get("subset"),
                    "dataset": self.pack.manifest.get("dataset"),
                    "n_neurons": self.pack.n_neurons,
                    "tag_size": self.tag_size,
                    "count": len(self.memories),
                },
                indent=2,
            )
        )
        if self._brain is not None:
            try:
                self._brain.checkpoint()
            except Exception as exc:
                print(f"[fluctfly] FluctlightDB checkpoint skipped: {exc}")
        self.path = target
        return target

    def load(self, path: str | Path | None = None) -> int:
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("no path to load from")
        f = target / "memories.jsonl"
        if not f.exists():
            return 0
        self.memories.clear()
        for line in f.read_text().splitlines():
            if line.strip():
                memory = Memory(**json.loads(line))
                self.memories[memory.id] = memory
        self.path = target
        return len(self.memories)
