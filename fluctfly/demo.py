"""A handful of memories so the viewer has something to recall on first run."""

from __future__ import annotations

SEED = [
    # (text, context, salience, verified, provenance)
    ("The user prefers dark mode in the editor", "settings", 0.8, False, "chat"),
    ("Release target is the crates.io registry", "release", 0.7, True, "file:Cargo.toml"),
    ("CI runs on ubuntu-latest with Rust 1.83", "ci", 0.6, True, "file:.github/workflows/ci.yml"),
    ("The team standup is at 9:30am UTC on weekdays", "calendar", 0.4, False, "chat"),
    ("Kenyon cells sparsify olfactory input into a high dimensional code", "notes", 0.5, True,
     "doi:10.1126/science.aam9868"),
    ("FlyWire proofread 139,255 neurons and 54.5 million synapses", "notes", 0.6, True,
     "doi:10.1038/s41586-024-07558-y"),
    ("Glutamate is inhibitory in the fly brain via GluCl channels", "notes", 0.5, True,
     "doi:10.1016/j.cell.2024.03.016"),
    ("Postgres stores rows, vector databases store neighbours, neither stores memory",
     "manifesto", 0.7, False, "chat"),
    ("Checkpoint before a restart or the working memory is lost", "ops", 0.9, True, "file:docs/PRODUCTION.md"),
    ("The mushroom body output neurons number about one hundred", "notes", 0.5, True,
     "doi:10.1038/s41586-024-07558-y"),
]


def seed(store) -> int:
    """Write the demo memories into a Fluctfly store."""
    for text, context, salience, verified, provenance in SEED:
        store.experience(
            text, context=context, salience=salience, verified=verified, provenance=provenance
        )
    return len(SEED)
