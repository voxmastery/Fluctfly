import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PACK_DIR = ROOT / "data" / "packs" / "mushroom_body"


@pytest.fixture(scope="session")
def pack():
    from fluctfly.pack import BrainPack

    if not (PACK_DIR / "manifest.json").exists():
        pytest.skip(f"no brain pack at {PACK_DIR}; run `make pack`")
    return BrainPack.load(PACK_DIR)


@pytest.fixture()
def store(pack):
    from fluctfly.demo import seed
    from fluctfly.engine import Fluctfly

    fly = Fluctfly(pack)
    seed(fly)
    return fly
