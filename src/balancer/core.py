from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SEED = 42
DEFAULT_TARGET_SIZE = 20000
DEFAULT_HAGRID_RATIO = 0.5
DEFAULT_EXTREME_MST_LEVELS = (1, 2, 3, 10)
DEFAULT_EXTREME_FACTOR = 2.0
DEFAULT_JITTER_FACTOR = 0.0
DEFAULT_IMPUTE_MISSING_MST = True

@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    source: str
    gesture: str
    mst: int | None = None
    mst_origin: str = "missing"
