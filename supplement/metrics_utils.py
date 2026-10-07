from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Set


def deep_getsizeof(obj: Any, seen: Set[int] | None = None) -> int:
    """
    Rough recursive size estimation (bytes) using sys.getsizeof.
    Good for relative comparisons; not a perfect measure of real memory usage.
    """
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        for k, v in obj.items():
            size += deep_getsizeof(k, seen)
            size += deep_getsizeof(v, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for i in obj:
            size += deep_getsizeof(i, seen)

    return size


@dataclass
class StageTimes:
    load_s: float = 0.0
    type_filter_s: float = 0.0
    spatial_match_s: float = 0.0
    prefix_tree_s: float = 0.0
    total_s: float = 0.0


class Timer:
    def __init__(self):
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def elapsed(self) -> float:
        return time.perf_counter() - self._t0
