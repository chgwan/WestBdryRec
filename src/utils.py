# -*- coding: utf-8 -*-
"""Tiny shared helpers (parallel map)."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def pmap(fn: Callable[[T], R], items: Sequence[T], workers: int = 1,
         desc: str = "") -> List[R]:
    """Order-preserving parallel map. ``workers`` <= 1 runs serially (clean tracebacks)."""
    items = list(items)
    if not items:
        return []
    if workers is None or workers <= 1:
        return [fn(x) for x in items]
    out: List[R] = [None] * len(items)  # type: ignore[list-item]
    with ProcessPoolExecutor(max_workers=int(workers)) as ex:
        futs = {ex.submit(fn, x): i for i, x in enumerate(items)}
        for fut in as_completed(futs):
            out[futs[fut]] = fut.result()
    return out
