"""Small targets with known answers, importable by worker processes."""

from __future__ import annotations

import random
import time
from typing import Any


def additive(params: dict[str, Any], seed: int, **_: Any) -> dict[str, float]:
    return {"y": 10.0 * params["a"] + params["b"]}


def xor(params: dict[str, Any], seed: int, **_: Any) -> dict[str, float]:
    return {"y": float(params["a"] != params["b"])}


def flaky(params: dict[str, Any], seed: int, **_: Any) -> dict[str, float]:
    if params["a"] == 2:
        raise RuntimeError("boom")
    return {"y": 1.0}


def slow(params: dict[str, Any], seed: int, **_: Any) -> dict[str, float]:
    if params["a"] == 1:
        time.sleep(5)
    return {"y": 1.0}


def seeded(params: dict[str, Any], seed: int, **_: Any) -> dict[str, float]:
    return {"y": random.Random(seed).random()}


def per_case(params: dict[str, Any], seed: int, cases: tuple[int, ...] = (0,), **_: Any) -> dict[str, float]:
    """Higher `a` is better by 0.1 per step, plus case-level noise shared by all values of `a`."""
    noise = sum(random.Random(c).gauss(0, 0.05) for c in cases) / len(cases)
    return {"y": 0.1 * params["a"] + noise}
