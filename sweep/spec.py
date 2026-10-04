"""Sweep specification: what to run, over which grid, scored by which metric.

A spec is a TOML file:

    [sweep]
    name = "detect"
    target = "sweep.demo.detect:evaluate"   # module:function(params, seed) -> {metric: value}
    metric = "f1"                           # the number the analysis explains
    seed = 2026

    [grid]
    detector = ["meanshift/global/1", ...]
    window = [5, 10, 20, 40]
    ...

Every configuration is the cartesian product of the grid, in a fixed order,
and has a stable id derived from its parameters. The same spec on another
machine produces the same ids, which is what makes runs resumable and
results comparable.
"""

from __future__ import annotations

import hashlib
import importlib
import itertools
import json
import math
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

Params = dict[str, Any]


class Target(Protocol):
    """What a sweep runs: target(params, seed, **options) -> {metric: value}.

    `options` come from the spec's [options] table. A target that supports the
    controlled sweep also accepts `cases`: which test cases to evaluate on."""

    def __call__(self, params: Params, seed: int, /, **options: Any) -> Mapping[str, float]: ...


@dataclass(frozen=True, slots=True)
class Spec:
    name: str
    target: str
    metric: str
    grid: dict[str, list[Any]]
    seed: int = 0
    maximize: bool = True
    options: dict[str, Any] = field(default_factory=dict)  # passed to the target as params["_options"]

    @property
    def size(self) -> int:
        return math.prod(len(v) for v in self.grid.values())

    @property
    def factors(self) -> list[str]:
        return list(self.grid)

    def configs(self) -> Iterator[Params]:
        keys = list(self.grid)
        for values in itertools.product(*(self.grid[k] for k in keys)):
            yield dict(zip(keys, values, strict=True))

    def load_target(self) -> Target:
        return load_target(self.target)


def config_id(params: Params) -> str:
    """Stable 16-hex-digit id: same parameters, same id, on any machine."""
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode()).hexdigest()[:16]


def config_seed(spec_seed: int, cid: str) -> int:
    """Deterministic per-config seed, independent of scheduling and worker count."""
    return (int(cid[:8], 16) ^ spec_seed) & 0x7FFFFFFF


def load_target(path: str) -> Target:
    module, _, func = path.partition(":")
    if not func:
        raise ValueError(f"target must look like 'package.module:function', got {path!r}")
    fn = getattr(importlib.import_module(module), func)
    if not callable(fn):
        raise TypeError(f"{path} is not callable")
    target: Target = fn
    return target


def load_spec(path: str | Path) -> Spec:
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    head = data.get("sweep", {})
    grid = data.get("grid", {})
    if not grid:
        raise ValueError(f"{path}: [grid] is empty")
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"{path}: grid.{key} must be a non-empty list")
        if len(set(map(json.dumps, values))) != len(values):
            raise ValueError(f"{path}: grid.{key} has duplicate values")
    try:
        return Spec(
            name=str(head["name"]),
            target=str(head["target"]),
            metric=str(head["metric"]),
            grid={k: list(v) for k, v in grid.items()},
            seed=int(head.get("seed", 0)),
            maximize=bool(head.get("maximize", True)),
            options=dict(data.get("options", {})),
        )
    except KeyError as missing:
        raise ValueError(f"{path}: [sweep] needs {missing}") from None
