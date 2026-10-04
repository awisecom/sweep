"""Controlled sweep: prove the effect on cases the sweep never saw.

The full sweep says which factor explains the most variance, measured on the
spec's own cases. That can still be an artefact of those few cases. So:
hold every other factor at the best configuration, vary only the one factor,
and evaluate each value on a fresh set of test cases. Every value sees the
same cases, so the comparison is paired: for each case, how much better or
worse is this value than the baseline value?

Uncertainty comes from a bootstrap over cases: resample the cases with
replacement many times and take the 2.5th and 97.5th percentiles of the mean.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from sweep.spec import Spec, config_id, config_seed


@dataclass(slots=True)
class Point:
    value: Any
    mean: float
    lo: float  # 95% interval of the mean
    hi: float
    diff: float  # mean paired difference against the baseline value
    diff_lo: float
    diff_hi: float
    wins: float  # share of cases where this value beat the baseline value (ties count half)


@dataclass(slots=True)
class Controlled:
    factor: str
    metric: str
    baseline: dict[str, Any]
    cases: list[int]
    points: list[Point]

    @property
    def best(self) -> Point:
        return max(self.points, key=lambda p: p.mean)


def bootstrap_mean(values: list[float], rng: random.Random, resamples: int = 2000) -> tuple[float, float]:
    n = len(values)
    means = sorted(math.fsum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def controlled_sweep(
    spec: Spec,
    baseline: dict[str, Any],
    factor: str,
    values: list[Any] | None = None,
    cases: int = 86,
    first_case: int = 10_000,
    resamples: int = 2000,
) -> Controlled:
    if factor not in spec.grid:
        raise ValueError(f"unknown factor {factor!r}; the grid has {', '.join(spec.grid)}")
    values = list(values if values is not None else spec.grid[factor])
    case_ids = list(range(first_case, first_case + cases))
    target = spec.load_target()
    options = {k: v for k, v in spec.options.items() if k != "cases"}

    scores: dict[int, list[float]] = {}
    for i, value in enumerate(values):
        params = {**baseline, factor: value}
        seed = config_seed(spec.seed, config_id(params))
        scores[i] = [float(target(params, seed, cases=[c], **options)[spec.metric]) for c in case_ids]

    base_value = baseline[factor]
    base_scores = next((scores[i] for i, v in enumerate(values) if v == base_value), None)
    if base_scores is None:
        params = dict(baseline)
        seed = config_seed(spec.seed, config_id(params))
        base_scores = [float(target(params, seed, cases=[c], **options)[spec.metric]) for c in case_ids]

    rng = random.Random(spec.seed)
    points: list[Point] = []
    for i, value in enumerate(values):
        s = scores[i]
        diffs = [a - b for a, b in zip(s, base_scores, strict=True)]
        lo, hi = bootstrap_mean(s, rng, resamples)
        dlo, dhi = bootstrap_mean(diffs, rng, resamples)
        wins = math.fsum(1.0 if d > 0 else 0.5 if d == 0 else 0.0 for d in diffs) / len(diffs)
        points.append(Point(value, math.fsum(s) / len(s), lo, hi, math.fsum(diffs) / len(diffs), dlo, dhi, wins))
    return Controlled(factor, spec.metric, dict(baseline), case_ids, points)
