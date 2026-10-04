"""Which variable actually drove the result?

For a full factorial grid the total variance of the metric splits cleanly:
each factor's main effect, then interactions between factors, then whatever
is left. The share a factor explains is eta squared:

    eta2(f) = sum over levels of n_level * (mean_level - grand_mean)^2  /  total sum of squares

In a balanced design (every combination run exactly once) the main-effect
sums of squares are orthogonal, so the shares add up and can be compared
directly. A two-way interaction is what the pair explains beyond its two
main effects. If runs are missing the design is unbalanced and the shares are
approximate; the report says so.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sweep.spec import Spec


@dataclass(slots=True)
class Level:
    value: Any
    mean: float
    n: int


@dataclass(slots=True)
class Effect:
    factor: str
    share: float  # eta squared, 0..1
    levels: list[Level]
    best: Any
    worst: Any

    @property
    def spread(self) -> float:
        means = [lv.mean for lv in self.levels]
        return max(means) - min(means)


@dataclass(slots=True)
class Analysis:
    metric: str
    n: int
    expected: int
    mean: float
    std: float
    effects: list[Effect]  # sorted, biggest share first
    interactions: list[tuple[str, str, float]]  # every pair, biggest first
    top: list[tuple[dict[str, Any], dict[str, float]]] = field(default_factory=list)

    @property
    def balanced(self) -> bool:
        return self.n == self.expected

    @property
    def main_share(self) -> float:
        return sum(e.share for e in self.effects)

    @property
    def pair_share(self) -> float:
        return sum(s for _, _, s in self.interactions)

    @property
    def higher_order(self) -> float:
        """Three-way and higher interactions: what no single factor or pair explains."""
        return max(0.0, 1.0 - self.main_share - self.pair_share)

    @property
    def dominant(self) -> Effect:
        return self.effects[0]


def _key(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def analyze(
    spec: Spec,
    rows: Sequence[tuple[dict[str, Any], dict[str, float]]],
    top: int = 10,
) -> Analysis:
    if not rows:
        raise ValueError("no successful runs to analyze")
    y = [m[spec.metric] for _, m in rows]
    n = len(y)
    grand = math.fsum(y) / n
    ss_total = math.fsum((v - grand) ** 2 for v in y) or 1e-300

    def ss_of(keys: Sequence[str]) -> float:
        groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
        for (params, _), v in zip(rows, y, strict=True):
            groups[tuple(_key(params[k]) for k in keys)].append(v)
        return math.fsum(len(g) * (math.fsum(g) / len(g) - grand) ** 2 for g in groups.values())

    effects: list[Effect] = []
    main_ss: dict[str, float] = {}
    for factor in spec.factors:
        sums: dict[str, list[float]] = defaultdict(list)
        for (params, _), v in zip(rows, y, strict=True):
            sums[_key(params[factor])].append(v)
        levels = [
            Level(value, math.fsum(sums[_key(value)]) / len(sums[_key(value)]), len(sums[_key(value)]))
            for value in spec.grid[factor]
            if _key(value) in sums
        ]
        main_ss[factor] = ss_of([factor])
        ordered = sorted(levels, key=lambda lv: lv.mean, reverse=spec.maximize)
        effects.append(Effect(factor, main_ss[factor] / ss_total, levels, ordered[0].value, ordered[-1].value))
    effects.sort(key=lambda e: e.share, reverse=True)

    interactions: list[tuple[str, str, float]] = []
    for a, b in itertools.combinations(spec.factors, 2):
        ss_ab = ss_of([a, b]) - main_ss[a] - main_ss[b]  # what the pair adds beyond its main effects
        interactions.append((a, b, max(0.0, ss_ab) / ss_total))
    interactions.sort(key=lambda t: t[2], reverse=True)

    best = sorted(rows, key=lambda r: r[1][spec.metric], reverse=spec.maximize)[:top]
    return Analysis(
        metric=spec.metric,
        n=n,
        expected=spec.size,
        mean=grand,
        std=math.sqrt(ss_total / n),
        effects=effects,
        interactions=interactions,
        top=list(best),
    )
