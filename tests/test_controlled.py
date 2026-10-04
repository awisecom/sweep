from __future__ import annotations

import random

from sweep.controlled import bootstrap_mean, controlled_sweep


def test_bootstrap_interval_brackets_the_mean() -> None:
    rng = random.Random(1)
    values = [rng.gauss(0.5, 0.1) for _ in range(200)]
    lo, hi = bootstrap_mean(values, random.Random(2))
    mean = sum(values) / len(values)
    assert lo < mean < hi
    assert hi - lo < 0.05


def test_a_controlled_sweep_ranks_values_and_pairs_them(make_spec) -> None:  # type: ignore[no-untyped-def]
    spec = make_spec("targets:per_case", "a = [1, 2, 3]\nb = [1]")
    result = controlled_sweep(spec, {"a": 2, "b": 1}, "a", cases=40, resamples=500)
    means = [p.mean for p in result.points]
    assert means == sorted(means)  # higher a is better
    by_value = {p.value: p for p in result.points}
    assert by_value[3].diff > 0 > by_value[1].diff
    assert by_value[3].wins == 1.0  # paired: shared case noise cancels out
    assert by_value[2].diff == 0.0
    assert result.best.value == 3
