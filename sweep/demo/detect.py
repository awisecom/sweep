"""Demo target: find step changes in noisy signals.

A synthetic benchmark with the same shape as the original experiment: 24
detector variants times 1,728 settings is 41,472 configurations. Everything is
generated from seeds, so every number in the README can be reproduced.

The signals are what makes this hard in the same ways real data is: a level
that jumps a few times, Gaussian noise at three different scales, some
autocorrelation, a slow drift, and on some signals outlier spikes. A detector
is scored by F1 against the true change points, with a tolerance of 8 samples.

Pipeline, one factor per stage (4 x 3 x 2 = 24 detector variants, times 1,728 settings):
  smoothing      none, EMA, or a 3-point median
  decimate       keep every k-th sample (cheaper, coarser)
  statistic      mean shift, median shift, z-score or CUSUM
  normalisation  how the noise level is estimated: plain std, robust MAD, or not at all
  confirm        samples above threshold before firing
  window         length of the statistic's windows, in original samples
  threshold      score needed to fire, in noise units
  debounce       minimum samples between two detections
  rearm          score must fall below rearm * threshold before the next detection
"""

from __future__ import annotations

import itertools
import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

LENGTH = 600
TOLERANCE = 8

STATISTICS = ("meanshift", "medianshift", "zscore", "cusum")
NORMALISATIONS = ("global", "mad", "none")
CONFIRMATIONS = (1, 2)


@dataclass(frozen=True, slots=True)
class Signal:
    values: tuple[float, ...]
    changes: tuple[int, ...]  # true change points
    sigma: float


@cache
def make_signal(case: int, length: int = LENGTH) -> Signal:
    rng = random.Random(1_000_003 * case + 17)
    sigma = rng.choice((0.5, 1.0, 2.0))
    count = rng.randint(3, 6)
    while True:
        changes = sorted(rng.sample(range(40, length - 40), count))
        if all(b - a >= 60 for a, b in itertools.pairwise(changes)):
            break
    drift = rng.uniform(0.0, 0.6) * sigma
    period = rng.uniform(150.0, 400.0)
    phase = rng.uniform(0.0, 2 * math.pi)
    ar = rng.uniform(0.0, 0.4)
    spikes = rng.choice((0.0, 0.005, 0.02))

    values: list[float] = []
    level, noise, upcoming = 0.0, 0.0, list(changes)
    for t in range(length):
        if upcoming and t == upcoming[0]:
            level += rng.choice((-1.0, 1.0)) * rng.uniform(1.5, 4.0) * sigma
            upcoming.pop(0)
        noise = ar * noise + math.sqrt(1 - ar * ar) * rng.gauss(0.0, sigma)
        x = level + noise + drift * math.sin(2 * math.pi * t / period + phase)
        if rng.random() < spikes:
            x += rng.choice((-1.0, 1.0)) * rng.uniform(4.0, 8.0) * sigma
        values.append(x)
    return Signal(tuple(values), tuple(changes), sigma)


@cache
def prepared(case: int, length: int, smoothing: str, decimate: int) -> tuple[float, ...]:
    x = make_signal(case, length).values
    if smoothing == "ema":
        y, acc = [], x[0]
        for v in x:
            acc = 0.3 * v + 0.7 * acc
            y.append(acc)
    elif smoothing == "median3":
        y = [x[0]] + [sorted(x[i - 1 : i + 2])[1] for i in range(1, len(x) - 1)] + [x[-1]]
    elif smoothing == "none":
        y = list(x)
    else:
        raise ValueError(f"unknown smoothing {smoothing!r}")
    return tuple(y[::decimate])


@cache
def noise_scale(case: int, length: int, smoothing: str, decimate: int, normalisation: str) -> float:
    """Noise level estimated from first differences: plain std, or robust MAD."""
    if normalisation == "none":
        return 1.0
    x = prepared(case, length, smoothing, decimate)
    d = [b - a for a, b in itertools.pairwise(x)]
    if normalisation == "global":
        s = statistics.pstdev(d) / math.sqrt(2)
    elif normalisation == "mad":
        med = statistics.median(d)
        s = 1.4826 * statistics.median(abs(v - med) for v in d) / math.sqrt(2)
    else:
        raise ValueError(f"unknown normalisation {normalisation!r}")
    return max(s, 1e-9)


@cache
def scores(
    case: int, length: int, smoothing: str, decimate: int, statistic: str, normalisation: str, window: int
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Score per sample, and where the change would be if a detection fired there."""
    x = prepared(case, length, smoothing, decimate)
    n, w = len(x), window
    sigma = noise_scale(case, length, smoothing, decimate, normalisation)
    raw = normalisation == "none"
    prefix = [0.0]
    for v in x:
        prefix.append(prefix[-1] + v)
    s = [0.0] * n
    pos = list(range(n))

    if statistic == "meanshift":  # mean of the last w samples vs the w before
        k = 1.0 if raw else sigma * math.sqrt(2 / w)
        for t in range(2 * w - 1, n):
            recent = prefix[t + 1] - prefix[t + 1 - w]
            before = prefix[t + 1 - w] - prefix[t + 1 - 2 * w]
            s[t] = abs(recent - before) / w / k
            pos[t] = t - w + 1
    elif statistic == "medianshift":  # same, with medians: ignores spikes
        k = 1.0 if raw else sigma * math.sqrt(2 / w) * math.sqrt(math.pi / 2)
        for t in range(2 * w - 1, n):
            recent = statistics.median(x[t - w + 1 : t + 1])
            before = statistics.median(x[t - 2 * w + 1 : t - w + 1])
            s[t] = abs(recent - before) / k
            pos[t] = t - w + 1
    elif statistic == "zscore":  # this sample vs the mean of the w before it
        k = 1.0 if raw else sigma * math.sqrt(1 + 1 / w)
        for t in range(w, n):
            s[t] = abs(x[t] - (prefix[t] - prefix[t - w]) / w) / k
    elif statistic == "cusum":  # two-sided CUSUM against a trailing mean, drift 0.5
        k = 1.0 if raw else sigma
        up = down = 0.0
        start_up = start_down = w
        for t in range(w, n):
            e = (x[t] - (prefix[t] - prefix[t - w]) / w) / k
            if up == 0.0:
                start_up = t
            if down == 0.0:
                start_down = t
            up = max(0.0, up + e - 0.5)
            down = max(0.0, down - e - 0.5)
            s[t], pos[t] = (up, start_up) if up >= down else (down, start_down)
    else:
        raise ValueError(f"unknown statistic {statistic!r}")
    return tuple(s), tuple(pos)


def detect(
    s: Sequence[float], pos: Sequence[int], threshold: float, confirm: int, debounce: int, rearm: float, peak: int
) -> list[int]:
    """Fire when the score stays >= threshold for `confirm` samples; report the
    change position at the local peak; then wait until the score falls below
    rearm * threshold, and at least `debounce` samples, before firing again."""
    found: list[int] = []
    armed, above, last = True, 0, -(10**9)
    n, floor = len(s), rearm * threshold
    for t in range(n):
        v = s[t]
        if armed:
            above = above + 1 if v >= threshold else 0
            if above >= confirm and t - last >= debounce:
                tp, limit = t, min(n - 1, t + peak)
                while tp < limit and s[tp + 1] > s[tp]:
                    tp += 1
                found.append(pos[tp])
                armed, above, last = False, 0, t
        elif v < floor:
            armed = True
    return found


def match(found: Sequence[int], truth: Sequence[int], tolerance: int) -> tuple[int, int, int]:
    """Greedy one-to-one matching within the tolerance: (true positives, false positives, misses)."""
    unused = sorted(found)
    tp = 0
    for change in truth:
        best = min(unused, key=lambda d: abs(d - change), default=None)
        if best is not None and abs(best - change) <= tolerance:
            unused.remove(best)
            tp += 1
    return tp, len(found) - tp, len(truth) - tp


def evaluate(
    params: dict[str, Any],
    seed: int,
    cases: Sequence[int] = (0, 1, 2, 3),
    length: int = LENGTH,
    tolerance: int = TOLERANCE,
) -> dict[str, float]:
    """Target for sweep: mean F1, precision and recall over the given signals.
    `seed` is unused: the signals come from their case numbers, so every
    configuration is judged on identical data."""
    dec = int(params["decimate"])
    window = max(2, round(int(params["window"]) / dec))  # the window is in original samples
    f1 = prec = rec = 0.0
    for case in cases:
        signal = make_signal(case, length)
        s, pos = scores(case, length, params["smoothing"], dec, params["statistic"], params["normalisation"], window)
        found = [
            p * dec
            for p in detect(
                s,
                pos,
                threshold=float(params["threshold"]),
                confirm=int(params["confirm"]),
                debounce=math.ceil(int(params["debounce"]) / dec),
                rearm=float(params["rearm"]),
                peak=window,
            )
        ]
        tp, fp, fn = match(found, signal.changes, tolerance)
        f1 += 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0
        prec += tp / (tp + fp) if (tp + fp) else 0.0
        rec += tp / (tp + fn) if (tp + fn) else 1.0
    n = len(cases)
    return {"f1": f1 / n, "precision": prec / n, "recall": rec / n}
