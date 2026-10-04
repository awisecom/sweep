# sweep

Run every combination, then find out which variable actually mattered. A parallel, resumable parameter-sweep harness with variance attribution and controlled follow-up sweeps, in the Python standard library only.

[![ci](https://github.com/awisecom/sweep/actions/workflows/ci.yml/badge.svg)](https://github.com/awisecom/sweep/actions/workflows/ci.yml)

I built the original for my own trading research: 41,472 paper-trading configurations across 24 strategies, to isolate the one variable that drove the results, then a controlled sweep across 86 test signals to prove it. The strategies stay private. This repo is the harness, with a synthetic benchmark of exactly the same shape, so every number below reproduces with three commands.

## What it does

- **Full factorial from a TOML file.** Every configuration has a stable id (a hash of its parameters) and a seed derived from it, so the same spec gives the same results on any machine, with any number of workers.
- **Parallel, with flat memory.** A process pool fed in chunks, never more than four chunks per worker in flight. The demo's 41,472 configurations finish in 7.7 s on two cores.
- **Resumable.** Results go to SQLite (WAL) one transaction per chunk. Ctrl-C keeps everything finished; the next `run` does only what is left. A results file refuses a different spec instead of silently mixing them.
- **Failures are data.** An exception or a timeout is recorded against its configuration and the sweep moves on. `--retry-failed` reruns only those.
- **Attribution.** How much of the metric's variance each factor explains on its own, what every pair of factors adds on top, and what is left for higher-order interactions.
- **Proof on unseen cases.** A controlled sweep holds everything at the best configuration, varies one factor, scores each value on fresh test cases, and reports paired differences with bootstrap intervals and win rates.

## The demo: finding step changes in noisy signals

Each test signal is 600 samples with 3 to 6 jumps in level, buried in Gaussian noise at one of three scales, with autocorrelation, a slow drift and, on some signals, outlier spikes. A detector is scored by F1 against the true change points, with a tolerance of 8 samples. The grid ([examples/detect.toml](examples/detect.toml)):

| Factor | Levels | |
|---|---|---|
| `statistic` | mean shift, median shift, z-score, CUSUM | 4 x 3 x 2 = **24 detector variants** |
| `normalisation` | noise from plain std, robust MAD, or none | |
| `confirm` | 1 or 2 samples over threshold | |
| `window` | 10, 20, 40, 80 samples | 4 x 4 x 3 x 3 x 3 x 4 = **1,728 settings** each |
| `threshold` | 2, 3, 4, 5 (noise units) | |
| `smoothing` | none, EMA, 3-point median | |
| `debounce` | 0, 10, 25 samples between detections | |
| `rearm` | 0.25, 0.5, 0.8 of the threshold | |
| `decimate` | keep every 1st to 4th sample | |

```bash
pip install -e .
python -m sweep run examples/detect.toml            # 41,472 configurations
python -m sweep analyze examples/detect.toml
python -m sweep controlled examples/detect.toml --values 5,10,15,20,30,40,60,80
```

```text
41,472 configurations, f1: mean 0.378, std 0.216

factor        explains   best level             worst level             spread
window           16.6%   20                     80                       0.243
statistic         4.7%   meanshift              zscore                   0.125
decimate          3.3%   3                      1                        0.097
normalisation      2.5%   global                 none                     0.080
threshold         1.7%   3.0                    5.0                      0.075
smoothing         0.7%   median3                none                     0.038
confirm           0.0%   1                      2                        0.008
debounce          0.0%   25                     0                        0.009
rearm             0.0%   0.5                    0.25                     0.004

main effects          29.5%
pairs of factors      34.0%   largest: statistic x window 7.9%, statistic x smoothing 4.0%, ...
higher-order          36.5%
```

![share of variance explained by each factor](docs/variance.svg)

What that says:

1. **The window is the lever.** It explains 16.6% of the variance in F1 on its own, 3.5 times more than the next factor, the choice of statistic.
2. **Three of nine knobs do nothing.** `confirm`, `debounce` and `rearm` explain 0.0%: any value is as good as any other, so they stop being tuning work.
3. **Tuning doesn't transfer.** A third of the variance sits in pairs of factors, led by statistic x window: the best window depends on the statistic. Tuning one detector family and reusing the settings on another would be a mistake.

Then the proof. Everything held at the best configuration, the window varied more finely than the grid did, each value scored on 86 signals the sweep never saw:

```text
      window     f1 mean     95% interval             vs 20    wins
           5       0.527   0.480 .. 0.570    -0.399 (-0.444..-0.354)      3%
          10       0.781   0.741 .. 0.816    -0.145 (-0.181..-0.108)     18%
          15       0.891   0.865 .. 0.915    -0.034 (-0.059..-0.008)     40%
          20       0.926   0.902 .. 0.948    +0.000 (+0.000..+0.000)     50%  <- best in sweep
          30       0.887   0.857 .. 0.914    -0.038 (-0.069..-0.009)     37%
          40       0.843   0.804 .. 0.880    -0.082 (-0.119..-0.049)     33%
          60       0.667   0.620 .. 0.714    -0.259 (-0.313..-0.206)     16%
          80       0.448   0.395 .. 0.503    -0.477 (-0.532..-0.420)      5%
```

![F1 by window on unseen signals, peaking at 20](docs/controlled.svg)

A window of 20 holds up on new data: F1 0.926, and every other value is worse with an interval that excludes zero. The best configuration scored 0.988 on the 12 signals it was selected on and 0.926 on signals it never saw: a generalisation gap that is measured, not assumed.

## Bring your own target

A target is any function `(params, seed, **options) -> {metric: value}`:

```python
# mypkg/model.py
def evaluate(params: dict, seed: int, cases=(0,), **options) -> dict[str, float]:
    ...
    return {"score": score}
```

```toml
[sweep]
name = "mymodel"
target = "mypkg.model:evaluate"
metric = "score"
seed = 1

[options]          # passed to every call
cases = [0, 1, 2, 3]

[grid]
learning_rate = [0.01, 0.03, 0.1]
depth = [2, 4, 8]
```

`cases` is how the controlled sweep asks for unseen test cases; a target that ignores it still works everywhere else. `python -m sweep status` shows progress and the first failures.

## How the attribution works

For a full factorial grid the total variance of the metric splits into parts. A factor's share is eta squared: the variance of the group means for its levels, weighted by group size, over the total. In a balanced design those main-effect parts are orthogonal, so they add up and compare directly. A pair's interaction share is what the two factors explain together beyond their two main effects. Whatever is left belongs to interactions of three or more factors. Details, formulas and the pitfalls (unbalanced grids, overfitting to the sweep's own cases, reading a 0.0% correctly) are in [docs/method.md](docs/method.md).

## Design notes

- **Determinism is tested, not hoped for:** the same grid with 1 worker and with 3 produces identical results.
- **Caching belongs to the target.** The demo caches each pipeline stage per worker process. Thousands of configurations share the same smoothed signal and the same statistic and differ only in the cheap last stage, which is why 41,472 of them take seconds.
- **Timeouts** use `SIGALRM` inside the worker (POSIX), so one hanging configuration is recorded as `timeout` and the pool keeps going.
- **No dependencies:** `tomllib`, `sqlite3`, `concurrent.futures`, `statistics` and hand-written SVG.

## Tests

25 tests, including known-answer checks for the attribution (an additive model splits exactly 100/101 to 1/101; an XOR model is pure interaction with zero main effects), determinism across worker counts, resume after a partial run, error and timeout capture, the spec-mismatch guard, the bootstrap and pairing in the controlled sweep, the demo detector, and the command line end to end. CI runs them on Python 3.11 to 3.13 with ruff and mypy in strict mode, then runs the full 41,472-configuration demo.

---

© 2026 Aleksander Wisniewski. Shared to show how I work; all rights reserved.
