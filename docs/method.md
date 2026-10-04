# Method

## From a grid to a ranking of factors

A sweep produces one metric value per configuration. With `N` configurations, values `y`, and grand mean `ȳ`, the total sum of squares is

```text
SS_total = Σ (y - ȳ)²
```

For a factor `F` with levels `l`, each level has `n_l` configurations and a mean `ȳ_l`. The part of the variance that `F` explains on its own is

```text
SS_F    = Σ_l  n_l · (ȳ_l - ȳ)²
share_F = SS_F / SS_total          (eta squared)
```

In words: if knowing a configuration's level of `F` lets you predict its metric much better than the grand mean does, `F` explains a lot.

**Why the shares can be compared.** In a full factorial grid every level of every factor appears with every combination of the others the same number of times. That makes the main-effect sums of squares orthogonal: they add up, never double count, and their total is at most 1. That is the reason to run the full product instead of sampling it: the decomposition is exact.

**Pairs.** Group by two factors at once and compute the same quantity over the cells. Subtracting the two main effects leaves what the pair adds:

```text
SS_AB = Σ_cells n_ab · (ȳ_ab - ȳ)²  -  SS_A  -  SS_B
```

A large `SS_AB` means the best level of `A` depends on the level of `B`. In the demo, statistic x window is the largest pair: the best window is not the same for every statistic.

**Higher order.** `1 - Σ main - Σ pairs` is what only combinations of three or more factors explain. It is reported, not hidden. When it is large, no simple story covers the whole grid, and the honest next step is a closer look at the leading factor, which is exactly what the controlled sweep does.

`tests/test_analyze.py` pins the arithmetic with models whose answer is known: `y = 10a + b` must split 100/101 to 1/101 with no interaction, and `y = (a ≠ b)` must have zero main effects and a pair share of 1.

## From a ranking to a proof

The attribution is measured on the sweep's own cases, the same few signals for every configuration. Selecting the best configuration on those cases and reporting its score there overstates it. So the follow-up:

1. Take the best configuration from the sweep.
2. Vary only the leading factor, optionally on a finer set of values than the grid had.
3. Score each value on fresh cases (by default 86, numbered from 10,000 so they never overlap the sweep's).
4. Compare **paired**: for each case, the difference between this value and the baseline value. Case-to-case difficulty cancels out, which is far more sensitive than comparing two independent averages.
5. Bootstrap over cases (2,000 resamples) for 95% intervals of the mean and of the mean difference, and report the share of cases each value wins (ties count half).

A value is credibly worse than the baseline when its whole difference interval sits below zero.

## Pitfalls this guards against

| Pitfall | What sweep does |
|---|---|
| Comparing shares from an incomplete grid | Flags `incomplete grid ... shares are approximate` when runs are missing |
| Picking the best on the same data that scores it | Controlled sweep on unseen cases, gap reported (0.988 vs 0.926 in the demo) |
| Reading 0.0% as "never matters" | 0.0% means no effect on average; pairs are listed separately so a factor that only matters together with another is still visible |
| Random noise posing as an effect | Paired differences with bootstrap intervals, not single numbers |
| Results that depend on the machine | Ids and seeds from the parameters; equal results across worker counts are tested |

## Cost

The harness overhead is a hash per configuration, a pickle round trip per chunk and one SQLite transaction per chunk. Everything else is the target. Targets that share expensive stages across configurations should cache them per process, as the demo does with `functools.lru_cache`: its 41,472 configurations compute only a few thousand distinct smoothed signals and statistics.
