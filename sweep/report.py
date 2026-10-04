"""Plain-text and Markdown rendering of analyses and controlled sweeps."""

from __future__ import annotations

import json
from typing import Any

from sweep.analyze import Analysis
from sweep.controlled import Controlled
from sweep.svg import band_line, share_bars


def fmt(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def analysis_text(a: Analysis, top: int = 5) -> str:
    lines = [
        f"{a.n:,} configurations, {a.metric}: mean {a.mean:.3f}, std {a.std:.3f}"
        + ("" if a.balanced else f"  (incomplete grid: {a.n:,} of {a.expected:,}, shares are approximate)"),
        "",
        f"{'factor':<12} {'explains':>9}   {'best level':<22} {'worst level':<22} {'spread':>7}",
    ]
    for e in a.effects:
        lines.append(f"{e.factor:<12} {100 * e.share:>8.1f}%   {fmt(e.best):<22} {fmt(e.worst):<22} {e.spread:>7.3f}")
    pairs = ", ".join(f"{x} x {y} {100 * sh:.1f}%" for x, y, sh in a.interactions[:3])
    lines += [
        "",
        f"{'main effects':<20} {100 * a.main_share:>5.1f}%",
        f"{'pairs of factors':<20} {100 * a.pair_share:>5.1f}%   largest: {pairs}",
        f"{'higher-order':<20} {100 * a.higher_order:>5.1f}%",
    ]
    lines += ["", f"top {top} configurations:"]
    for params, metrics in a.top[:top]:
        shown = ", ".join(f"{k}={fmt(v)}" for k, v in params.items())
        lines.append(f"  {metrics[a.metric]:.3f}  {shown}")
    return "\n".join(lines)


def controlled_text(c: Controlled) -> str:
    base = c.baseline[c.factor]
    lines = [
        f"{c.factor} varied on {len(c.cases)} unseen cases, everything else at the best configuration",
        "",
        f"{c.factor:>12}  {c.metric + ' mean':>10}  {'95% interval':>15}  {'vs ' + fmt(base):>16}  {'wins':>6}",
    ]
    for p in c.points:
        mark = "  <- best in sweep" if p.value == base else ""
        lines.append(
            f"{fmt(p.value):>12}  {p.mean:>10.3f}  {p.lo:>6.3f} .. {p.hi:<6.3f}  "
            f"{p.diff:>+7.3f} ({p.diff_lo:+.3f}..{p.diff_hi:+.3f})  {100 * p.wins:>5.0f}%{mark}"
        )
    return "\n".join(lines)


def shares_svg(a: Analysis) -> str:
    items = [(e.factor, e.share, True) for e in a.effects]
    items.append(("pairs of factors", a.pair_share, False))
    items.append(("higher-order", a.higher_order, False))
    return share_bars(
        items,
        title=f"What explains the variance in {a.metric}",
        subtitle=f"{a.n:,} configurations. Share of total variance (eta squared): each factor, then interactions.",
    )


def controlled_svg(c: Controlled) -> str:
    return band_line(
        [fmt(p.value) for p in c.points],
        [p.mean for p in c.points],
        [p.lo for p in c.points],
        [p.hi for p in c.points],
        title=f"{c.metric} by {c.factor}, on {len(c.cases)} signals the sweep never saw",
        subtitle="Every other factor held at the best configuration. Band: 95% bootstrap interval of the mean.",
        x_label=c.factor,
        y_label=f"mean {c.metric}",
        marker=fmt(c.baseline[c.factor]),
    )
