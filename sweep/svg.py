"""Dependency-free SVG charts for the report: variance shares and the controlled sweep.

Static images for a README. One accent colour (one series per chart, so no
legend box: the title names it), hairline grid, values labelled at the bar
ends, text in text colours, and a light/dark palette that follows the
viewer's colour scheme.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from xml.sax.saxutils import escape

LIGHT = {
    "accent": "#2a78d6",
    "wash": "rgba(42,120,214,0.12)",
    "rest": "#c3c2b7",
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}
DARK = {
    "accent": "#3987e5",
    "wash": "rgba(57,135,229,0.18)",
    "rest": "#5f5e5a",
    "surface": "#1a1a19",
    "ink": "#ffffff",
    "ink2": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
}


def _head(width: int, height: int, title: str, subtitle: str) -> list[str]:
    def css(c: dict[str, str]) -> str:
        return ":root { " + " ".join(f"--{k}: {v};" for k, v in c.items()) + " }"

    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        "<style>",
        css(LIGHT),
        "@media (prefers-color-scheme: dark) { " + css(DARK) + " }",
        "text { font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; fill: var(--ink2); font-size: 12px; }",
        ".title { fill: var(--ink); font-size: 16px; font-weight: 600; }",
        ".tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }",
        ".value { fill: var(--ink); font-variant-numeric: tabular-nums; }",
        ".grid { stroke: var(--grid); stroke-width: 1; } .axis { stroke: var(--axis); stroke-width: 1; }",
        ".bar { fill: var(--accent); } .rest { fill: var(--rest); } .band { fill: var(--wash); }",
        ".line { fill: none; stroke: var(--accent); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }",
        ".dot { fill: var(--accent); stroke: var(--surface); stroke-width: 2; } .bg { fill: var(--surface); }",
        ".mark { stroke: var(--muted); stroke-width: 1; }",
        "</style>",
        f'<rect class="bg" width="{width}" height="{height}" rx="8"/>',
        f'<text class="title" x="24" y="30">{escape(title)}</text>',
        f'<text x="24" y="50">{escape(subtitle)}</text>',
    ]


def _bar(x: float, y: float, w: float, h: float, cls: str) -> str:
    """Horizontal bar: square at the baseline, 4px rounded data end."""
    r = min(4.0, w / 2, h / 2)
    if w <= 0:
        return ""
    return (
        f'<path class="{cls}" d="M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} '
        f'V{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z"/>'
    )


def share_bars(items: Sequence[tuple[str, float, bool]], title: str, subtitle: str, width: int = 760) -> str:
    """items: (label, share 0..1, is_factor). Non-factor rows (the remainder) are drawn neutral."""
    label_w, top, row, bar_h = 150, 74, 30, 18
    plot_w = width - label_w - 90
    height = top + row * len(items) + 34
    o = _head(width, height, title, subtitle)
    x0 = 24 + label_w
    axis_max = 0.5 if max((share for _, share, _ in items), default=0.0) <= 0.5 else 1.0
    ticks = range(0, 51, 10) if axis_max == 0.5 else range(0, 101, 25)
    for pct in ticks:
        x = x0 + plot_w * (pct / 100) / axis_max
        o.append(f'<line class="grid" x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{top + row * len(items) - 6}"/>')
        o.append(f'<text class="tick" x="{x:.1f}" y="{top + row * len(items) + 12}" text-anchor="middle">{pct}%</text>')
    for i, (label, share, is_factor) in enumerate(items):
        y = top + i * row
        o.append(f'<text x="{x0 - 10}" y="{y + bar_h - 4}" text-anchor="end">{escape(label)}</text>')
        w = plot_w * max(0.0, min(axis_max, share)) / axis_max
        o.append(_bar(x0, y, w, bar_h, "bar" if is_factor else "rest"))
        o.append(f'<text class="value" x="{x0 + w + 8:.1f}" y="{y + bar_h - 4}">{100 * share:.1f}%</text>')
    o.append(f'<line class="axis" x1="{x0}" y1="{top - 6}" x2="{x0}" y2="{top + row * len(items) - 6}"/>')
    o.append("</svg>")
    return "\n".join(o) + "\n"


def band_line(
    xs: Sequence[Any],
    mean: Sequence[float],
    lo: Sequence[float],
    hi: Sequence[float],
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    marker: Any = None,
    width: int = 760,
    height: int = 340,
) -> str:
    """Mean with a 95% band over categorical x positions; `marker` labels one x (the baseline)."""
    left, right, top, bottom = 64, 40, 78, 52
    pw, ph = width - left - right, height - top - bottom
    y_min = max(0.0, min(lo) - 0.05)
    y_max = min(1.0, max(hi) + 0.05)
    span = (y_max - y_min) or 1.0
    n = len(xs)

    def sx(i: int) -> float:
        return left + (pw * i / (n - 1) if n > 1 else pw / 2)

    def sy(v: float) -> float:
        return top + ph * (1 - (v - y_min) / span)

    o = _head(width, height, title, subtitle)
    step = 0.05 if span <= 0.4 else 0.1
    v = round(y_min / step) * step
    while v <= y_max + 1e-9:
        if v >= y_min - 1e-9:
            o.append(f'<line class="grid" x1="{left}" y1="{sy(v):.1f}" x2="{left + pw}" y2="{sy(v):.1f}"/>')
            o.append(f'<text class="tick" x="{left - 8}" y="{sy(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>')
        v += step
    o.append(f'<text class="tick" x="{left - 8}" y="{top - 16}" text-anchor="end">{escape(y_label)}</text>')
    band = [f"{sx(i):.1f},{sy(h):.1f}" for i, h in enumerate(hi)] + [
        f"{sx(i):.1f},{sy(lv):.1f}" for i, lv in reversed(list(enumerate(lo)))
    ]
    o.append(f'<polygon class="band" points="{" ".join(band)}"/>')
    o.append(f'<polyline class="line" points="{" ".join(f"{sx(i):.1f},{sy(m):.1f}" for i, m in enumerate(mean))}"/>')
    for i, x in enumerate(xs):
        o.append(f'<circle class="dot" cx="{sx(i):.1f}" cy="{sy(mean[i]):.1f}" r="4"/>')
        o.append(f'<text class="tick" x="{sx(i):.1f}" y="{top + ph + 18}" text-anchor="middle">{escape(str(x))}</text>')
        if x == marker:
            o.append(f'<line class="mark" x1="{sx(i):.1f}" y1="{top}" x2="{sx(i):.1f}" y2="{top + ph}"/>')
            o.append(f'<text class="tick" x="{sx(i) + 6:.1f}" y="{top + ph - 8}">best in sweep</text>')
    best = max(range(n), key=lambda i: mean[i])
    o.append(
        f'<text class="value" x="{sx(best):.1f}" y="{sy(mean[best]) - 12:.1f}" text-anchor="middle">'
        f"{mean[best]:.3f}</text>"
    )
    o.append(f'<line class="axis" x1="{left}" y1="{top + ph}" x2="{left + pw}" y2="{top + ph}"/>')
    o.append(f'<text class="tick" x="{left + pw}" y="{top + ph + 38}" text-anchor="end">{escape(x_label)}</text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"
