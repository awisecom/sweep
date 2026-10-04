"""Command line: python -m sweep run | analyze | controlled | status."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sweep import __version__
from sweep.analyze import analyze
from sweep.controlled import controlled_sweep
from sweep.report import analysis_text, controlled_svg, controlled_text, shares_svg
from sweep.runner import run
from sweep.spec import Spec, load_spec
from sweep.store import Store


def _value(text: str) -> object:
    """'3.5' -> 3.5, 'true' -> True, 'ema' -> 'ema'."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text.strip()


def _db(spec: Spec, path: str | None) -> Path:
    return Path(path) if path else Path("runs") / f"{spec.name}.sqlite"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sweep", description="Run every combination, then find out which variable mattered."
    )
    p.add_argument("--version", action="version", version=f"sweep {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("spec", help="the sweep's TOML file")
        sp.add_argument("--db", help="results file (default: runs/<name>.sqlite)")

    r = sub.add_parser("run", help="run every configuration not yet in the results file")
    common(r)
    r.add_argument("--workers", type=int, help="processes (default: all CPUs)")
    r.add_argument("--chunk", type=int, default=64, help="configurations per task")
    r.add_argument("--limit", type=int, help="stop after this many configurations")
    r.add_argument("--timeout", type=float, default=0.0, help="seconds per configuration, 0 = none")
    r.add_argument("--retry-failed", action="store_true", help="run errored and timed-out configurations again")

    a = sub.add_parser("analyze", help="which factors explain the metric")
    common(a)
    a.add_argument("--top", type=int, default=5)
    a.add_argument("--svg", type=Path, help="write the variance chart here")
    a.add_argument("--json", action="store_true")

    c = sub.add_parser("controlled", help="vary one factor around the best configuration, on unseen cases")
    common(c)
    c.add_argument("--factor", help="default: the factor that explains the most variance")
    c.add_argument("--values", help="comma separated JSON values, default: the grid's levels")
    c.add_argument("--cases", type=int, default=86)
    c.add_argument("--svg", type=Path, help="write the chart here")

    s = sub.add_parser("status", help="how far a sweep has got")
    common(s)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    spec = load_spec(args.spec)
    db = _db(spec, args.db)

    if args.command == "run":
        with Store(db) as store:
            print(f"{spec.name}: {spec.size:,} configurations, results in {db}", file=sys.stderr)
            rep = run(spec, store, args.workers, args.chunk, args.limit, args.timeout, args.retry_failed)
        state = "interrupted" if rep.interrupted else "done"
        print(
            f"{state}: ran {rep.ran:,} in {rep.seconds:.1f} s ({rep.rate:,.0f}/s), {rep.ok:,} ok, "
            f"{rep.errors} errors, {rep.timeouts} timeouts; {rep.already_done:,} were already done"
        )
        return 130 if rep.interrupted else (1 if rep.errors or rep.timeouts else 0)

    with Store(db) as store:
        rows = list(store.results())
        counts = store.counts()
        failures = store.failures()

    if args.command == "status":
        done = sum(counts.values())
        print(f"{spec.name}: {done:,} of {spec.size:,} ({100 * done / spec.size:.1f}%), {counts}")
        for cid, status, error in failures:
            print(f"  {status:<7} {cid}  {error}")
        return 0

    analysis = analyze(spec, rows, top=max(args.top if args.command == "analyze" else 1, 1))

    if args.command == "analyze":
        if args.json:
            print(
                json.dumps(
                    {
                        "n": analysis.n,
                        "metric": analysis.metric,
                        "mean": analysis.mean,
                        "effects": [
                            {"factor": e.factor, "share": e.share, "best": e.best, "worst": e.worst}
                            for e in analysis.effects
                        ],
                        "pairs": [{"factors": [a, b], "share": s} for a, b, s in analysis.interactions],
                        "higher_order": analysis.higher_order,
                        "top": [{"params": p, "metrics": m} for p, m in analysis.top],
                    },
                    indent=2,
                )
            )
        else:
            print(analysis_text(analysis, args.top))
        if args.svg:
            args.svg.write_text(shares_svg(analysis), encoding="utf-8")
        return 0

    factor = args.factor or analysis.dominant.factor
    values = [_value(v) for v in args.values.split(",")] if args.values else None
    best_params = analysis.top[0][0]
    result = controlled_sweep(spec, best_params, factor, values, cases=args.cases)
    print(controlled_text(result))
    if args.svg:
        args.svg.write_text(controlled_svg(result), encoding="utf-8")
    return 0
