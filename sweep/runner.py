"""Run every configuration of a spec, in parallel, resumably.

Configurations go to a process pool in chunks (fewer round trips than one task
per configuration) with a bounded number of chunks in flight, so memory stays
flat whether the grid has 400 or 400,000 points. Results are written as each
chunk returns. Ctrl-C stops cleanly: finished chunks are kept, the rest runs
next time.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import itertools
import json
import os
import signal
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from types import FrameType
from typing import Any

from sweep.spec import Params, Spec, Target, config_id, config_seed, load_target
from sweep.store import Result, Store

Task = tuple[str, Params, int]  # (config id, params, seed)

_target: Target | None = None
_options: dict[str, Any] = {}
_timeout = 0.0


class _Timeout(Exception):
    pass


def _on_alarm(_signum: int, _frame: FrameType | None) -> None:
    raise _Timeout


def _init_worker(target: str, options: dict[str, Any], timeout: float, ignore_sigint: bool = True) -> None:
    global _target, _options, _timeout
    if ignore_sigint:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C belongs to the parent
    _target, _options, _timeout = load_target(target), options, timeout


def _run_chunk(chunk: list[Task]) -> list[Result]:
    assert _target is not None, "worker not initialised"
    use_alarm = _timeout > 0 and hasattr(signal, "setitimer")
    if use_alarm:
        signal.signal(signal.SIGALRM, _on_alarm)
    out: list[Result] = []
    for cid, params, seed in chunk:
        started = time.perf_counter()
        metrics: dict[str, float] | None = None
        status, error = "ok", None
        try:
            if use_alarm:
                signal.setitimer(signal.ITIMER_REAL, _timeout)
            metrics = {k: float(v) for k, v in _target(params, seed, **_options).items()}
        except _Timeout:
            status, error = "timeout", f"took longer than {_timeout:g} s"
        except Exception as exc:  # one bad configuration must not stop the sweep
            status, error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            if use_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0)
        out.append(Result(cid, params, metrics, status, error, (time.perf_counter() - started) * 1000))
    return out


def fingerprint(spec: Spec) -> str:
    """Identifies the experiment, not the file: grid, target, metric, seed, options."""
    blob = json.dumps(
        {"target": spec.target, "metric": spec.metric, "grid": spec.grid, "seed": spec.seed, "options": spec.options},
        sort_keys=True,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


@dataclass(slots=True)
class RunReport:
    total: int  # configurations in the grid
    already_done: int
    ran: int
    ok: int
    errors: int
    timeouts: int
    seconds: float
    interrupted: bool

    @property
    def rate(self) -> float:
        return self.ran / self.seconds if self.seconds > 0 else 0.0


def _chunks(tasks: Iterable[Task], size: int) -> Iterator[list[Task]]:
    it = iter(tasks)
    while chunk := list(itertools.islice(it, size)):
        yield chunk


class _Progress:
    def __init__(self, total: int, enabled: bool) -> None:
        self.total, self.enabled, self.done = total, enabled, 0
        self.started = self.last = time.monotonic()

    def add(self, n: int) -> None:
        self.done += n
        now = time.monotonic()
        if self.enabled and (now - self.last >= 0.5 or self.done == self.total):
            self.last = now
            rate = self.done / max(now - self.started, 1e-9)
            eta = (self.total - self.done) / rate if rate else 0
            sys.stderr.write(
                f"\r  {self.done:>9,} / {self.total:,}  {100 * self.done / max(self.total, 1):5.1f}%"
                f"  {rate:8,.0f} configs/s  eta {eta:5.0f} s "
            )
            sys.stderr.flush()

    def end(self) -> None:
        if self.enabled:
            sys.stderr.write("\n")


def run(
    spec: Spec,
    store: Store,
    workers: int | None = None,
    chunk: int = 64,
    limit: int | None = None,
    timeout: float = 0.0,
    retry_failed: bool = False,
    progress: bool | None = None,
) -> RunReport:
    store.check_spec(fingerprint(spec))
    done = store.done_ids(retry_failed=retry_failed)
    tasks: Iterable[Task] = (
        (cid, params, config_seed(spec.seed, cid))
        for params in spec.configs()
        if (cid := config_id(params)) not in done
    )
    todo = spec.size - len(done)
    if limit is not None:
        tasks = itertools.islice(tasks, limit)
        todo = min(todo, limit)
    workers = max(1, workers or os.cpu_count() or 1)
    bar = _Progress(todo, sys.stderr.isatty() if progress is None else progress)
    counts: Counter[str] = Counter()
    started = time.monotonic()
    interrupted = False

    def collect(results: list[Result]) -> None:
        store.write(results)
        counts.update(r.status for r in results)
        bar.add(len(results))

    try:
        if workers == 1:
            _init_worker(spec.target, spec.options, timeout, ignore_sigint=False)
            for c in _chunks(tasks, chunk):
                collect(_run_chunk(c))
        else:
            with cf.ProcessPoolExecutor(
                max_workers=workers, initializer=_init_worker, initargs=(spec.target, spec.options, timeout)
            ) as pool:
                inflight: set[cf.Future[list[Result]]] = set()
                try:
                    for c in _chunks(tasks, chunk):
                        inflight.add(pool.submit(_run_chunk, c))
                        if len(inflight) >= workers * 4:
                            finished, inflight = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)
                            for f in finished:
                                collect(f.result())
                    while inflight:
                        finished, inflight = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)
                        for f in finished:
                            collect(f.result())
                except KeyboardInterrupt:
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise
    except KeyboardInterrupt:
        interrupted = True
    finally:
        bar.end()

    return RunReport(
        total=spec.size,
        already_done=len(done),
        ran=sum(counts.values()),
        ok=counts["ok"],
        errors=counts["error"],
        timeouts=counts["timeout"],
        seconds=time.monotonic() - started,
        interrupted=interrupted,
    )
