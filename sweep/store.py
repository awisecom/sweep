"""Results on disk: one SQLite file per sweep.

Each configuration is one row keyed by its stable id. Writes are batched in
transactions and the file runs in WAL mode, so a crash or Ctrl-C loses at most
the batch in flight, and the next `run` picks up exactly where this one stopped.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    config_id   TEXT PRIMARY KEY,
    params      TEXT NOT NULL,
    metrics     TEXT,
    status      TEXT NOT NULL CHECK (status IN ('ok', 'error', 'timeout')),
    error       TEXT,
    ms          REAL NOT NULL,
    finished_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


@dataclass(frozen=True, slots=True)
class Result:
    config_id: str
    params: dict[str, Any]
    metrics: dict[str, float] | None
    status: str  # ok | error | timeout
    error: str | None
    ms: float


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def check_spec(self, fingerprint: str) -> None:
        """Refuse to mix results from two different specs in one file."""
        row = self.db.execute("SELECT value FROM meta WHERE key = 'spec'").fetchone()
        if row is None:
            with self.db:
                self.db.execute("INSERT INTO meta (key, value) VALUES ('spec', ?)", (fingerprint,))
        elif row[0] != fingerprint:
            raise RuntimeError(f"{self.path} holds results of a different spec; use another --db or delete it")

    def done_ids(self, retry_failed: bool = False) -> set[str]:
        query = "SELECT config_id FROM runs" + (" WHERE status = 'ok'" if retry_failed else "")
        return {r[0] for r in self.db.execute(query)}

    def write(self, results: Iterable[Result]) -> int:
        rows = [
            (
                r.config_id,
                json.dumps(r.params, sort_keys=True),
                None if r.metrics is None else json.dumps(r.metrics),
                r.status,
                r.error,
                r.ms,
                time.time(),
            )
            for r in results
        ]
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO runs (config_id, params, metrics, status, error, ms, finished_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def counts(self) -> dict[str, int]:
        return dict(self.db.execute("SELECT status, count(*) FROM runs GROUP BY status").fetchall())

    def results(self) -> Iterator[tuple[dict[str, Any], dict[str, float]]]:
        """(params, metrics) for every successful run, in a stable order."""
        cur = self.db.execute("SELECT params, metrics FROM runs WHERE status = 'ok' ORDER BY config_id")
        for params, metrics in cur:
            yield json.loads(params), json.loads(metrics)

    def failures(self, limit: int = 10) -> list[tuple[str, str, str]]:
        q = "SELECT config_id, status, error FROM runs WHERE status != 'ok' LIMIT ?"
        return [(a, b, c or "") for a, b, c in self.db.execute(q, (limit,))]
