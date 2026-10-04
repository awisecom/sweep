from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from sweep.cli import main

QUICK = str(Path(__file__).parent.parent / "examples" / "quick.toml")


def test_run_analyze_controlled_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = str(tmp_path / "quick.sqlite")
    assert main(["run", QUICK, "--db", db, "--workers", "2"]) == 0
    assert "ran 576" in capsys.readouterr().out

    assert main(["status", QUICK, "--db", db]) == 0
    assert "576 of 576" in capsys.readouterr().out

    assert main(["analyze", QUICK, "--db", db, "--json", "--svg", str(tmp_path / "v.svg")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["n"] == 576
    assert (
        abs(
            sum(e["share"] for e in report["effects"])
            + sum(p["share"] for p in report["pairs"])
            + report["higher_order"]
            - 1.0
        )
        < 1e-6
    )
    ET.parse(tmp_path / "v.svg")

    assert main(["controlled", QUICK, "--db", db, "--cases", "6", "--svg", str(tmp_path / "c.svg")]) == 0
    assert "on 6 unseen cases" in capsys.readouterr().out
    ET.parse(tmp_path / "c.svg")
