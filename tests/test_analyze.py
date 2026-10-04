from __future__ import annotations

from pathlib import Path

import pytest

from sweep.analyze import analyze
from sweep.runner import run
from sweep.store import Store


def _analyzed(make_spec, tmp_path: Path, target: str, grid: str):  # type: ignore[no-untyped-def]
    spec = make_spec(target, grid)
    with Store(tmp_path / "a.sqlite") as store:
        run(spec, store, workers=1, progress=False)
        return analyze(spec, list(store.results()))


def test_an_additive_model_splits_exactly_into_main_effects(make_spec, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    # y = 10a + b with a, b in {1, 2, 3}: variance from a is 100x the variance from b
    result = _analyzed(make_spec, tmp_path, "targets:additive", "a = [1, 2, 3]\nb = [1, 2, 3]")
    shares = {e.factor: e.share for e in result.effects}
    assert shares["a"] == pytest.approx(100 / 101)
    assert shares["b"] == pytest.approx(1 / 101)
    assert result.pair_share == pytest.approx(0.0, abs=1e-12)
    assert result.dominant.factor == "a" and result.dominant.best == 3


def test_a_pure_interaction_has_no_main_effects(make_spec, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    # y = (a != b): each factor alone tells you nothing, the pair tells you everything
    result = _analyzed(make_spec, tmp_path, "targets:xor", "a = [0, 1]\nb = [0, 1]")
    assert result.main_share == pytest.approx(0.0, abs=1e-12)
    assert result.interactions[0][2] == pytest.approx(1.0)
    assert result.higher_order == pytest.approx(0.0, abs=1e-12)


def test_shares_never_add_up_to_more_than_one(make_spec, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    result = _analyzed(make_spec, tmp_path, "targets:seeded", "a = [1, 2, 3]\nb = [1, 2]\nc = [1, 2, 3, 4]")
    assert result.main_share + result.pair_share <= 1.0 + 1e-9
    assert result.balanced
