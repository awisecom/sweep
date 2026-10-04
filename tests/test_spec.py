from __future__ import annotations

from pathlib import Path

import pytest

from sweep.spec import config_id, config_seed, load_spec, load_target

EXAMPLES = Path(__file__).parent.parent / "examples"


def test_the_example_grid_has_41472_configurations() -> None:
    spec = load_spec(EXAMPLES / "detect.toml")
    assert spec.size == 41_472
    assert spec.size == sum(1 for _ in spec.configs())
    assert len(spec.grid["statistic"]) * len(spec.grid["normalisation"]) * len(spec.grid["confirm"]) == 24


def test_configurations_come_in_a_fixed_order() -> None:
    spec = load_spec(EXAMPLES / "quick.toml")
    first = [config_id(p) for p in spec.configs()]
    again = [config_id(p) for p in spec.configs()]
    assert first == again
    assert len(set(first)) == spec.size


def test_config_ids_do_not_depend_on_key_order() -> None:
    assert config_id({"a": 1, "b": "x"}) == config_id({"b": "x", "a": 1})
    assert config_id({"a": 1}) != config_id({"a": 1.5})
    assert config_id({"window": 20, "threshold": 3.0}) == "15e554912fac0dc9"  # stable across machines


def test_seeds_are_deterministic_and_spread_out() -> None:
    ids = [config_id({"a": i}) for i in range(1000)]
    seeds = [config_seed(7, cid) for cid in ids]
    assert seeds == [config_seed(7, cid) for cid in ids]
    assert len(set(seeds)) == 1000


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('[sweep]\nname="x"\ntarget="m:f"\nmetric="y"\n', "[grid] is empty"),
        ('[sweep]\nname="x"\ntarget="m:f"\nmetric="y"\n[grid]\na=[]\n', "non-empty list"),
        ('[sweep]\nname="x"\ntarget="m:f"\nmetric="y"\n[grid]\na=[1, 1]\n', "duplicate"),
        ('[sweep]\nname="x"\nmetric="y"\n[grid]\na=[1]\n', "needs 'target'"),
    ],
)
def test_bad_specs_are_rejected_with_a_reason(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        load_spec(path)


def test_targets_are_resolved_by_name() -> None:
    assert load_target("sweep.demo.detect:evaluate").__name__ == "evaluate"
    with pytest.raises(ValueError):
        load_target("sweep.demo.detect")
