from __future__ import annotations

from pathlib import Path

import pytest

from sweep.spec import Spec, load_spec


@pytest.fixture
def make_spec(tmp_path: Path):  # type: ignore[no-untyped-def]
    def make(target: str, grid: str, options: str = "", name: str = "t") -> Spec:
        path = tmp_path / f"{name}.toml"
        path.write_text(
            f'[sweep]\nname = "{name}"\ntarget = "{target}"\nmetric = "y"\nseed = 7\n\n'
            f"[options]\n{options}\n\n[grid]\n{grid}\n"
        )
        return load_spec(path)

    return make
