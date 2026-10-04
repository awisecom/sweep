from __future__ import annotations

from pathlib import Path

from sweep.runner import run
from sweep.store import Store


def test_results_do_not_depend_on_the_number_of_workers(make_spec, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    spec = make_spec("targets:seeded", "a = [1, 2, 3, 4, 5, 6, 7, 8]\nb = [1, 2, 3, 4, 5]")
    outputs = []
    for workers in (1, 3):
        with Store(tmp_path / f"w{workers}.sqlite") as store:
            report = run(spec, store, workers=workers, chunk=4, progress=False)
            assert report.ok == 40
            outputs.append(sorted((tuple(sorted(p.items())), m["y"]) for p, m in store.results()))
    assert outputs[0] == outputs[1]


def test_a_second_run_only_does_what_is_left(make_spec, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    spec = make_spec("targets:additive", "a = [1, 2, 3, 4]\nb = [1, 2, 3, 4, 5]")
    with Store(tmp_path / "r.sqlite") as store:
        first = run(spec, store, workers=1, limit=7, progress=False)
        second = run(spec, store, workers=2, chunk=3, progress=False)
        third = run(spec, store, workers=2, progress=False)
        assert (first.ran, second.ran, third.ran) == (7, 13, 0)
        assert second.already_done == 7
        assert store.counts() == {"ok": 20}


def test_errors_are_recorded_and_do_not_stop_the_sweep(make_spec, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    spec = make_spec("targets:flaky", "a = [1, 2, 3]\nb = [1, 2]")
    with Store(tmp_path / "f.sqlite") as store:
        report = run(spec, store, workers=2, chunk=1, progress=False)
        assert (report.ok, report.errors) == (4, 2)
        assert all("RuntimeError: boom" in err for _, _, err in store.failures())
        retried = run(spec, store, workers=1, retry_failed=True, progress=False)
        assert retried.ran == 2


def test_a_hanging_configuration_times_out(make_spec, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    spec = make_spec("targets:slow", "a = [1, 2]\nb = [1]")
    with Store(tmp_path / "s.sqlite") as store:
        report = run(spec, store, workers=1, timeout=0.2, progress=False)
        assert (report.ok, report.timeouts) == (1, 1)


def test_one_results_file_never_mixes_two_specs(make_spec, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    a = make_spec("targets:additive", "a = [1]\nb = [1]", name="one")
    b = make_spec("targets:additive", "a = [1, 2]\nb = [1]", name="two")
    with Store(tmp_path / "x.sqlite") as store:
        run(a, store, workers=1, progress=False)
        try:
            run(b, store, workers=1, progress=False)
        except RuntimeError as exc:
            assert "different spec" in str(exc)
        else:
            raise AssertionError("mixed two specs in one file")
