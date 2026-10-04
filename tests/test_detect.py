from __future__ import annotations

from sweep.demo.detect import detect, evaluate, make_signal, match, scores

BEST = {
    "statistic": "meanshift",
    "normalisation": "global",
    "confirm": 1,
    "window": 20,
    "threshold": 5.0,
    "smoothing": "ema",
    "debounce": 0,
    "rearm": 0.5,
    "decimate": 3,
}


def test_signals_are_reproducible_and_well_formed() -> None:
    a, b = make_signal(5), make_signal(5)
    assert a == b
    assert 3 <= len(a.changes) <= 6
    assert all(y - x >= 60 for x, y in zip(a.changes, a.changes[1:], strict=False))
    assert len(a.values) == 600


def test_matching_is_one_to_one_within_the_tolerance() -> None:
    assert match([100, 104, 300], [102, 200], tolerance=8) == (1, 2, 1)
    assert match([], [50], tolerance=8) == (0, 0, 1)


def test_detection_finds_a_clean_step() -> None:
    s = [0.0] * 50 + [10.0] * 3 + [0.0] * 50
    assert detect(s, list(range(len(s))), threshold=5.0, confirm=1, debounce=0, rearm=0.5, peak=2) == [50]


def test_scores_are_cached_per_stage() -> None:
    first = scores(0, 600, "none", 1, "meanshift", "mad", 20)
    assert scores(0, 600, "none", 1, "meanshift", "mad", 20) is first


def test_the_best_configuration_from_the_sweep_holds_up() -> None:
    train = evaluate(BEST, 0, cases=range(12))
    unseen = evaluate(BEST, 0, cases=range(10_000, 10_086))
    assert train["f1"] > 0.95
    assert unseen["f1"] > 0.85  # a small generalisation gap, not a collapse
    assert all(0.0 <= v <= 1.0 for v in (*train.values(), *unseen.values()))
