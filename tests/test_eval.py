"""Smoke test for Elo conversion (doesn't require stockfish)."""

from __future__ import annotations

from alphazero.eval import _elo_from_score


def test_elo_symmetry():
    e, _ = _elo_from_score(0.5, 10)
    assert abs(e) < 1e-6


def test_elo_magnitude():
    # 75% score ~ +191 elo, 25% ~ -191. Standard.
    e_high, _ = _elo_from_score(0.75, 100)
    e_low, _ = _elo_from_score(0.25, 100)
    assert abs(e_high - 190.8) < 1.0
    assert abs(e_high + e_low) < 1e-6


def test_elo_stderr_shrinks_with_n():
    _, se_small = _elo_from_score(0.6, 10)
    _, se_large = _elo_from_score(0.6, 1000)
    assert se_large < se_small
