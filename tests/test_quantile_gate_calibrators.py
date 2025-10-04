from __future__ import annotations

"""Unit coverage for quantile-gate calibrators."""

from scripts.engine import calibrate_quantile_gates as qg
from scripts.engine import calibrate_thresholds as qt


def test_quantile_gate_keep_all_returns_zero():
    # Case A: pool of one, keep one -> no gating
    assert qg._solve_q(1, 1) == 0.0
    assert qt._target_q(1, 1) == 0.0


def test_quantile_gate_large_quota_keeps_full_pool():
    # Case B: quota greater than pool size -> keep everything
    assert qg._solve_q(3, 10) == 0.0
    assert qt._target_q(10, 3) == 0.0


def test_quantile_gate_standard_formula():
    # Case C: orthodox 1 - K/N behaviour without lower clamp
    assert qg._solve_q(5, 2) == 0.6
    assert qt._target_q(2, 5) == 0.6


def test_quantile_gate_zero_quota_drops_all():
    # Case D: zero target means drop entire pool
    assert qg._solve_q(5, 0) == 0.995
    assert qt._target_q(0, 5) == 0.995


def test_quantile_gate_empty_pool_drops_all():
    # Case E: empty pools should fail-safe to dropping everything
    assert qg._solve_q(0, 3) == 0.995
    assert qt._target_q(3, 0) == 0.995
