from __future__ import annotations

from datetime import date
from typing import Tuple, Dict, Any


def should_cooldown(account, policy) -> Tuple[bool, Dict[str, Any]]:
    """Decide if kill-switch cooldown should be engaged.

    Expects:
      - account.metrics.peak_to_trough_drawdown_pct_rolling(days)
      - account.metrics.stopouts_in_last_n_days(days)
      - account.state.cooldown_until (optional)
      - policy has .enable, .dd_hard_pct, .sl_streak_n, .window_days, .cooldown_days, .actions
    Returns (engage: bool, actions: dict)
    """
    try:
        if not getattr(policy, 'enable', False):
            return False, {}
    except Exception:
        return False, {}
    try:
        window = int(getattr(policy, 'window_days', 3) or 3)
    except Exception:
        window = 3
    try:
        dd_hard = float(getattr(policy, 'dd_hard_pct', 0.0) or 0.0)
    except Exception:
        dd_hard = 0.0
    try:
        sl_streak_n = int(getattr(policy, 'sl_streak_n', 0) or 0)
    except Exception:
        sl_streak_n = 0

    dd = 0.0
    sl_streak = 0
    try:
        dd = float(account.metrics.peak_to_trough_drawdown_pct_rolling(days=window))
    except Exception:
        dd = 0.0
    try:
        sl_streak = int(account.metrics.stopouts_in_last_n_days(window))
    except Exception:
        sl_streak = 0
    try:
        cooldown_until = getattr(account.state, 'cooldown_until', None)
    except Exception:
        cooldown_until = None

    if dd_hard > 0.0 and dd >= dd_hard:
        return True, dict(getattr(policy, 'actions', {}) or {})
    if sl_streak_n > 0 and sl_streak >= sl_streak_n:
        return True, dict(getattr(policy, 'actions', {}) or {})
    if cooldown_until is not None:
        try:
            if isinstance(cooldown_until, date) and date.today() <= cooldown_until:
                return True, dict(getattr(policy, 'actions', {}) or {})
        except Exception:
            pass
    return False, {}

