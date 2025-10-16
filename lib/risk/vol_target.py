from __future__ import annotations

import math


def ann_from_daily(std_daily: float) -> float:
    try:
        s = float(std_daily)
    except Exception:
        return 0.0
    if s <= 0.0:
        return 0.0
    return s * math.sqrt(252.0)


def scale_budget(realized_ann: float, target_ann: float, lo: float, hi: float) -> float:
    try:
        r = float(realized_ann)
        t = float(target_ann)
        lo_v = float(lo)
        hi_v = float(hi)
    except Exception:
        return 1.0
    if r <= 0.0 or t <= 0.0:
        return 1.0
    s = t / r
    if s < lo_v:
        return lo_v
    if s > hi_v:
        return hi_v
    return s

