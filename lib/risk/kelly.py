from __future__ import annotations

"""Kelly-lite sizing helper.

Implements a conservative Kelly fraction suitable for tranche scaling.

- p: success probability in [0,1]
- R: expected reward-to-risk ratio (TP% / SL%); must be >0
- f_max: hard clamp on fraction (default 0.02 i.e., +2% NAV per name equivalent)

Returns a non-negative fraction in [0, f_max]. For invalid inputs returns 0.0.
"""

def kelly_fraction(p: float, R: float, f_max: float = 0.02) -> float:
    if R is None or p is None:
        return 0.0
    try:
        Rv = float(R)
        pv = float(p)
    except Exception:
        return 0.0
    if Rv <= 0.0 or not (0.0 <= pv <= 1.0):
        return 0.0
    f = (pv * Rv - (1.0 - pv)) / Rv
    if f < 0.0:
        f = 0.0
    try:
        f_max_v = float(f_max)
    except Exception:
        f_max_v = 0.02
    if f_max_v <= 0.0:
        f_max_v = 0.0
    return f if f <= f_max_v else f_max_v

