from __future__ import annotations

import statistics
from typing import Iterable, Optional, Sequence, Any


def merge_budget(baseline: Optional[float], candidates: Sequence[float], ctx: Any) -> Optional[float]:
    """Context-aware merge of buy budgets.

    baseline: float or None
    candidates: list of floats (other sources)
    ctx.regime.risk_on_probability in [0,1] if available

    Rule:
      - p >= 0.70 -> max(all)
      - p <= 0.30 -> min(all)
      - else -> median(all)
    Returns None if no inputs.
    """
    values = []
    if baseline is not None:
        try:
            values.append(float(baseline))
        except Exception:
            pass
    for v in (candidates or []):
        try:
            values.append(float(v))
        except Exception:
            continue
    if not values:
        return baseline
    try:
        p = getattr(getattr(ctx, 'regime', None), 'risk_on_probability', 0.5)
        p = float(p)
    except Exception:
        p = 0.5
    if p >= 0.70:
        return max(values)
    if p <= 0.30:
        return min(values)
    return statistics.median(values)

