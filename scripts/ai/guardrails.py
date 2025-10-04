from __future__ import annotations

import json
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple, Any

import math

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = BASE_DIR / 'config'
OUT_DIR = BASE_DIR / 'out'
STATE_PATH = OUT_DIR / 'orders' / 'ai_override_state.json'
DEFAULT_POLICY_PATH = CONFIG_DIR / 'policy_default.json'
METRICS_PATH = OUT_DIR / 'metrics.csv'
AUDIT_CSV_PATH = OUT_DIR / 'orders' / 'ai_overrides_audit.csv'
AUDIT_JSONL_PATH = OUT_DIR / 'orders' / 'ai_overrides_audit.jsonl'

# Guardrail constants
BUY_BUDGET_BOUNDS = (0.02, 0.18)
BUY_BUDGET_RATE_LIMIT = 0.04
BUY_BUDGET_TTL_DAYS = 3
# Weekly positive tilt cap relative to baseline (absolute cap, not cumulative)
WEEKLY_BUDGET_POS_CAP = 0.06  # e.g., allow up to +6pp over baseline within a week

SLOTS_MAX_CAP = 10
SLOTS_RATE_LIMIT = 2
SLOTS_TTL_DAYS = 3

BIAS_DECAY = 0.80
BIAS_TTL_DAYS = 5
BIAS_BOUNDS = (-0.20, 0.20)


class GuardrailError(SystemExit):
    pass


def _strip_json_comments(text: str) -> str:
    import re
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(^|\s)//.*$", "", text, flags=re.M)
    text = re.sub(r"(^|\s)#.*$", "", text, flags=re.M)
    return text


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise GuardrailError(f'Missing required file: {path}')
    text = _strip_json_comments(path.read_text(encoding='utf-8'))
    return json.loads(text)


def _try_load_metrics_count() -> int:
    if not METRICS_PATH.exists():
        return 0
    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(METRICS_PATH)
        if df is None or df.empty:
            return 0
        return int(len(df))
    except Exception:
        return 0


def _load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _save_state(state: Dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except Exception:
        return None


def _format_date(dt: date) -> str:
    return dt.isoformat()


def _clip(value: float, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    return float(max(lo, min(hi, value)))


def _enforce_scalar(
    new_value: Any,
    *,
    baseline: float,
    state_entry: Dict[str, Any] | None,
    bounds: Tuple[float, float],
    rate_limit: float,
    ttl_days: int,
    today: date,
) -> Tuple[float | None, Dict[str, Any] | None]:
    prev_value = baseline if not state_entry else float(state_entry.get('value', baseline))
    if new_value is not None:
        try:
            val = float(new_value)
        except Exception as exc:  # pragma: no cover - guard
            raise GuardrailError(f'Invalid numeric override value: {new_value}') from exc
        val = _clip(val, bounds)
        delta = val - prev_value
        if abs(delta) > rate_limit:
            val = prev_value + rate_limit * (1 if delta > 0 else -1)
            val = _clip(val, bounds)
        sanitized = float(round(val, 5))
        expiry = today + timedelta(days=ttl_days)
        next_state = {
            'value': sanitized,
            'last_updated': _format_date(today),
            'expiry': _format_date(expiry),
        }
        return sanitized, next_state

    if not state_entry:
        return None, None
    expiry = _parse_date(state_entry.get('expiry'))
    if expiry and today >= expiry:
        # TTL expired -> revert to baseline
        return None, None
    sanitized = float(round(state_entry.get('value', baseline), 5))
    next_state = dict(state_entry)
    return sanitized, next_state


def _max_slots_bound(metrics_count: int) -> int:
    if metrics_count <= 0:
        return 0
    return int(min(SLOTS_MAX_CAP, math.ceil(metrics_count / 6.0)))


def _apply_news_tilt(overrides: Dict[str, Any], *, buy_baseline: float, add_baseline: int, new_baseline: int) -> Tuple[Dict[str, Any], float, str]:
    """Map optional 'news_risk_tilt' in [-1..+1] to budget/slot deltas.
    Returns (mutated_overrides, tilt_value, tilt_note).
    """
    if overrides is None:
        return {}, 0.0, ''
    od = dict(overrides)
    tilt_raw = od.pop('news_risk_tilt', None)
    if tilt_raw is None:
        return od, 0.0, ''
    try:
        tilt = float(tilt_raw)
    except Exception as exc:  # pragma: no cover
        raise GuardrailError(f'invalid news_risk_tilt: {tilt_raw}') from exc
    tilt = _clip(tilt, (-1.0, 1.0))
    # Budget delta ±2pp NAV per unit tilt
    delta_budget = float(round(tilt * 0.02, 5))
    # Slot delta ±1 per unit tilt (rounded to nearest int)
    delta_slots = int(round(tilt * 1.0))
    note = f"tilt={tilt:+.2f} -> Δbudget={delta_budget:+.3f}, Δslots={delta_slots:+d}"
    # Compose with explicit values if provided; else create new entries
    if 'buy_budget_frac' in od and od['buy_budget_frac'] is not None:
        try:
            od['buy_budget_frac'] = float(od['buy_budget_frac']) + delta_budget
        except Exception as exc:  # pragma: no cover
            raise GuardrailError(f'invalid buy_budget_frac override: {od["buy_budget_frac"]}') from exc
    else:
        # No explicit value provided -> treat tilt as delta off baseline
        od['buy_budget_frac'] = float(buy_baseline) + delta_budget
    if 'add_max' in od and od['add_max'] is not None:
        try:
            od['add_max'] = int(round(float(od['add_max']))) + delta_slots
        except Exception as exc:  # pragma: no cover
            raise GuardrailError(f'invalid add_max override: {od["add_max"]}') from exc
    else:
        od['add_max'] = int(add_baseline) + delta_slots
    if 'new_max' in od and od['new_max'] is not None:
        try:
            od['new_max'] = int(round(float(od['new_max']))) + delta_slots
        except Exception as exc:  # pragma: no cover
            raise GuardrailError(f'invalid new_max override: {od["new_max"]}') from exc
    else:
        od['new_max'] = int(new_baseline) + delta_slots
    return od, float(tilt), note


def _guard_biases(
    new_bias: Dict[str, Any],
    *,
    state_bias: Dict[str, Any],
    today: date,
    ttl_days: int,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    result: Dict[str, float] = {}
    next_state: Dict[str, Any] = {}
    new_bias = new_bias or {}

    # Reaffirmed entries
    for key, raw_val in new_bias.items():
        try:
            val = float(raw_val)
        except Exception as exc:  # pragma: no cover
            raise GuardrailError(f'Invalid bias value for {key}: {raw_val}') from exc
        val = _clip(val, BIAS_BOUNDS)
        if abs(val) < 1e-4:
            continue
        result[key] = float(round(val, 5))
        next_state[key] = {
            'value': result[key],
            'last_updated': _format_date(today),
            'expiry': _format_date(today + timedelta(days=ttl_days)),
        }

    # Decay previous entries not reaffirmed
    for key, entry in (state_bias or {}).items():
        if key in result:
            continue
        expiry = _parse_date(entry.get('expiry'))
        if expiry and today >= expiry:
            continue
        val = float(entry.get('value', 0.0)) * BIAS_DECAY
        if abs(val) < 1e-3:
            continue
        val = _clip(val, BIAS_BOUNDS)
        result[key] = float(round(val, 5))
        next_state[key] = {
            'value': result[key],
            'last_updated': entry.get('last_updated', _format_date(today)),
            'expiry': entry.get('expiry', _format_date(today + timedelta(days=ttl_days))),
        }
    return result, next_state


def enforce_guardrails(
    overrides: Dict[str, Any],
    *,
    baseline: Dict[str, Any],
    metrics_count: int,
    state: Dict[str, Any],
    today: date,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # Kill-switch: if present in state, bypass overrides
    if (state or {}).get('kill_switch'):
        return {}, dict(state or {})

    raw_overrides = deepcopy(overrides) if overrides else {}
    # Extract rationale for audit; not persisted to policy file
    rationale_text = str(raw_overrides.pop('rationale', '') or '').strip()
    # Apply optional news tilt mapping into working copy
    working, tilt_value, tilt_note = _apply_news_tilt(
        raw_overrides,
        buy_baseline=float(baseline.get('buy_budget_frac', 0.10)),
        add_baseline=int(baseline.get('add_max', 0)),
        new_baseline=int(baseline.get('new_max', 0)),
    )

    sanitized = deepcopy(working)
    next_state = deepcopy(state) if state else {}

    # Ensure nested dicts exist in sanitized copy
    if 'sizing' not in sanitized and 'sizing' in overrides:
        sanitized['sizing'] = deepcopy(overrides['sizing'])

    baseline_slots_cap = _max_slots_bound(metrics_count)

    # --- buy_budget_frac ---
    buy_baseline = float(baseline.get('buy_budget_frac', 0.10))
    buy_state = (state or {}).get('buy_budget_frac')
    buy_override = working.get('buy_budget_frac')
    # If buy_override is expressed as delta (from tilt without explicit absolute), convert to absolute here
    if buy_override is not None and abs(float(buy_override)) <= 0.30 and abs(float(buy_override) - buy_baseline) > 0.30:
        # Heuristic: treat small magnitude values as deltas if baseline+value still within bounds but raw value not
        try:
            buy_override = float(buy_baseline) + float(buy_override)
        except Exception:
            pass
    # Weekly positive cap relative to baseline
    if isinstance(buy_override, (int, float)):
        cap_hi = float(buy_baseline) + float(WEEKLY_BUDGET_POS_CAP)
        try:
            buy_override = float(min(float(buy_override), cap_hi))
        except Exception:
            pass
    buy_val, buy_state_next = _enforce_scalar(
        buy_override,
        baseline=buy_baseline,
        state_entry=buy_state,
        bounds=BUY_BUDGET_BOUNDS,
        rate_limit=BUY_BUDGET_RATE_LIMIT,
        ttl_days=BUY_BUDGET_TTL_DAYS,
        today=today,
    )
    if buy_state_next:
        next_state['buy_budget_frac'] = buy_state_next
    elif 'buy_budget_frac' in next_state:
        next_state.pop('buy_budget_frac')

    if buy_val is None or abs(buy_val - buy_baseline) < 1e-6:
        sanitized.pop('buy_budget_frac', None)
    else:
        sanitized['buy_budget_frac'] = float(round(buy_val, 5))

    # --- add_max / new_max ---
    def _guard_slots(key: str) -> None:
        baseline_value = int(baseline.get(key, 0))
        state_entry = (state or {}).get(key)
        new_value = working.get(key)
        val, state_next = _enforce_scalar(
            new_value,
            baseline=float(baseline_value),
            state_entry=state_entry,
            bounds=(0.0, float(baseline_slots_cap if baseline_slots_cap > 0 else SLOTS_MAX_CAP)),
            rate_limit=float(SLOTS_RATE_LIMIT),
            ttl_days=SLOTS_TTL_DAYS,
            today=today,
        )
        if state_next:
            next_state[key] = state_next
        elif key in next_state:
            next_state.pop(key)

        if val is None or int(round(val)) == baseline_value:
            sanitized.pop(key, None)
            return
        sanitized[key] = int(max(0, min(int(round(val)), SLOTS_MAX_CAP)))

    _guard_slots('add_max')
    _guard_slots('new_max')

    # --- sector_bias / ticker_bias ---
    def _guard_bias(key: str) -> None:
        baseline_bias = baseline.get(key) or {}
        override_bias = overrides.get(key) or {}
        state_bias = (state or {}).get(key) or {}
        sanitized_bias, state_next = _guard_biases(
            override_bias,
            state_bias=state_bias,
            today=today,
            ttl_days=BIAS_TTL_DAYS,
        )
        if state_next:
            next_state[key] = state_next
        elif key in next_state:
            next_state.pop(key)

        if not sanitized_bias:
            sanitized.pop(key, None)
            return
        # Remove entries equal to baseline (which is typically empty)
        sanitized[key] = {k: float(round(v, 5)) for k, v in sanitized_bias.items()}

    _guard_bias('sector_bias')
    _guard_bias('ticker_bias')

    # Clean empty nested dicts for sizing/thresholds if any were removed
    if 'sizing' in sanitized and (not isinstance(sanitized['sizing'], dict) or not sanitized['sizing']):
        sanitized.pop('sizing', None)
    if 'thresholds' in sanitized and (not isinstance(sanitized['thresholds'], dict) or not sanitized['thresholds']):
        sanitized.pop('thresholds', None)

    # Attach ephemeral meta for audit to next_state (not persisted to policy)
    if rationale_text:
        next_state['_last_rationale'] = rationale_text
    if tilt_note:
        next_state['_last_tilt'] = tilt_note

    return sanitized, next_state


def _audit_write(row: Dict[str, Any]) -> None:
    """Append a structured audit row to CSV and JSONL outputs."""
    import csv
    AUDIT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    # CSV header if new file
    write_header = not AUDIT_CSV_PATH.exists()
    with AUDIT_CSV_PATH.open('a', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    # JSONL
    with AUDIT_JSONL_PATH.open('a', encoding='utf-8') as jf:
        jf.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_audit_row(
    *,
    baseline: Dict[str, Any],
    state_before: Dict[str, Any],
    sanitized: Dict[str, Any],
    overrides: Dict[str, Any],
    metrics_count: int,
    today: date,
) -> Dict[str, Any]:
    def _state_val(key: str, default: float) -> float:
        e = (state_before or {}).get(key) or {}
        try:
            return float(e.get('value', default))
        except Exception:
            return float(default)

    bb0 = float(baseline.get('buy_budget_frac', 0.0))
    add0 = int(baseline.get('add_max', 0))
    new0 = int(baseline.get('new_max', 0))
    bb_prev = _state_val('buy_budget_frac', bb0)
    add_prev = int(round(_state_val('add_max', float(add0))))
    new_prev = int(round(_state_val('new_max', float(new0))))
    bb_new = float(sanitized.get('buy_budget_frac', bb_prev))
    add_new = int(sanitized.get('add_max', add_prev))
    new_new = int(sanitized.get('new_max', new_prev))
    sec_bias = sanitized.get('sector_bias') or {}
    tkr_bias = sanitized.get('ticker_bias') or {}
    rationale = str((state_before or {}).get('_last_rationale') or overrides.get('rationale') or '').strip()
    news_tilt = overrides.get('news_risk_tilt')

    row = {
        'date': today.isoformat(),
        'ts': datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds'),
        'metrics_n': int(metrics_count),
        'buy_budget_baseline': float(round(bb0, 5)),
        'buy_budget_prev': float(round(bb_prev, 5)),
        'buy_budget_new': float(round(bb_new, 5)),
        'add_max_baseline': int(add0),
        'add_max_prev': int(add_prev),
        'add_max_new': int(add_new),
        'new_max_baseline': int(new0),
        'new_max_prev': int(new_prev),
        'new_max_new': int(new_new),
        'sector_bias_n': int(len(sec_bias)),
        'sector_bias_sum': float(round(sum(sec_bias.values()) if sec_bias else 0.0, 5)),
        'ticker_bias_n': int(len(tkr_bias)),
        'ticker_bias_sum': float(round(sum(tkr_bias.values()) if tkr_bias else 0.0, 5)),
        'news_risk_tilt': float(news_tilt) if isinstance(news_tilt, (int, float)) else None,
        'rationale': rationale,
        'kill_switch': bool((state_before or {}).get('kill_switch', False)),
        'source': 'generate_policy_overrides',
    }
    return row


def apply_guardrails(overrides: Dict[str, Any]) -> Dict[str, Any]:
    baseline = _load_json(DEFAULT_POLICY_PATH)
    state = _load_state()
    metrics_count = _try_load_metrics_count()
    today = datetime.now(timezone.utc).date()
    sanitized, next_state = enforce_guardrails(
        overrides,
        baseline=baseline,
        metrics_count=metrics_count,
        state=state,
        today=today,
    )
    _save_state(next_state)
    try:
        row = _build_audit_row(
            baseline=baseline,
            state_before=state,
            sanitized=sanitized,
            overrides=overrides,
            metrics_count=metrics_count,
            today=today,
        )
        _audit_write(row)
    except Exception:
        # Audit must never block guardrails
        pass
    return sanitized


__all__ = [
    'apply_guardrails',
    'enforce_guardrails',
    '_build_audit_row',
]
