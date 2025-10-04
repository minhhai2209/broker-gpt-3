from __future__ import annotations

"""Config and tuning I/O helpers for the Order Engine.
Implementation details are separated to keep order_engine.py focused.
"""

from pathlib import Path
import os
from typing import Dict, Any, Iterable, Tuple

import pandas as pd
from .schema import PolicyOverrides

# Local paths (resolved relative to repo root)
BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "out"
OUT_ORDERS_DIR = OUT_DIR / "orders"
OVERRIDE_SRC = BASE_DIR / "config" / "policy_overrides.json"


def _strip_json_comments(text: str) -> str:
    import re
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(^|\s)//.*$", "", text, flags=re.M)
    text = re.sub(r"(^|\s)#.*$", "", text, flags=re.M)
    return text


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge dictionaries with base taking precedence.
    overlay provides defaults; base overrides where keys exist.
    """
    from copy import deepcopy
    out = deepcopy(overlay)
    for k, v in base.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(v, out[k])
        else:
            out[k] = v
    return out


# Whitelisted override paths that Codex may tune at runtime.
# Only these keys from config/policy_overrides.json will override policy_default.json.
# Represented as tuples of nested keys, e.g., ("thresholds", "base_add").
ALLOWED_OVERRIDE_PATHS: set[Tuple[str, ...]] = {
    # Minimal knobs (AI generator)
    ("buy_budget_frac",),
    ("add_max",),
    ("new_max",),
    ("sector_bias",),
    ("ticker_bias",),
    # Calibrator surfaces (persist tuned values between runs)
    ("thresholds",),           # q_add, q_new, min_liq_norm, near_ceiling_pct, tp/sl multiples, etc.
    ("sizing",),               # cov_reg, dynamic_caps.*, default_stop_atr_mult
    ("pricing",),              # fill_prob.*
    ("orders_ui",),            # ttl_minutes.*, watchlist.*
    ("market_filter",),        # calibrated market guards
    ("regime_model",),         # intercept/threshold and components mean/std
}


def _filter_overrides(ov: Dict[str, Any]) -> Dict[str, Any]:
    """Filter an overrides dict to only allowed tunable keys.
    - Entire maps allowed for sector_bias/ticker_bias.
    - For nested keys, copy only whitelisted fields.
    """
    out: Dict[str, Any] = {}
    for path in ALLOWED_OVERRIDE_PATHS:
        node = ov
        ok = True
        for key in path:
            if not isinstance(node, dict) or key not in node:
                ok = False
                break
            if key is path[-1]:
                # Leaf: copy value
                pass
            node = node[key]
        if not ok:
            continue
        # Construct into output
        cur = out
        for idx, key in enumerate(path):
            is_leaf = (idx == len(path) - 1)
            if is_leaf:
                # Assign whole map if path length is 1 and value is dict and key denotes bias map
                src_parent = ov
                for k2 in path[:-1]:
                    src_parent = src_parent[k2]
                cur[key] = src_parent[key]
            else:
                cur = cur.setdefault(key, {})  # type: ignore[assignment]
    return out


def ensure_policy_override_file() -> Path:
    """Prepare runtime policy by merging a tunable base with calibrated defaults.

    Sources checked in order:
      1) POLICY_FILE env (absolute or repo‑relative)
      2) config/policy_overrides.json
      3) existing out/orders/policy_overrides.json (as last resort)

    If a sibling defaults file `policy_for_calibration.json` exists next to
    the chosen source, or at `config/policy_for_calibration.json`, it will
    be merged (defaults then overridden by base). Otherwise, the file is copied
    verbatim to preserve formatting.
    """
    dest = OUT_ORDERS_DIR / "policy_overrides.json"
    policy_path_env = os.environ.get("POLICY_FILE", "").strip()

    # 0) If a complete baseline exists (policy_default.json) and no POLICY_FILE override is provided,
    # merge it with a restricted subset of config/policy_overrides.json. If overrides are absent,
    # write the baseline as-is. This is the preferred modern path.
    default_baseline = BASE_DIR / "config" / "policy_default.json"
    if not policy_path_env and default_baseline.exists():
        import json
        OUT_ORDERS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            default_obj = json.loads(_strip_json_comments(default_baseline.read_text(encoding="utf-8")))
        except Exception as exc:
            raise SystemExit(f"Invalid JSON in baseline policy {default_baseline}: {exc}") from exc

        if OVERRIDE_SRC.exists():
            try:
                ov_obj = json.loads(_strip_json_comments(OVERRIDE_SRC.read_text(encoding="utf-8")))
            except Exception as exc:
                raise SystemExit(f"Invalid JSON in overrides {OVERRIDE_SRC}: {exc}") from exc
            ov_filtered = _filter_overrides(ov_obj)
            merged = _deep_merge(ov_filtered, default_obj)  # ov overrides default for allowed fields
        else:
            merged = default_obj
        dest.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Prepared runtime policy from baseline {default_baseline} with restricted overrides from {OVERRIDE_SRC if OVERRIDE_SRC.exists() else '<none>'} -> {dest}")
        return dest

    # 1) Legacy path retained for backward compatibility and tests:
    # Choose a base policy and optionally merge policy_for_calibration.json
    src_path: Path | None = None
    if policy_path_env:
        p = Path(policy_path_env)
        if not p.is_absolute():
            p = BASE_DIR / p
        if not p.exists():
            raise SystemExit(f"POLICY_FILE not found: {p}")
        src_path = p
    elif OVERRIDE_SRC.exists():
        src_path = OVERRIDE_SRC
    elif dest.exists():
        return dest
    else:
        raise SystemExit("Missing policy overrides. Provide config/policy_overrides.json or out/orders/policy_overrides.json")

    # Try to locate calibrated defaults: prefer sibling of src, else config/
    defaults_path = None
    # Prefer local (sibling) defaults, then repo config. Do NOT fall back to sample for runtime.
    cand1 = src_path.parent / "policy_for_calibration.json"
    cand2 = BASE_DIR / "config" / "policy_for_calibration.json"
    if cand1.exists():
        defaults_path = cand1
    elif cand2.exists():
        defaults_path = cand2

    # If no defaults, copy verbatim (preserve exact original formatting)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if defaults_path is None:
        dest.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Copied policy overrides from {src_path} -> {dest}")
        return dest

    # Merge defaults -> base (base wins), pretty‑print merged JSON
    import json
    try:
        base_obj = json.loads(_strip_json_comments(src_path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise SystemExit(f"Invalid JSON in base policy {src_path}: {exc}") from exc
    try:
        def_obj = json.loads(_strip_json_comments(defaults_path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise SystemExit(f"Invalid JSON in calibrated defaults {defaults_path}: {exc}") from exc

    merged = _deep_merge(base_obj, def_obj)
    dest.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Merged policy overrides: base {src_path} + defaults {defaults_path} -> {dest}")
    return dest


def suggest_tuning(session_summary: pd.DataFrame, sector_strength: pd.DataFrame) -> Dict[str, object]:
    """Parse and validate policy overrides using a single typed schema.
    Returns a plain dict suitable for the current engine.
    """
    import re

    ov_path = OUT_ORDERS_DIR / "policy_overrides.json"
    if not ov_path.exists():
        raise SystemExit("Missing out/orders/policy_overrides.json. Please generate a full override (no defaults).")
    raw = ov_path.read_text()
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    raw = re.sub(r"(^|\s)//.*$", "", raw, flags=re.M)
    raw = re.sub(r"(^|\s)#.*$", "", raw, flags=re.M)
    model = PolicyOverrides.model_validate_json(raw)
    return model.model_dump()
