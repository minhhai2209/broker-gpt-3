from __future__ import annotations

"""Config and tuning I/O helpers for the Order Engine.
Implementation details are separated to keep order_engine.py focused.
"""

from pathlib import Path
from typing import Dict, Any, Iterable

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


# Note on overrides surface
# AI-generated overlays are now written verbatim by the tuners. The runtime merge
# keeps the behaviour simple: baseline defaults are layered with any available
# overlays without additional filtering in this module.


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
    # 0) If a complete baseline exists (policy_default.json),
    # deep‑merge it with zero or more overlay files (if present):
    #   - config/policy_nightly_overrides.json (output of nightly calibrations)
    #   - config/policy_ai_overrides.json (output of AI tuner)
    #   - config/policy_overrides.json (legacy/compat)
    # AI-generated overlays are written verbatim; calibrators may also write broader tuned sections.
    default_baseline = BASE_DIR / "config" / "policy_default.json"
    if default_baseline.exists():
        import json
        OUT_ORDERS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            default_obj = json.loads(_strip_json_comments(default_baseline.read_text(encoding="utf-8")))
        except Exception as exc:
            raise SystemExit(f"Invalid JSON in baseline policy {default_baseline}: {exc}") from exc

        merged = default_obj
        overlays = [
            BASE_DIR / "config" / "policy_nightly_overrides.json",
            BASE_DIR / "config" / "policy_ai_overrides.json",
            OVERRIDE_SRC,  # legacy unified overlay (tuner-managed)
            BASE_DIR / "config" / "policy_curated_overrides.json",  # manual long-term curated bias overlay
        ]
        for path in overlays:
            if path.exists():
                try:
                    ov_obj = json.loads(_strip_json_comments(path.read_text(encoding="utf-8")))
                except Exception as exc:
                    raise SystemExit(f"Invalid JSON in overlay {path}: {exc}") from exc
                merged = _deep_merge(ov_obj, merged)  # later overlays take precedence
        # Slim runtime cleanup — remove deprecated/legacy runtime keys
        def _cleanup_policy(obj: dict) -> dict:
            if not isinstance(obj, dict):
                return obj
            out = dict(obj)
            # 1) Remove calibration block
            out.pop('calibration', None)
            # 2) Remove thresholds_profiles
            out.pop('thresholds_profiles', None)
            # 3) Remove execution.filter_buy_limit_gt_market and execution.fill
            exec_conf = dict(out.get('execution') or {})
            if exec_conf:
                exec_conf.pop('filter_buy_limit_gt_market', None)
                exec_conf.pop('fill', None)
                out['execution'] = exec_conf
            # 4) Conditional: remove thresholds.tp_pct/sl_pct when fully ATR-dynamic
            # Keep tp_pct/sl_pct for ATR calibration even if engine uses ATR-dynamic
            th = dict(out.get('thresholds') or {})
            if th is not None:
                if 'tp_pct' not in th:
                    th['tp_pct'] = 0.0
                if 'sl_pct' not in th:
                    th['sl_pct'] = 0.0
                out['thresholds'] = th
            return out
        merged = _cleanup_policy(merged)
        # Tag as machine-generated snapshot
        try:
            from datetime import datetime, timezone
            meta = {
                "machine_generated": True,
                "generated_by": "broker-gpt runtime merge",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "note": "DO NOT EDIT: regenerated by tune/order runs; persist edits under config/ overlays",
            }
            merged = {"_meta": meta, **merged}
        except Exception:
            # best-effort; never block runtime if clock/env is odd
            merged = {"_meta": {"machine_generated": True}, **merged}
        dest.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Prepared runtime policy by merging baseline + overlays -> {dest}")
        return dest

    # 1) Legacy path retained for config-based workflows:
    # Choose a base policy and optionally merge policy_for_calibration.json
    src_path: Path | None = None
    if OVERRIDE_SRC.exists():
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
        # Copy verbatim to preserve exact formatting/content as required by tests.
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
    try:
        from datetime import datetime, timezone
        merged = {
            "_meta": {
                "machine_generated": True,
                "generated_by": "broker-gpt runtime merge",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "note": "DO NOT EDIT: regenerated by tune/order runs; persist edits under config/ overlays",
            },
            **merged,
        }
    except Exception:
        merged = {"_meta": {"machine_generated": True}, **merged}
    dest.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Merged policy overrides: base {src_path} + defaults {defaults_path} -> {dest}")
    return dest


def suggest_tuning(session_summary: pd.DataFrame, sector_strength: pd.DataFrame) -> Dict[str, object]:
    """Parse and validate policy overrides using a single typed schema.
    Returns a plain dict suitable for the current engine.
    """
    import re

    runtime_path = OUT_ORDERS_DIR / "policy_runtime.json"
    ov_path = runtime_path if runtime_path.exists() else OUT_ORDERS_DIR / "policy_overrides.json"
    if not ov_path.exists():
        raise SystemExit(
            "Missing out/orders/policy_overrides.json. Generate policy_overrides.json via ensure_policy_override_file()."
        )
    raw = ov_path.read_text()
    raw = _strip_json_comments(raw)
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    raw = re.sub(r"(^|\s)//.*$", "", raw, flags=re.M)
    raw = re.sub(r"(^|\s)#.*$", "", raw, flags=re.M)
    import json as _json

    try:
        parsed = _json.loads(raw)
    except Exception as exc:
        raise SystemExit(f"Invalid runtime policy JSON: {exc}") from exc

    runtime_overrides = parsed.pop("_runtime_overrides", None)
    model = PolicyOverrides.model_validate(parsed)
    data = model.model_dump()
    if runtime_overrides is not None:
        data["_runtime_overrides"] = runtime_overrides
    return data
