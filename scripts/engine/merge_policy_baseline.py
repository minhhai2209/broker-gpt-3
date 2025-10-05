from __future__ import annotations

"""
Selective deep-merge: update baseline policy_default.json with tuned keys
from a runtime policy snapshot (out/orders/policy_overrides.json).

Why selective? We avoid persisting ephemeral/UI state while committing
calibration outputs (thresholds/sizing/regime/pricing.fill_prob/market_filter,
and a few orders_ui knobs) that are intended to become new defaults.

Usage
  python -m scripts.engine.merge_policy_baseline \
    --tuned out/orders/policy_overrides.json \
    --baseline config/policy_default.json
"""

from pathlib import Path
import argparse
import json
import re
from typing import Any, Dict


def _strip_json_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(^|\s)//.*$", "", text, flags=re.M)
    text = re.sub(r"(^|\s)#.*$", "", text, flags=re.M)
    return text


def _load_json(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    return json.loads(_strip_json_comments(raw))


def _set_path(obj: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = obj
    for key in parts[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _maybe_get(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _merge_selected(tuned: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)

    # thresholds
    for k in (
        "q_add",
        "q_new",
        "min_liq_norm",
        "near_ceiling_pct",
        "tp_atr_mult",
        "sl_atr_mult",
        "tp_floor_pct",
        "sl_floor_pct",
    ):
        v = _maybe_get(tuned, f"thresholds.{k}")
        if v is not None:
            _set_path(out, f"thresholds.{k}", v)

    # sizing core
    for k in (
        "cov_reg",
        "softmax_tau",
        "default_stop_atr_mult",
        "add_share",
        "new_share",
    ):
        v = _maybe_get(tuned, f"sizing.{k}")
        if v is not None:
            _set_path(out, f"sizing.{k}", v)

    # sizing.dynamic_caps
    for k in ("pos_min", "pos_max", "sector_min", "sector_max"):
        v = _maybe_get(tuned, f"sizing.dynamic_caps.{k}")
        if v is not None:
            _set_path(out, f"sizing.dynamic_caps.{k}", v)

    # regime_model core
    for k in ("intercept", "threshold"):
        v = _maybe_get(tuned, f"regime_model.{k}")
        if v is not None:
            _set_path(out, f"regime_model.{k}", v)

    comps = _maybe_get(tuned, "regime_model.components")
    if isinstance(comps, dict):
        # copy mean/std only; do not alter weights here
        dst = out.setdefault("regime_model", {}).setdefault("components", {})
        for name, conf in comps.items():
            if not isinstance(conf, dict):
                continue
            dst_conf = dict(dst.get(name) or {})
            if "mean" in conf:
                dst_conf["mean"] = conf["mean"]
            if "std" in conf:
                dst_conf["std"] = conf["std"]
            if dst_conf:
                dst[name] = dst_conf
        out["regime_model"]["components"] = dst

    # market_filter
    for k in (
        "risk_off_index_drop_pct",
        "idx_chg_smoothed_hard_drop",
        "vol_ann_hard_ceiling",
        "trend_norm_hard_floor",
        "index_atr_soft_pct",
        "index_atr_hard_pct",
        "market_score_soft_floor",
        "market_score_hard_floor",
        "risk_off_drawdown_floor",
        "leader_min_rsi",
        "leader_min_mom_norm",
        "risk_off_breadth_floor",
    ):
        v = _maybe_get(tuned, f"market_filter.{k}")
        if v is not None:
            _set_path(out, f"market_filter.{k}", v)

    # pricing.fill_prob
    for k in ("base", "cross", "near_ceiling", "min", "decay_scale_min_ticks"):
        v = _maybe_get(tuned, f"pricing.fill_prob.{k}")
        if v is not None:
            _set_path(out, f"pricing.fill_prob.{k}", v)

    # orders_ui.watchlist
    for k in ("min_priority", "micro_window"):
        v = _maybe_get(tuned, f"orders_ui.watchlist.{k}")
        if v is not None:
            _set_path(out, f"orders_ui.watchlist.{k}", v)

    # orders_ui.ttl_minutes (exclude bucket state/diagnostics)
    for k in ("base", "soft", "hard"):
        v = _maybe_get(tuned, f"orders_ui.ttl_minutes.{k}")
        if v is not None:
            _set_path(out, f"orders_ui.ttl_minutes.{k}", v)

    # Optional: bucket minutes are stable templates; safe to carry over
    bucket = _maybe_get(tuned, "orders_ui.ttl_bucket_minutes")
    if isinstance(bucket, dict) and bucket:
        _set_path(out, "orders_ui.ttl_bucket_minutes", bucket)

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuned", required=True, help="Path to tuned runtime policy (out/orders/policy_overrides.json)")
    ap.add_argument("--baseline", required=True, help="Path to baseline (config/policy_default.json)")
    args = ap.parse_args()
    tuned_p = Path(args.tuned)
    base_p = Path(args.baseline)
    if not tuned_p.exists():
        raise SystemExit(f"Tuned policy not found: {tuned_p}")
    if not base_p.exists():
        raise SystemExit(f"Baseline not found: {base_p}")
    tuned = _load_json(tuned_p)
    base = _load_json(base_p)
    merged = _merge_selected(tuned, base)
    base_p.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Updated baseline defaults from tuned snapshot -> {base_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

