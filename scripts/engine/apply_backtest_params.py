from __future__ import annotations

"""
Apply backtest best_params.json to policy_default.json for overlapping keys.

This is intentionally conservative: only parameters that have a clear mapping
to live policy defaults are applied. Others are ignored with a note.

Supported mappings
  - buy_budget_frac -> buy_budget_frac
  - new_max -> new_max
  - add_max -> add_max
  - watchlist.min_priority -> orders_ui.watchlist.min_priority

Unsupported (examples, ignored): ttl_days, entry_price_offset_ticks,
limit_safety_spread_ticks — these are backtest-only knobs.
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


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(_strip_json_comments(p.read_text(encoding="utf-8")))


def _set(obj: Dict[str, Any], path: str, value: Any) -> None:
    cur = obj
    parts = path.split(".")
    for k in parts[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[parts[-1]] = value


MAPPING = {
    "buy_budget_frac": "buy_budget_frac",
    "new_max": "new_max",
    "add_max": "add_max",
    "watchlist.min_priority": "orders_ui.watchlist.min_priority",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-params", required=True, help="Path to summary/best_params.json from backtest.tune")
    ap.add_argument("--baseline", required=True, help="Path to baseline (config/policy_default.json)")
    args = ap.parse_args()
    best_p = Path(args.best_params)
    base_p = Path(args.baseline)
    if not best_p.exists():
        raise SystemExit(f"best_params.json not found: {best_p}")
    obj = json.loads(best_p.read_text(encoding="utf-8")) or {}
    params = (obj.get("params") or obj.get("best", {}).get("params")) or {}
    if not isinstance(params, dict) or not params:
        raise SystemExit("No params found in best_params.json")
    base = _load_json(base_p)
    applied: Dict[str, Any] = {}
    ignored: Dict[str, Any] = {}
    for k, v in params.items():
        dest = MAPPING.get(k)
        if dest is None:
            ignored[k] = v
            continue
        _set(base, dest, v)
        applied[k] = v
    base_p.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Applied {len(applied)} mapped params -> {base_p}")
    if ignored:
        print("Ignored keys (backtest-only):", ", ".join(sorted(ignored.keys())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

