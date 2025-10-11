from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from scripts.aggregate_patches import PatchInfo, aggregate_policy, aggregate_to_runtime
from scripts.calibrators.ml_runtime import build_feature_panel


def _make_patch(source: str, payload: dict, *, hours: int = 1) -> PatchInfo:
    now = datetime.now(timezone.utc)
    meta = payload.setdefault("meta", {"source": source})
    if "ttl" not in meta:
        meta["ttl"] = (now + timedelta(hours=hours)).isoformat()
    return PatchInfo(path=Path(f"/tmp/{source}.json"), payload=payload, source=source, ttl=now + timedelta(hours=hours), valid=True)


def test_aggregate_policy_min_rules():
    base = {
        "buy_budget_frac": 0.1,
        "add_max": 5,
        "new_max": 3,
        "sector_bias": {"Energy": 0.0},
        "ticker_bias": {"AAA": 0.0},
    }
    patch_a = _make_patch(
        "tune",
        {
            "set": {"buy_budget_frac": 0.08},
            "limits": {"new_max": 2},
            "bias": {"sector_bias.Energy": 0.05},
        },
    )
    patch_b = _make_patch(
        "cal_ml",
        {
            "set": {"buy_budget_frac": 0.06},
            "limits": {"add_max": 4},
            "bias": {"ticker_bias.AAA": 0.1},
            "gate": {"AAA": {"p_succ": 0.7}},
        },
    )
    merged, runtime, winners = aggregate_policy(base, [patch_a, patch_b])
    assert merged["buy_budget_frac"] == 0.06
    assert merged["new_max"] == 2
    assert merged["add_max"] == 4
    assert merged["sector_bias"]["Energy"] == 0.05
    assert merged["ticker_bias"]["AAA"] == 0.1
    assert runtime["gate"]["AAA"]["p_succ"] == 0.7
    assert any(item["key"] == "buy_budget_frac" for item in winners)


def test_aggregate_respects_policy_lock(tmp_path, monkeypatch):
    import scripts.aggregate_patches as agg

    base_policy = {"buy_budget_frac": 0.1, "add_max": 5, "new_max": 3, "sector_bias": {}, "ticker_bias": {}}
    runtime_path = tmp_path / "policy_runtime.json"
    runtime_path.write_text(json.dumps({"buy_budget_frac": 0.2}), encoding="utf-8")
    baseline_path = tmp_path / "policy_overrides.json"
    baseline_path.write_text(json.dumps(base_policy), encoding="utf-8")
    lock_path = tmp_path / ".policy_lock"
    lock_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(agg, "OUT_ORDERS_DIR", tmp_path)
    monkeypatch.setattr(agg, "OUT_DEBUG_DIR", tmp_path)
    monkeypatch.setattr(agg, "RUNTIME_PATH", runtime_path)
    monkeypatch.setattr(agg, "MERGE_LOG_PATH", tmp_path / "policy_merge.log")
    monkeypatch.setattr(agg, "LOCK_PATH", lock_path)
    monkeypatch.setattr(agg, "PATCH_PATHS", tuple())

    result_path = aggregate_to_runtime()
    assert result_path == runtime_path
    data = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert data["buy_budget_frac"] == 0.2


def test_feature_panel_no_future_leakage():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    records = []
    prices = [100, 101, 102, 101, 103, 104]
    index_prices = [900, 905, 907, 903, 910, 912]
    for date, price, idx_price in zip(dates, prices, index_prices):
        records.append({"Date": date, "Ticker": "AAA", "Close": price})
        records.append({"Date": date, "Ticker": "VNINDEX", "Close": idx_price})
    history = pd.DataFrame(records)
    panel = build_feature_panel(history, horizon=3)
    latest_date = panel["date"].max()
    latest_rows = panel[panel["date"] == latest_date]
    assert latest_rows["excess_future"].isna().all()
