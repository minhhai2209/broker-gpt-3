from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from scripts.tuning.calibrators import calibrate_ticker_stops as mod


def test_calibrate_ticker_stops_writes_overrides(tmp_path: Path):
    out = tmp_path / 'out'
    orders = out / 'orders'
    out.mkdir()
    orders.mkdir()
    # Minimal policy to write back into
    pol_path = orders / 'policy_overrides.json'
    pol_path.write_text(json.dumps({"ticker_overrides": {}}, ensure_ascii=False, indent=2), encoding='utf-8')
    # Snapshot
    snap = pd.DataFrame({
        'Ticker': ['AAA','BBB','CCC'],
        'Price': [20.0, 10.0, 15.0],
    })
    snap.to_csv(out / 'snapshot.csv', index=False)
    # Presets (MA50)
    pre = pd.DataFrame({
        'Ticker': ['AAA','BBB','CCC'],
        'MA20': [19.0, 11.0, 16.0],
        'MA50': [19.5, 12.0, 16.0],
    })
    pre.to_csv(out / 'presets_all.csv', index=False)
    # Metrics (ATR14_Pct, MomRet_12_1)
    met = pd.DataFrame({
        'Ticker': ['AAA','BBB','CCC'],
        'ATR14_Pct': [0.010, 0.030, 0.050],
        'MomRet_12_1': [0.05, -0.02, 0.01],
    })
    met.to_csv(out / 'metrics.csv', index=False)

    overrides = mod.calibrate(
        write=True,
        snapshot_path=out / 'snapshot.csv',
        metrics_path=out / 'metrics.csv',
        presets_path=out / 'presets_all.csv',
        explicit_policy_path=pol_path,
    )
    assert isinstance(overrides, dict)
    saved = json.loads(pol_path.read_text(encoding='utf-8'))
    ovs = saved.get('ticker_overrides', {})
    assert 'AAA' in ovs and 'sl_atr_mult' in ovs['AAA']
    assert 'BBB' in ovs and 'sl_atr_mult' in ovs['BBB']
    assert 'CCC' in ovs and 'sl_atr_mult' in ovs['CCC']
