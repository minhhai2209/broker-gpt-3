import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import scripts.engine.calibrate_market_filter as cmf
from scripts.engine.schema import MarketFilter


class TestFailFastMessages(unittest.TestCase):
    def test_missing_high_low_for_atr(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            cmf.BASE_DIR = base
            cmf.OUT_DIR = base / 'out'
            cmf.ORDERS_PATH = cmf.OUT_DIR / 'orders' / 'policy_overrides.json'
            cmf.CONFIG_PATH = base / 'config' / 'policy_overrides.json'
            # Seed history without High/Low
            dates = pd.date_range('2023-01-02', periods=420, freq='B')
            df = pd.DataFrame({'Date': dates.strftime('%Y-%m-%d'), 'Ticker': ['VNINDEX'] * len(dates), 'Close': [1000.0 + i for i in range(len(dates))]})
            cmf.OUT_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(cmf.OUT_DIR / 'prices_history.csv', index=False)
            # Seed policy with targets
            pol = {
                'calibration_targets': {
                    'market_filter': {'idx_drop_q': 0.1, 'vol_ann_q': 0.95, 'trend_floor_q': 0.1, 'atr_soft_q': 0.8, 'atr_hard_q': 0.95}
                }
            }
            (cmf.OUT_DIR / 'orders').mkdir(parents=True, exist_ok=True)
            (cmf.OUT_DIR / 'orders' / 'policy_overrides.json').write_text(json.dumps(pol), encoding='utf-8')
            with self.assertRaises(SystemExit) as ctx:
                cmf.calibrate(write=False)
            self.assertIn('missing High/Low', str(ctx.exception))

    def test_invalid_schema_ranges_raise(self):
        # Invalid values should raise validation errors
        with self.assertRaises(Exception):
            MarketFilter(  # type: ignore[arg-type]
                risk_off_index_drop_pct=None,
                risk_off_trend_floor=0.0,
                risk_off_breadth_floor=0.4,
                market_score_soft_floor=0.3,
                market_score_hard_floor=0.35,  # soft<hard invalid
                leader_min_rsi=55.0,
                leader_min_mom_norm=0.6,
                leader_require_ma20=1,
                leader_require_ma50=1,
                leader_max=2,
                risk_off_drawdown_floor=0.2,
                index_atr_soft_pct=0.8,
                index_atr_hard_pct=0.7,  # soft>hard invalid
                guard_new_scale_cap=1.2,  # out of [0,1]
                atr_soft_scale_cap=-0.1,  # out of [0,1]
                severe_drop_mult=0.0,     # must be >0
            )
