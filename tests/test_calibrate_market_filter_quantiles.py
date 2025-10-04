import json
import math
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import scripts.engine.calibrate_market_filter as cmf


class TestCalibrateMarketFilterQuantiles(unittest.TestCase):
    def _seed_history(self, out_dir: Path, days: int = 420) -> None:
        dates = pd.bdate_range(datetime(2022, 1, 3), periods=days)
        base = 1000.0
        # Create a gentle uptrend with varying daily amplitude to vary ATR%
        close = np.array([base + i * 1.2 for i in range(days)], dtype=float)
        amp = 0.008 + 0.006 * np.sin(np.linspace(0, 10, days))  # 0.8%..1.4%
        high = close * (1.0 + amp)
        low = close * (1.0 - amp)
        df = pd.DataFrame({
            'Date': dates.strftime('%Y-%m-%d'),
            'Ticker': ['VNINDEX'] * days,
            'Close': close,
            'High': high,
            'Low': low,
        })
        (out_dir / 'prices_history.csv').parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / 'prices_history.csv', index=False)

    def _seed_policy(self, orders_dir: Path, include_targets: bool = True, include_ms: bool = False) -> None:
        # Minimal policy with regime_model components and optional calibration targets
        pol = {
            'regime_model': {
                'intercept': 0.0,
                'threshold': 0.5,
                'components': {
                    'trend': {'mean': 0.5, 'std': 0.2, 'weight': 1.0},
                    'breadth': {'mean': 0.5, 'std': 0.2, 'weight': 1.0},
                    'index_return': {'mean': 0.0, 'std': 0.6, 'weight': 0.5},
                    'volatility': {'mean': 0.5, 'std': 0.2, 'weight': 0.5},
                },
            }
        }
        if include_targets:
            mf = {
                'idx_drop_q': 0.10,
                'vol_ann_q': 0.95,
                'trend_floor_q': 0.10,
                'atr_soft_q': 0.80,
                'atr_hard_q': 0.95,
            }
            if include_ms:
                mf.update({'ms_soft_q': 0.60, 'ms_hard_q': 0.35})
            pol['calibration_targets'] = {'market_filter': mf}
        orders_dir.mkdir(parents=True, exist_ok=True)
        (orders_dir / 'policy_overrides.json').write_text(json.dumps(pol), encoding='utf-8')

    def test_requires_targets(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            # Patch module paths
            cmf.BASE_DIR = base
            cmf.OUT_DIR = base / 'out'
            cmf.ORDERS_PATH = cmf.OUT_DIR / 'orders' / 'policy_overrides.json'
            cmf.CONFIG_PATH = base / 'config' / 'policy_overrides.json'
            self._seed_history(cmf.OUT_DIR, days=420)
            self._seed_policy(cmf.OUT_DIR / 'orders', include_targets=False)
            with self.assertRaises(SystemExit) as ctx:
                cmf.calibrate(write=False)
            self.assertIn('Missing calibration_targets.market_filter', str(ctx.exception))

    def test_atr_quantiles_on_percentile(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            cmf.BASE_DIR = base
            cmf.OUT_DIR = base / 'out'
            cmf.ORDERS_PATH = cmf.OUT_DIR / 'orders' / 'policy_overrides.json'
            cmf.CONFIG_PATH = base / 'config' / 'policy_overrides.json'
            self._seed_history(cmf.OUT_DIR, days=500)
            self._seed_policy(cmf.OUT_DIR / 'orders', include_targets=True)
            out = cmf.calibrate(write=False)
            soft = float(out['index_atr_soft_pct'])
            hard = float(out['index_atr_hard_pct'])
            self.assertTrue(0.0 <= soft <= 1.0)
            self.assertTrue(0.0 <= hard <= 1.0)
            self.assertLess(soft, hard)
            # sanity: expect roughly around targets, allow wide tolerance
            self.assertGreater(soft, 0.6)
            self.assertGreaterEqual(hard, 0.90)

    def test_drawdown_and_trend_vol_thresholds(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            cmf.BASE_DIR = base
            cmf.OUT_DIR = base / 'out'
            cmf.ORDERS_PATH = cmf.OUT_DIR / 'orders' / 'policy_overrides.json'
            cmf.CONFIG_PATH = base / 'config' / 'policy_overrides.json'
            self._seed_history(cmf.OUT_DIR, days=450)
            self._seed_policy(cmf.OUT_DIR / 'orders', include_targets=True)
            out = cmf.calibrate(write=False)
            self.assertIn('risk_off_index_drop_pct', out)
            self.assertIn('vol_ann_hard_ceiling', out)
            self.assertIn('trend_norm_hard_floor', out)
            self.assertGreater(out['risk_off_index_drop_pct'], 0.0)
            self.assertGreater(out['vol_ann_hard_ceiling'], 0.0)
            self.assertTrue(-1.0 <= out['trend_norm_hard_floor'] <= 1.0)

    def test_market_score_floors_optional(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            cmf.BASE_DIR = base
            cmf.OUT_DIR = base / 'out'
            cmf.ORDERS_PATH = cmf.OUT_DIR / 'orders' / 'policy_overrides.json'
            cmf.CONFIG_PATH = base / 'config' / 'policy_overrides.json'
            self._seed_history(cmf.OUT_DIR, days=420)
            # Without ms_* targets -> no floors
            self._seed_policy(cmf.OUT_DIR / 'orders', include_targets=True, include_ms=False)
            out = cmf.calibrate(write=False)
            self.assertNotIn('market_score_soft_floor', out)
            self.assertNotIn('market_score_hard_floor', out)
            # With ms_* targets -> floors present
            self._seed_policy(cmf.OUT_DIR / 'orders', include_targets=True, include_ms=True)
            out2 = cmf.calibrate(write=False)
            self.assertIn('market_score_soft_floor', out2)
            self.assertIn('market_score_hard_floor', out2)

