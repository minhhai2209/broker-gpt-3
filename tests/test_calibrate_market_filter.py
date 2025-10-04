import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

import scripts.engine.calibrate_market_filter as cmf


class TestCalibrateMarketFilter(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.out = self.base / 'out'
        self.config_dir = self.base / 'config'
        self.orders_dir = self.out / 'orders'
        self.out.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        # Snapshot originals
        self._orig_base = cmf.BASE_DIR
        self._orig_out = cmf.OUT_DIR
        self._orig_orders = cmf.ORDERS_PATH
        self._orig_config = cmf.CONFIG_PATH
        cmf.BASE_DIR = self.base
        cmf.OUT_DIR = self.out
        cmf.ORDERS_PATH = self.orders_dir / 'policy_overrides.json'
        cmf.CONFIG_PATH = self.config_dir / 'policy_overrides.json'

    def tearDown(self):
        cmf.BASE_DIR = self._orig_base
        cmf.OUT_DIR = self._orig_out
        cmf.ORDERS_PATH = self._orig_orders
        cmf.CONFIG_PATH = self._orig_config
        self.tmp.cleanup()

    def _write_policy(self, extra_targets=None):
        targets = {
            'idx_drop_q': 0.10,
            'vol_ann_q': 0.90,
            'trend_floor_q': 0.15,
            'atr_soft_q': 0.80,
            'atr_hard_q': 0.95,
        }
        if extra_targets:
            targets.update(extra_targets)
        payload = {'calibration_targets': {'market_filter': targets}}
        cmf.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        cmf.CONFIG_PATH.write_text(json.dumps(payload), encoding='utf-8')

    def _write_history(self, include_high_low=True, rows=420):
        dates = pd.date_range('2020-01-01', periods=rows, freq='B')
        close = 1000 + (pd.Series(range(rows)) * 0.5)
        data = {
            'Date': dates.strftime('%Y-%m-%d'),
            'Ticker': ['VNINDEX'] * rows,
            'Close': close,
        }
        if include_high_low:
            data['High'] = close + 5
            data['Low'] = close - 5
        df = pd.DataFrame(data)
        self.out.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.out / 'prices_history.csv', index=False)

    def test_missing_high_low_raises(self):
        self._write_policy()
        self._write_history(include_high_low=False)
        with self.assertRaises(SystemExit) as ctx:
            cmf.calibrate(write=False)
        self.assertIn('missing High/Low', str(ctx.exception))

    def test_calibration_outputs_and_write(self):
        self._write_policy()
        self._write_history(include_high_low=True, rows=500)
        result = cmf.calibrate(write=True)
        expected_keys = {
            'risk_off_index_drop_pct',
            'idx_chg_smoothed_hard_drop',
            'vol_ann_hard_ceiling',
            'trend_norm_hard_floor',
            'index_atr_soft_pct',
            'index_atr_hard_pct',
        }
        self.assertTrue(expected_keys.issubset(result.keys()))
        self.assertGreater(result['index_atr_hard_pct'], result['index_atr_soft_pct'])
        config = json.loads(cmf.CONFIG_PATH.read_text(encoding='utf-8'))
        market_filter = config.get('market_filter', {})
        self.assertTrue(expected_keys.issubset(market_filter.keys()))
