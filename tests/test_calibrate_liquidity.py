import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.tuning.calibrators import calibrate_liquidity as cl


class TestCalibrateLiquidity(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.out = self.base / 'out'
        self.config_dir = self.base / 'config'
        self.orders_dir = self.out / 'orders'
        self.out.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.orders_dir.mkdir(parents=True, exist_ok=True)

        self._orig_base = cl.BASE_DIR
        self._orig_out = cl.OUT_DIR
        self._orig_orders = cl.ORDERS_PATH
        self._orig_config = cl.CONFIG_PATH
        self._orig_defaults = cl.DEFAULTS_PATH

        cl.BASE_DIR = self.base
        cl.OUT_DIR = self.out
        cl.ORDERS_PATH = self.orders_dir / 'policy_overrides.json'
        cl.CONFIG_PATH = self.config_dir / 'policy_overrides.json'
        cl.DEFAULTS_PATH = self.config_dir / 'policy_default.json'

        self._write_baseline()
        self._write_nav()
        self._write_metrics()

    def tearDown(self):
        cl.BASE_DIR = self._orig_base
        cl.OUT_DIR = self._orig_out
        cl.ORDERS_PATH = self._orig_orders
        cl.CONFIG_PATH = self._orig_config
        cl.DEFAULTS_PATH = self._orig_defaults
        self.tmp.cleanup()

    def _write_baseline(self, min_liq: float = 0.31):
        payload = {'thresholds': {'min_liq_norm': min_liq}}
        cl.DEFAULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        cl.DEFAULTS_PATH.write_text(json.dumps(payload), encoding='utf-8')

    def _write_policy(self, *, new_max: int, min_liq: float | None = None, include_thresholds: bool = True):
        thresholds = {}
        if include_thresholds and min_liq is not None:
            thresholds['min_liq_norm'] = min_liq
        payload = {
            'buy_budget_frac': 0.10,
            'new_max': new_max,
            'sizing': {'new_share': 0.5},
            'calibration_targets': {'liquidity': {'adtv_multiple': 20}},
        }
        if thresholds:
            payload['thresholds'] = thresholds
        cl.CONFIG_PATH.write_text(json.dumps(payload), encoding='utf-8')

    def _write_nav(self):
        df = pd.DataFrame([
            {'TotalMarket': 2_000_000_000, 'TotalCost': 1_900_000_000}
        ])
        df.to_csv(self.out / 'portfolio_pnl_summary.csv', index=False)

    def _write_metrics(self):
        df = pd.DataFrame({'AvgTurnover20D_k': [5_000, 12_000, 20_000, 35_000]})
        df.to_csv(self.out / 'metrics.csv', index=False)

    def test_new_max_zero_uses_existing_threshold(self):
        self._write_policy(new_max=0, min_liq=0.44)
        result = cl.calibrate(write=False)
        self.assertEqual(result, 0.44)

    def test_new_max_zero_with_existing_threshold_writes_policy(self):
        self._write_policy(new_max=0, min_liq=0.44)
        result = cl.calibrate(write=True)
        self.assertEqual(result, 0.44)
        config = json.loads(cl.CONFIG_PATH.read_text(encoding='utf-8'))
        thresholds = config.get('thresholds', {})
        self.assertAlmostEqual(float(thresholds['min_liq_norm']), 0.44)

    def test_new_max_zero_without_threshold_falls_back_to_baseline(self):
        self._write_policy(new_max=0, include_thresholds=False)
        result = cl.calibrate(write=True)
        self.assertAlmostEqual(result, 0.31)
        config = json.loads(cl.CONFIG_PATH.read_text(encoding='utf-8'))
        thresholds = config.get('thresholds', {})
        self.assertAlmostEqual(float(thresholds['min_liq_norm']), 0.31)

    def test_new_max_zero_prefers_orders_runtime_when_present(self):
        payload = {
            'buy_budget_frac': 0.10,
            'new_max': 0,
            'sizing': {'new_share': 0.5},
            'calibration_targets': {'liquidity': {'adtv_multiple': 20}},
            'thresholds': {'min_liq_norm': 0.52},
        }
        cl.ORDERS_PATH.write_text(json.dumps(payload), encoding='utf-8')
        result = cl.calibrate(write=True)
        self.assertAlmostEqual(result, 0.52)
        runtime = json.loads(cl.ORDERS_PATH.read_text(encoding='utf-8'))
        thresholds = runtime.get('thresholds', {})
        self.assertAlmostEqual(float(thresholds['min_liq_norm']), 0.52)


if __name__ == '__main__':
    unittest.main()
