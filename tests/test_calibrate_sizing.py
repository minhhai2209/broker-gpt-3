import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.tuning.calibrators import calibrate_sizing as cs


class TestCalibrateSizing(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.out = self.base / 'out'
        self.config_dir = self.base / 'config'
        self.orders_dir = self.out / 'orders'
        self.out.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._orig_base = cs.BASE_DIR
        self._orig_out = cs.OUT_DIR
        self._orig_orders = cs.ORDERS_PATH
        self._orig_config = cs.CONFIG_PATH
        cs.BASE_DIR = self.base
        cs.OUT_DIR = self.out
        cs.ORDERS_PATH = self.orders_dir / 'policy_overrides.json'
        cs.CONFIG_PATH = self.config_dir / 'policy_overrides.json'

    def tearDown(self):
        cs.BASE_DIR = self._orig_base
        cs.OUT_DIR = self._orig_out
        cs.ORDERS_PATH = self._orig_orders
        cs.CONFIG_PATH = self._orig_config
        self.tmp.cleanup()

    def _write_policy(self, lookback=60):
        payload = {'sizing': {'cov_lookback_days': lookback}}
        cs.CONFIG_PATH.write_text(json.dumps(payload), encoding='utf-8')

    def _write_history(self, tickers, days=120):
        dates = pd.date_range('2021-01-01', periods=days, freq='B')
        rows = []
        for t in tickers:
            base_price = 10 if t == tickers[0] else 20
            prices = base_price + pd.Series(range(days)).mul(0.1 if t == tickers[0] else 0.15)
            rows.append(pd.DataFrame({
                'Date': dates.strftime('%Y-%m-%d'),
                'Ticker': t,
                'Close': prices,
            }))
        df = pd.concat(rows, ignore_index=True)
        df.to_csv(self.out / 'prices_history.csv', index=False)

    def test_requires_non_index_tickers(self):
        self._write_policy()
        self._write_history(['VNINDEX'], days=120)
        with self.assertRaises(SystemExit) as ctx:
            cs.calibrate(write=False)
        self.assertIn('No non-index tickers', str(ctx.exception))

    def test_calibrate_writes_cov_reg(self):
        self._write_policy(lookback=40)
        self._write_history(['AAA', 'BBB'], days=80)
        cov_reg = cs.calibrate(write=True)
        self.assertGreaterEqual(cov_reg, 0.0)
        config = json.loads(cs.CONFIG_PATH.read_text(encoding='utf-8'))
        self.assertIn('cov_reg', config.get('sizing', {}))
        self.assertAlmostEqual(config['sizing']['cov_reg'], cov_reg, places=6)
