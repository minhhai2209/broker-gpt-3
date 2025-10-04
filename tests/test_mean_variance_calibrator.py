import unittest
import numpy as np
import pandas as pd

from scripts.engine.mean_variance_calibrator import calibrate_mean_variance_params


class TestMeanVarianceCalibrator(unittest.TestCase):
    def setUp(self):
        rng = np.linspace(1.0, 2.0, 320)
        dates = pd.date_range('2023-01-01', periods=rng.size, freq='B')
        data = []
        for i, date in enumerate(dates):
            data.append({'Date': date, 'Ticker': 'AAA', 'Close': 100 * rng[i]})
            data.append({'Date': date, 'Ticker': 'BBB', 'Close': 50 * (1.0 + 0.1 * np.sin(i / 20.0))})
            data.append({'Date': date, 'Ticker': 'VNINDEX', 'Close': 1000 * (1.0 + 0.02 * np.sin(i / 30.0))})
        self.prices_history = pd.DataFrame(data)
        self.scores = {'AAA': 0.8, 'BBB': 0.2}
        self.tickers = ['AAA', 'BBB']
        self.sector_map = {'AAA': 'Tech', 'BBB': 'Materials'}
        self.sizing_conf = {
            'max_pos_frac': 0.2,
            'max_sector_frac': 0.3,
            'min_names_target': 2,
            'bl_rf_annual': 0.05,
            'bl_mkt_prem_annual': 0.08,
        }

    def test_calibration_returns_best_params(self):
        calibration_conf = {
            'enable': 1,
            'risk_alpha': [3.0, 5.0],
            'cov_reg': [0.0001, 0.0003],
            'bl_alpha_scale': [0.015, 0.025],
            'lookback_days': 200,
            'test_horizon_days': 40,
            'min_history_days': 260,
        }
        outcome = calibrate_mean_variance_params(
            prices_history=self.prices_history,
            scores=self.scores,
            tickers=self.tickers,
            market_symbol='VNINDEX',
            sector_by_ticker=self.sector_map,
            sizing_conf=self.sizing_conf,
            calibration_conf=calibration_conf,
        )
        self.assertIn(outcome.params['risk_alpha'], calibration_conf['risk_alpha'])
        self.assertIn(outcome.params['cov_reg'], calibration_conf['cov_reg'])
        self.assertIn(outcome.params['bl_alpha_scale'], calibration_conf['bl_alpha_scale'])
        self.assertIn('results', outcome.diagnostics)
        self.assertGreater(outcome.diagnostics['grid_size'], 0)


if __name__ == '__main__':
    unittest.main()
