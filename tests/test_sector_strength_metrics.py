import unittest
import pandas as pd

from scripts.compute_sector_strength import compute_metrics


class TestSectorStrength(unittest.TestCase):
    def test_compute_metrics_above_ma(self):
        df = pd.DataFrame([
            {'Price': 10, 'MA20': 9, 'MA50': 8, 'RSI14': 60, 'ATR14': 0.5},
            {'Price': 9, 'MA20': 10, 'MA50': 8, 'RSI14': 50, 'ATR14': 0.3},
            {'Price': 12, 'MA20': 11, 'MA50': 13, 'RSI14': 55, 'ATR14': 0.8},
        ])
        m = compute_metrics(df)
        self.assertAlmostEqual(m['breadth_above_ma20_pct'], 66.67, places=2)
        self.assertAlmostEqual(m['breadth_above_ma50_pct'], 66.67, places=2)
        self.assertIn('avg_rsi14', m)


if __name__ == '__main__':
    unittest.main()
