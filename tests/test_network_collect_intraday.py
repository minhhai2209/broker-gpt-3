import unittest

from scripts.collect_intraday import fetch_intraday_series


class TestNetworkCollectIntraday(unittest.TestCase):
    def test_fetch_intraday_series(self):
        # 24h window to increase chance of data availability
        df = fetch_intraday_series('VNINDEX', window_minutes=24*60)
        if df is None or df.empty:
            self.skipTest('No intraday data returned for window')
        self.assertIn('ts', df.columns)
        self.assertIn('price', df.columns)
        self.assertGreater(len(df), 0)


if __name__ == '__main__':
    unittest.main()

