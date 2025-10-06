import unittest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta, timezone

from scripts.data_fetching.fetch_ticker_data import fetch_history, ensure_and_load_history_df, VN_TZ


class TestNetworkFetchTickerData(unittest.TestCase):
    def test_fetch_history_vnindex_ok(self):
        now = int(datetime.now(VN_TZ).timestamp())
        frm = int((datetime.now(VN_TZ) - timedelta(days=7)).timestamp())
        js = fetch_history('VNINDEX', 'D', frm, now)
        self.assertIsInstance(js, dict)
        self.assertEqual(js.get('s'), 'ok')
        self.assertTrue(len(js.get('t', [])) > 0)

    def test_ensure_and_load_history_df_minimal(self):
        with tempfile.TemporaryDirectory() as d:
            outdir = str(Path(d) / 'data')
            df = ensure_and_load_history_df(['VNINDEX', 'FPT'], outdir=outdir, min_days=15, resolution='D')
            self.assertFalse(df.empty)
            self.assertTrue(set(['VNINDEX', 'FPT']).issubset(set(df['Ticker'].unique())))
            for c in ['Date','Ticker','Open','High','Low','Close','Volume','t']:
                self.assertIn(c, df.columns)


if __name__ == '__main__':
    unittest.main()

