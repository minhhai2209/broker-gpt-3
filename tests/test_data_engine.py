from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from scripts.engine.data_engine import DataEngine, EngineConfig, MarketDataService


class FakeMarketDataService(MarketDataService):
    def __init__(self, history: pd.DataFrame, intraday: pd.DataFrame) -> None:
        self._history = history
        self._intraday = intraday

    def load_history(self, tickers):
        return self._history

    def load_intraday(self, tickers):
        return self._intraday


class DataEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def _write_config(self) -> Path:
        config_path = self.base / "config.yaml"
        industry_csv = self.base / "industry.csv"
        industry_df = pd.DataFrame(
            [
                {"Ticker": "AAA", "Sector": "Tech"},
                {"Ticker": "BBB", "Sector": "Finance"},
            ]
        )
        industry_df.to_csv(industry_csv, index=False)
        portfolio_dir = self.base / "pf"
        order_dir = self.base / "orders"
        (portfolio_dir / "alpha").mkdir(parents=True, exist_ok=True)
        (portfolio_dir / "alpha" / "portfolio.csv").write_text(
            "Ticker,Quantity,AvgPrice\nAAA,10,12\nBBB,5,20\n", encoding="utf-8"
        )
        config_path.write_text(
            """
            universe:
              csv: {industry_csv}
              include_indices: true
            technical_indicators:
              moving_averages: [2]
              rsi_periods: [2]
              atr_periods: [2]
              macd:
                fast: 2
                slow: 3
                signal: 2
            presets:
              sample:
                buy_tiers: [-0.1]
                sell_tiers: [0.1]
            portfolio:
              directory: {portfolio_dir}
              order_history_directory: {order_dir}
            output:
              base_dir: {out_dir}
              market_snapshot: technical.csv
              presets_dir: .
              portfolios_dir: .
              diagnostics_dir: .
            data:
              history_cache: {cache_dir}
              history_min_days: 1
              intraday_window_minutes: 60
            """.format(
                industry_csv=industry_csv,
                portfolio_dir=portfolio_dir,
                order_dir=order_dir,
                out_dir=self.base / "out",
                cache_dir=self.base / "cache",
            ),
            encoding="utf-8",
        )
        return config_path

    def test_engine_generates_outputs(self):
        history_df = pd.DataFrame(
            {
                "Date": ["2024-07-01", "2024-07-02", "2024-07-01", "2024-07-02"],
                "Ticker": ["AAA", "AAA", "BBB", "BBB"],
                "Open": [10, 11, 19, 21],
                "High": [11, 12, 21, 22],
                "Low": [9, 10, 18, 19],
                "Close": [11, 12, 20, 21],
                "Volume": [1000, 1200, 800, 900],
                "t": [1, 2, 1, 2],
            }
        )
        intraday_df = pd.DataFrame(
            {
                "Ticker": ["AAA", "BBB"],
                "Ts": [3, 3],
                "Price": [12.5, 21.5],
                "RSI14": [55, 60],
                "TimeVN": ["2024-07-02 14:30:00", "2024-07-02 14:30:00"],
            }
        )
        config_path = self._write_config()
        config = EngineConfig.from_yaml(config_path)
        engine = DataEngine(config, FakeMarketDataService(history_df, intraday_df))
        summary = engine.run()
        snapshot_path = config.market_snapshot_path
        self.assertTrue(snapshot_path.exists())
        snapshot = pd.read_csv(snapshot_path)
        self.assertIn("SMA_2", snapshot.columns)
        self.assertEqual(len(snapshot), 2)
        presets_path = config.presets_dir / "preset_sample.csv"
        self.assertTrue(presets_path.exists())
        presets_df = pd.read_csv(presets_path)
        self.assertIn("Buy_1", presets_df.columns)
        self.assertAlmostEqual(presets_df.loc[0, "Buy_1"], round(presets_df.loc[0, "LastPrice"] * 0.9, 4))
        bundle_path = config.output_base_dir / "bundle_alpha.zip"
        self.assertTrue(bundle_path.exists())
        with zipfile.ZipFile(bundle_path) as zf:
            names = set(zf.namelist())
            self.assertIn("technical.csv", names)
            self.assertIn("preset_sample.csv", names)
            self.assertIn("positions.csv", names)
            positions_df = pd.read_csv(zf.open("positions.csv"))
            self.assertIn("UnrealizedPnL", positions_df.columns)
            self.assertAlmostEqual(positions_df.loc[0, "LastPrice"], 12.5)
        self.assertGreater(summary["tickers"], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
