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
            portfolio:
              directory: {portfolio_dir}
              order_history_directory: {order_dir}
            output:
              base_dir: {out_dir}
              presets_dir: .
              portfolios_dir: .
              diagnostics_dir: .
            execution:
              aggressiveness: med
              max_order_pct_adv: 0.1
              slice_adv_ratio: 0.25
              min_lot: 100
              max_qty_per_order: 500000
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

        out_dir = config.output_base_dir
        technical_path = out_dir / "technical.csv"
        bands_path = out_dir / "bands.csv"
        levels_path = out_dir / "levels.csv"
        sizing_path = out_dir / "sizing.csv"
        signals_path = out_dir / "signals.csv"
        limits_path = out_dir / "limits.csv"
        positions_path = out_dir / "positions.csv"
        sector_path = out_dir / "sector.csv"

        for path in [
            technical_path,
            bands_path,
            levels_path,
            sizing_path,
            signals_path,
            limits_path,
            positions_path,
            sector_path,
        ]:
            self.assertTrue(path.exists(), f"Missing output: {path}")

        technical_df = pd.read_csv(technical_path)
        self.assertIn("Z20", technical_df.columns)
        self.assertEqual(len(technical_df), 2)
        aaa_row = technical_df.loc[technical_df["Ticker"] == "AAA"].iloc[0]
        self.assertAlmostEqual(aaa_row["Last"], 12.5)
        self.assertAlmostEqual(aaa_row["Ref"], 12.0)
        self.assertAlmostEqual(aaa_row["ChangePct"], 12.5 / 12.0 - 1.0, places=6)

        bands_df = pd.read_csv(bands_path)
        aaa_band = bands_df.loc[bands_df["Ticker"] == "AAA"].iloc[0]
        self.assertAlmostEqual(aaa_band["Ceil"], 12.8)
        self.assertAlmostEqual(aaa_band["Floor"], 11.2)

        sizing_df = pd.read_csv(sizing_path)
        self.assertTrue((sizing_df["DeltaQty"] == 0).all())
        self.assertTrue((sizing_df["SliceCount"] == 0).all())

        positions_df = pd.read_csv(positions_path)
        self.assertIn("PNLPct", positions_df.columns)
        self.assertEqual(len(positions_df), 2)
        self.assertAlmostEqual(
            positions_df.loc[positions_df["Ticker"] == "AAA", "Last"].iloc[0], 12.5
        )

        bundle_path = out_dir / "bundle_alpha.zip"
        self.assertTrue(bundle_path.exists())
        with zipfile.ZipFile(bundle_path) as zf:
            names = set(zf.namelist())
            self.assertEqual(
                names,
                {
                    "technical.csv",
                    "bands.csv",
                    "levels.csv",
                    "sizing.csv",
                    "signals.csv",
                    "limits.csv",
                    "positions.csv",
                    "sector.csv",
                },
            )
            positions_zip = pd.read_csv(zf.open("positions.csv"))
            self.assertIn("PNLPct", positions_zip.columns)
            self.assertAlmostEqual(
                positions_zip.loc[positions_zip["Ticker"] == "AAA", "Last"].iloc[0], 12.5
            )

        self.assertGreater(summary["tickers"], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
