from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from scripts.engine.data_engine import DataEngine, EngineConfig, EngineError, _as_repo_relative


class DataEngineContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        (self.repo / ".git").mkdir()
        (self.repo / "config").mkdir()
        (self.repo / "data" / "portfolios").mkdir(parents=True)
        (self.repo / "data" / "order_history").mkdir(parents=True)
        (self.repo / "data" / "universe").mkdir(parents=True)
        (self.repo / "out" / "market").mkdir(parents=True)
        (self.repo / "out" / "presets").mkdir()
        (self.repo / "out" / "portfolios").mkdir()
        (self.repo / "out" / "signals").mkdir()
        (self.repo / "out" / "orders").mkdir()
        (self.repo / "out" / "run").mkdir()
        (self.repo / "out" / "news").mkdir()

    def _write_csv(self, path: Path, rows: list[dict]) -> None:
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)

    def test_engine_generates_contract_outputs(self) -> None:
        config_path = self.repo / "config" / "data_engine.yaml"
        config_path.write_text(
            """
paths:
  out: out
  data: data
  config: config
  bundle: .artifacts/engine
""".strip(),
            encoding="utf-8",
        )

        snapshot_rows = [
            {
                "Ticker": "AAA",
                "Last": 24.5,
                "Ref": 24.0,
                "SMA20": 23.5,
                "ATR14": 1.2,
                "RSI14": 60,
                "MACD": 0.5,
                "MACDSignal": 0.3,
                "Z20": -0.6,
                "Ret5d": 0.02,
                "Ret20d": 0.05,
                "ADV20": 150000,
                "High52w": 30.0,
                "Low52w": 20.0,
                "Sector": "Tech",
            },
            {
                "Ticker": "BBB",
                "Last": 14.0,
                "Ref": 14.5,
                "SMA20": 14.3,
                "ATR14": 0.0,
                "RSI14": 40,
                "MACD": -0.1,
                "MACDSignal": 0.0,
                "Z20": 0.2,
                "Ret5d": -0.01,
                "Ret20d": -0.02,
                "ADV20": 0,
                "High52w": 18.0,
                "Low52w": 10.0,
                "Sector": "Finance",
            },
        ]
        self._write_csv(self.repo / "out" / "market" / "technical_snapshot.csv", snapshot_rows)

        presets_dir = self.repo / "out" / "presets"
        self._write_csv(
            presets_dir / "balanced.csv",
            [
                {"Ticker": "AAA", "Side": "BOTH", "RuleType": "offset_ticks_from_last", "Param1": -1, "Param2": 1},
                {"Ticker": "BBB", "Side": "SELL", "RuleType": "risk_off_trim", "Param1": 1, "Param2": ""},
            ],
        )
        self._write_csv(
            presets_dir / "momentum.csv",
            [{"Ticker": "AAA", "Side": "BUY", "RuleType": "weighted_momentum_meanrev", "Param1": 0.7, "Param2": 0.3}],
        )
        self._write_csv(
            presets_dir / "mean_reversion.csv",
            [{"Ticker": "AAA", "Side": "BUY", "RuleType": "SMA20±kATR", "Param1": 0.5, "Param2": 0.25}],
        )
        self._write_csv(
            presets_dir / "risk_off.csv",
            [{"Ticker": "AAA", "Side": "SELL", "RuleType": "risk_off_trim", "Param1": 2, "Param2": ""}],
        )

        self._write_csv(
            self.repo / "data" / "portfolios" / "alpha.csv",
            [{"Ticker": "AAA", "Quantity": 100, "AvgPrice": 22.0}, {"Ticker": "BBB", "Quantity": 0, "AvgPrice": 0.0}],
        )
        self._write_csv(
            self.repo / "out" / "portfolios" / "alpha_positions.csv",
            [{"Ticker": "AAA", "MarketValue": 2450000, "CostBasis": 2200000, "Unrealized": 250000, "PNLPct": 0.113}],
        )
        self._write_csv(
            self.repo / "out" / "portfolios" / "alpha_sector.csv",
            [{"Sector": "Tech", "MarketValue": 5000000, "PNLPct": 6.0}, {"Sector": "Finance", "MarketValue": 1000000, "PNLPct": -3.0}],
        )
        self._write_csv(
            self.repo / "data" / "order_history" / "alpha_fills.csv",
            [
                {
                    "Time": "2024-07-01T09:15:00+07:00",
                    "Ticker": "AAA",
                    "Side": "BUY",
                    "Qty": 100,
                    "Price": 24.0,
                    "WAP": 24.0,
                    "OrderId": "ord-1",
                }
            ],
        )
        self._write_csv(self.repo / "data" / "universe" / "vn100.csv", [{"Ticker": "AAA"}, {"Ticker": "BBB"}])
        self._write_csv(self.repo / "config" / "blocklist.csv", [{"Ticker": "BBB", "Reason": "Watchlist"}])
        (self.repo / "config" / "params.yaml").write_text(
            """
buy_budget_vnd: 10000000
sell_budget_vnd: 0
max_order_pct_adv: 0.05
aggressiveness: med
slice_adv_ratio: 0.02
min_lot: 100
max_qty_per_order: 500000
""".strip(),
            encoding="utf-8",
        )
        self._write_csv(
            self.repo / "out" / "news" / "news_score.csv",
            [
                {
                    "Ticker": "AAA",
                    "Score": 0.6,
                    "Flags": "KQKD",
                    "Sources": "source-a",
                    "AsOf": "2024-07-02T09:00:00+07:00",
                }
            ],
        )

        config = EngineConfig.from_yaml(config_path)
        engine = DataEngine(config)
        summary = engine.run()

        self.assertEqual(summary["tickers"], 2)
        bundle_path = Path(summary["attachment_bundle"])
        self.assertIn(".artifacts/engine", summary["attachment_bundle"])
        self.assertTrue(bundle_path.exists())
        with zipfile.ZipFile(bundle_path) as archive:
            entries = set(archive.namelist())
        self.assertIn("out/market/trading_bands.csv", entries)
        self.assertIn("out/signals/levels.csv", entries)
        self.assertIn("out/signals/sizing.csv", entries)
        self.assertIn("out/signals/signals.csv", entries)
        self.assertIn("out/orders/alpha_LO_latest.csv", entries)

        trading_bands = pd.read_csv(self.repo / "out" / "market" / "trading_bands.csv")
        self.assertIn("AAA", trading_bands["Ticker"].tolist())
        aaa_band = trading_bands.loc[trading_bands["Ticker"] == "AAA"].iloc[0]
        self.assertGreater(aaa_band["Ceil"], aaa_band["Floor"])

        levels = pd.read_csv(self.repo / "out" / "signals" / "levels.csv")
        aaa_balanced = levels[(levels["Ticker"] == "AAA") & (levels["Preset"] == "balanced")].iloc[0]
        self.assertIsInstance(aaa_balanced["NearTouchBuy"], float)
        self.assertIsInstance(aaa_balanced["NearTouchSell"], float)

        sizing = pd.read_csv(self.repo / "out" / "signals" / "sizing.csv")
        self.assertEqual(float(sizing.loc[sizing["Ticker"] == "AAA", "DeltaQty"].iloc[0]), 0.0)

        signals = pd.read_csv(self.repo / "out" / "signals" / "signals.csv")
        bbb_guards = signals.loc[signals["Ticker"] == "BBB", "RiskGuards"].iloc[0]
        self.assertIn("BLOCKLIST", bbb_guards)
        self.assertIn("ZERO_ATR", bbb_guards)

        orders = pd.read_csv(self.repo / "out" / "orders" / "alpha_LO_latest.csv")
        self.assertTrue(orders.empty)

        manifest_path = self.repo / "out" / "run" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertIn("out/market/technical_snapshot.csv", manifest["source_files"])
        preset_rel = _as_repo_relative(self.repo / "out" / "presets" / "balanced.csv", "balanced.csv", self.repo)
        self.assertIn(preset_rel, manifest["source_files"])


    def test_engine_validates_required_outputs_non_empty(self) -> None:
        config_path = self.repo / "config" / "data_engine.yaml"
        config_path.write_text(
            """
paths:
  out: out
  data: data
  config: config
  bundle: .artifacts/engine
""".strip(),
            encoding="utf-8",
        )

        snapshot_rows = [
            {
                "Ticker": "AAA",
                "Last": 24.5,
                "Ref": 24.0,
                "SMA20": 23.5,
                "ATR14": 1.2,
                "RSI14": 60,
                "MACD": 0.5,
                "MACDSignal": 0.3,
                "Z20": -0.6,
                "Ret5d": 0.02,
                "Ret20d": 0.05,
                "ADV20": 150000,
                "High52w": 30.0,
                "Low52w": 20.0,
                "Sector": "Tech",
            }
        ]
        self._write_csv(self.repo / "out" / "market" / "technical_snapshot.csv", snapshot_rows)

        presets_dir = self.repo / "out" / "presets"
        self._write_csv(
            presets_dir / "balanced.csv",
            [{"Ticker": "AAA", "Side": "BOTH", "RuleType": "offset_ticks_from_last", "Param1": -1, "Param2": 1}],
        )

        self._write_csv(
            self.repo / "data" / "portfolios" / "alpha.csv",
            [{"Ticker": "AAA", "Quantity": 100, "AvgPrice": 22.0}],
        )
        self._write_csv(
            self.repo / "out" / "portfolios" / "alpha_positions.csv",
            [{"Ticker": "AAA", "MarketValue": 2450000, "CostBasis": 2200000, "Unrealized": 250000, "PNLPct": 0.113}],
        )
        self._write_csv(
            self.repo / "out" / "portfolios" / "alpha_sector.csv",
            [{"Sector": "Tech", "MarketValue": 5000000, "PNLPct": 6.0}],
        )
        self._write_csv(
            self.repo / "data" / "order_history" / "alpha_fills.csv",
            [
                {
                    "Time": "2024-07-01T09:15:00+07:00",
                    "Ticker": "AAA",
                    "Side": "BUY",
                    "Qty": 100,
                    "Price": 24.0,
                    "WAP": 24.0,
                    "OrderId": "ord-1",
                }
            ],
        )
        self._write_csv(self.repo / "data" / "universe" / "vn100.csv", [{"Ticker": "BBB"}])
        self._write_csv(self.repo / "config" / "blocklist.csv", [{"Ticker": "", "Reason": ""}])
        (self.repo / "config" / "params.yaml").write_text(
            """
buy_budget_vnd: 10000000
sell_budget_vnd: 0
max_order_pct_adv: 0.05
aggressiveness: med
slice_adv_ratio: 0.02
min_lot: 100
max_qty_per_order: 500000
""".strip(),
            encoding="utf-8",
        )

        config = EngineConfig.from_yaml(config_path)
        engine = DataEngine(config)

        with self.assertRaises(EngineError):
            engine.run()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
