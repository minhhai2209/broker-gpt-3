from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.api.server import create_app


class ServerUploadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.config_path = self._create_config()
        self.app = create_app(self.config_path)
        self.client = self.app.test_client()

    def _create_config(self) -> Path:
        config_path = self.base / "config.yaml"
        portfolio_dir = self.base / "portfolios"
        order_dir = self.base / "orders"
        portfolio_dir.mkdir()
        order_dir.mkdir()
        industry_csv = self.base / "industry.csv"
        pd.DataFrame([{"Ticker": "AAA", "Sector": "Tech"}]).to_csv(industry_csv, index=False)
        config_path.write_text(
            """
            universe:
              csv: {industry_csv}
            presets:
              basic:
                buy_tiers: [0]
                sell_tiers: [0]
            portfolio:
              directory: {portfolio_dir}
              order_history_directory: {order_dir}
            output:
              base_dir: {out_dir}
            data:
              history_cache: {cache_dir}
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

    def test_upload_persists_portfolio_and_fills(self):
        payload = {
            "profile": "alpha",
            "portfolio": [
                {"Ticker": "AAA", "Quantity": 10, "AvgPrice": 12.5},
            ],
            "fills": [
                {"timestamp": "2024-07-01T09:00:00+07:00", "ticker": "AAA", "side": "BUY", "quantity": 10, "price": 12.5}
            ],
        }
        resp = self.client.post("/upload", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")
        portfolio_file = Path(data["portfolio_path"])
        self.assertTrue(portfolio_file.exists())
        stored = pd.read_csv(portfolio_file)
        self.assertEqual(stored.loc[0, "Ticker"], "AAA")
        fills_file = Path(data["fills_path"])
        self.assertTrue(fills_file.exists())
        fills_df = pd.read_csv(fills_file)
        self.assertEqual(fills_df.loc[0, "ticker"], "AAA")

    def test_upload_rejects_missing_profile(self):
        payload = {"portfolio": []}
        resp = self.client.post("/upload", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertEqual(data["error"], "missing_profile")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
