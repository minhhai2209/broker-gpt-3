from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.tools.build_universe import UniverseBuilderError, build_universe


class BuildUniverseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_builds_unique_uppercase_tickers(self) -> None:
        source = self.root / "industry_map.csv"
        df = pd.DataFrame(
            {
                "Ticker": ["abc", "ABC", "def", " ", None],
                "Sector": ["Tech", "Tech", "Finance", "Other", "Other"],
            }
        )
        df.to_csv(source, index=False)

        dest = self.root / "universe" / "vn100.csv"
        result = build_universe(source, dest)

        self.assertTrue(dest.exists())
        self.assertEqual(result, dest.resolve())
        built = pd.read_csv(dest)
        self.assertListEqual(sorted(built.columns), ["Ticker"])
        self.assertListEqual(built["Ticker"].tolist(), ["ABC", "DEF"])

    def test_missing_ticker_column_raises(self) -> None:
        source = self.root / "industry_map.csv"
        pd.DataFrame({"Code": ["AAA"]}).to_csv(source, index=False)
        with self.assertRaises(UniverseBuilderError):
            build_universe(source, self.root / "vn100.csv")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

