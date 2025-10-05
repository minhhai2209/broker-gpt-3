from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import pandas as pd

from scripts.build_presets_all import build_presets_all


class TestBuildPresetsFailFast(unittest.TestCase):
    def test_raises_when_session_summary_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pre_path = base / "precomputed.csv"
            snapshot_path = base / "snapshot.csv"
            session_summary_path = base / "session_summary.csv"
            out_path = base / "presets.csv"

            pd.DataFrame(
                [
                    {
                        "Ticker": "AAA",
                        "MA10": 10.0,
                        "MA20": 11.0,
                        "MA50": 12.0,
                        "BB20Upper": 13.0,
                        "BB20Lower": 9.0,
                        "ATR14": 1.5,
                    }
                ]
            ).to_csv(pre_path, index=False)

            pd.DataFrame(
                [
                    {
                        "Ticker": "AAA",
                        "Price": 15.0,
                    }
                ]
            ).to_csv(snapshot_path, index=False)

            session_summary_path.write_text("", encoding="utf-8")

            with self.assertRaises(pd.errors.EmptyDataError):
                build_presets_all(
                    precomputed_path=str(pre_path),
                    snapshot_path=str(snapshot_path),
                    out_path=str(out_path),
                    session_summary_path=str(session_summary_path),
                    daily_band_pct=0.1,
                )


if __name__ == "__main__":
    unittest.main()
