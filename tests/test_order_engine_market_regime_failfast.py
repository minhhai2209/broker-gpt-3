import unittest

import pandas as pd

from scripts.order_engine import get_market_regime


class TestMarketRegimeFailFast(unittest.TestCase):
    def setUp(self) -> None:
        self.sector_strength = pd.DataFrame()
        self.tuning = {}

    def test_empty_session_summary_raises(self):
        session_summary = pd.DataFrame(columns=["SessionPhase", "InVNSession", "IndexChangePct"])
        with self.assertRaisesRegex(ValueError, "session_summary is empty"):
            get_market_regime(session_summary, self.sector_strength, self.tuning)

    def test_missing_required_columns_raises(self):
        session_summary = pd.DataFrame([
            {"SessionPhase": "morning", "IndexChangePct": 0.1}
        ])
        with self.assertRaisesRegex(KeyError, "InVNSession"):
            get_market_regime(session_summary, self.sector_strength, self.tuning)

    def test_non_numeric_index_change_pct_raises(self):
        session_summary = pd.DataFrame([
            {"SessionPhase": "morning", "InVNSession": 1, "IndexChangePct": "not-a-number"}
        ])
        with self.assertRaisesRegex(ValueError, "IndexChangePct"):
            get_market_regime(session_summary, self.sector_strength, self.tuning)


if __name__ == "__main__":
    unittest.main()
