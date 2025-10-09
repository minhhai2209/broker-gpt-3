import pandas as pd
from unittest import TestCase

from scripts.orders import order_engine


class TestNewBuyExecution(TestCase):
    def setUp(self) -> None:
        self.snap = pd.Series({"Price": 20.0})
        self.metrics = pd.Series({
            "TickSizeHOSE_Thousand": 0.05,
            "AvgTurnover20D_k": 200000.0,
            "ATR14_Pct": 1.5,
        })
        self.fill_cfg = {
            "target_prob": 0.20,
            "max_chase_ticks": 1,
            "horizon_s": 60,
            "window_sigma_s": 45,
            "window_vol_s": 90,
            "cancel_ratio_per_min": 0.10,
            "joiner_factor": 0.10,
            "no_cross": True,
        }

    def test_skip_when_probability_below_target(self):
        fill_cfg = dict(self.fill_cfg)
        fill_cfg["target_prob"] = 0.95
        limit, diag = order_engine._apply_new_buy_execution(
            "AAA", 19.8, 20.0, self.snap, self.metrics, 100, fill_cfg
        )
        self.assertIsNone(limit)
        self.assertIsInstance(diag, dict)
        self.assertEqual(diag.get("status"), "skipped")

    def test_accepts_when_probability_sufficient(self):
        fill_cfg = dict(self.fill_cfg)
        fill_cfg["target_prob"] = 0.10
        limit, diag = order_engine._apply_new_buy_execution(
            "AAA", 20.0, 20.0, self.snap, self.metrics, 100, fill_cfg
        )
        self.assertIsNotNone(limit)
        self.assertIsInstance(diag, dict)
        self.assertEqual(diag.get("status"), "accepted")
        self.assertLessEqual(limit, 20.0)

    def test_insufficient_data_falls_back(self):
        snap = pd.Series({"Price": 0.0})
        limit, diag = order_engine._apply_new_buy_execution(
            "AAA", 19.8, 20.0, snap, self.metrics, 100, self.fill_cfg
        )
        self.assertEqual(limit, 19.8)
        self.assertIsInstance(diag, dict)
        self.assertEqual(diag.get("status"), "insufficient_data")
