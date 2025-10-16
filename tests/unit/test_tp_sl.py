import unittest

from scripts.orders.order_engine import resolve_tp_sl


class TestTpSl(unittest.TestCase):
    def test_tp_sl_bounds(self):
        th = {
            'tp_pct': 0.0,
            'sl_pct': 0.0,
            'tp_atr_mult': 1.35,
            'sl_atr_mult': 1.50,
            'tp_floor_pct': 0.03,
            'sl_floor_pct': 0.02,
            'tp_cap_pct': 0.10,
            'sl_cap_pct': 0.08,
            'tp_sl_mode': 'atr_per_ticker',
            'tp_rule': 'dynamic_only',
            'sl_rule': 'dynamic_only',
        }
        feats = {'atr_pct': 0.015}  # 1.5%
        tp, sl, info = resolve_tp_sl(th, feats)
        self.assertIsNotNone(tp)
        self.assertIsNotNone(sl)
        # Respect floor/cap
        self.assertGreaterEqual(tp, th['tp_floor_pct'])
        self.assertLessEqual(tp, th['tp_cap_pct'])
        self.assertGreaterEqual(sl, th['sl_floor_pct'])
        self.assertLessEqual(sl, th['sl_cap_pct'])

