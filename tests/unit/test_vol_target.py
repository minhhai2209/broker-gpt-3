import unittest

from lib.risk.vol_target import ann_from_daily, scale_budget


class TestVolTarget(unittest.TestCase):
    def test_scale_budget_bounds(self):
        # daily std that annualizes to target/2 -> scale capped at hi (1.4)
        target = 0.15
        realized = target / 2.0
        s = scale_budget(realized, target, 0.6, 1.4)
        self.assertAlmostEqual(s, 1.4, places=6)
        # realized 2x target -> scale is lo bound (0.6)
        s2 = scale_budget(2.0 * target, target, 0.6, 1.4)
        self.assertAlmostEqual(s2, 0.6, places=6)

    def test_ann_from_daily(self):
        self.assertAlmostEqual(ann_from_daily(0.01), 0.01 * (252.0 ** 0.5))

