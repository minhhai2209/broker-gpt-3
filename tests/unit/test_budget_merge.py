import unittest

from types import SimpleNamespace
from lib.utils.budget import merge_budget


class TestBudgetMerge(unittest.TestCase):
    def test_merge_budget_modes(self):
        baseline = 0.10
        cands = [0.06, 0.12]
        # risk-on (p>=0.70) -> max
        ctx = SimpleNamespace(regime=SimpleNamespace(risk_on_probability=0.80))
        self.assertAlmostEqual(merge_budget(baseline, cands, ctx), max([baseline] + cands))
        # risk-off (p<=0.30) -> min
        ctx2 = SimpleNamespace(regime=SimpleNamespace(risk_on_probability=0.20))
        self.assertAlmostEqual(merge_budget(baseline, cands, ctx2), min([baseline] + cands))
        # neutral -> median
        ctx3 = SimpleNamespace(regime=SimpleNamespace(risk_on_probability=0.50))
        self.assertEqual(merge_budget(baseline, cands, ctx3), sorted([baseline] + cands)[1])

