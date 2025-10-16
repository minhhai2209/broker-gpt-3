import unittest

from lib.risk.kelly import kelly_fraction


class TestKellyLite(unittest.TestCase):
    def test_kelly_monotonic(self):
        # p increases -> fraction increases
        f1 = kelly_fraction(0.55, 2.0, f_max=0.02)
        f2 = kelly_fraction(0.65, 2.0, f_max=0.02)
        self.assertGreaterEqual(f2, f1)
        # R increases -> fraction increases
        f3 = kelly_fraction(0.60, 1.5, f_max=0.02)
        f4 = kelly_fraction(0.60, 2.0, f_max=0.02)
        self.assertGreaterEqual(f4, f3)
        # clamp to [0, f_max]
        self.assertEqual(kelly_fraction(0.0, 2.0, f_max=0.02), 0.0)
        self.assertLessEqual(kelly_fraction(0.90, 10.0, f_max=0.02), 0.02)

