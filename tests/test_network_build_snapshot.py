import unittest

from scripts.build_snapshot import latest_price_from_dchart


class TestNetworkBuildSnapshot(unittest.TestCase):
    def test_latest_price_from_dchart(self):
        px = latest_price_from_dchart('VNINDEX')
        if px is None:
            self.skipTest('No intraday price returned (possibly outside availability)')
        self.assertGreater(px, 0.0)


if __name__ == '__main__':
    unittest.main()

