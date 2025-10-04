from pathlib import Path
import unittest

from scripts.utils import hose_tick_size, round_to_tick, clip_to_band, load_universe_from_files


class TestUtils(unittest.TestCase):
    def test_hose_tick_size(self):
        self.assertEqual(hose_tick_size(9.99), 0.01)
        self.assertEqual(hose_tick_size(10.0), 0.05)
        self.assertEqual(hose_tick_size(49.94), 0.05)
        self.assertEqual(hose_tick_size(49.95), 0.10)
        self.assertEqual(hose_tick_size(100.0), 0.10)

    def test_round_and_clip(self):
        self.assertEqual(round_to_tick(20.03, 0.05), 20.05)
        self.assertEqual(round_to_tick(19.97, 0.05), 19.95)
        self.assertEqual(clip_to_band(19.0, 19.1, 21.0), 19.1)
        self.assertEqual(clip_to_band(22.0, 19.1, 21.0), 21.0)
        self.assertEqual(clip_to_band(20.0, None, None), 20.0)

    def test_load_universe_from_files_reads_csv(self):
        path = 'data/industry_map.csv'
        self.assertTrue(Path(path).exists(), 'industry_map.csv missing in repo')
        uni = load_universe_from_files(path)
        self.assertIsInstance(uni, list)
        self.assertGreater(len(uni), 0)
        self.assertIn('FPT', uni)


if __name__ == '__main__':
    unittest.main()
