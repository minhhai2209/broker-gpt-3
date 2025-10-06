import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.engine import config_io


class TestPolicyOverrides(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        base = Path(self.tmp_dir.name)
        (base / 'config').mkdir(parents=True, exist_ok=True)
        (base / 'out' / 'orders').mkdir(parents=True, exist_ok=True)

        # Patch config_io module-level paths to isolate the test sandbox
        config_io.BASE_DIR = base
        config_io.OUT_DIR = base / 'out'
        config_io.OUT_ORDERS_DIR = base / 'out' / 'orders'
        config_io.OVERRIDE_SRC = base / 'config' / 'policy_overrides.json'

        # Provide a minimal baseline policy so ensure_policy_override_file() can merge overlays
        default_obj = {
            'buy_budget_frac': 0.10,
            'add_max': 4,
            'new_max': 3,
            'thresholds': {
                'base_add': 0.35,
                'base_new': 0.40,
                'trim_th': -0.05,
            },
            'sizing': {
                'add_share': 0.5,
                'new_share': 0.5,
            },
            'sector_bias': {},
            'ticker_bias': {},
        }
        (base / 'config' / 'policy_default.json').write_text(
            json.dumps(default_obj, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    def test_runtime_deep_merge_overrides(self) -> None:
        base = Path(config_io.BASE_DIR)
        override_obj = {
            'buy_budget_frac': 0.12,
            'thresholds': {'base_add': 0.99},
            'sizing': {'add_share': 0.9},
            'sector_bias': {'Energy': 0.1},
            'ticker_bias': {'AAA': 0.05},
        }
        (base / 'config' / 'policy_overrides.json').write_text(
            json.dumps(override_obj, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

        dest = config_io.ensure_policy_override_file()
        merged = json.loads(dest.read_text(encoding='utf-8'))

        # All overrides should be honored directly; no guardrails strip or alter keys.
        self.assertEqual(merged['buy_budget_frac'], 0.12)
        self.assertEqual(merged['sector_bias'], {'Energy': 0.1})
        self.assertEqual(merged['ticker_bias'], {'AAA': 0.05})
        self.assertEqual(merged['thresholds']['base_add'], 0.99)
        self.assertEqual(merged['sizing']['add_share'], 0.9)

    def test_ai_overlay_full_replace(self) -> None:
        base = Path(config_io.BASE_DIR)
        ai_override = {
            'buy_budget_frac': 0.08,
            'add_max': 2,
            'new_max': 1,
            'rationale': 'test override',
        }
        (base / 'config' / 'policy_ai_overrides.json').write_text(
            json.dumps(ai_override, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

        dest = config_io.ensure_policy_override_file()
        merged = json.loads(dest.read_text(encoding='utf-8'))

        # The AI overlay should override baseline values directly (no guardrail merge).
        self.assertEqual(merged['buy_budget_frac'], 0.08)
        self.assertEqual(merged['add_max'], 2)
        self.assertEqual(merged['new_max'], 1)
        self.assertEqual(merged['rationale'], 'test override')


if __name__ == '__main__':
    unittest.main()
