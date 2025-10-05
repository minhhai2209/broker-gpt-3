import json
import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import scripts.ai.guardrails as gr
from scripts.engine import config_io


class TestGuardrails(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        (base / 'config').mkdir(parents=True, exist_ok=True)
        (base / 'out' / 'orders').mkdir(parents=True, exist_ok=True)

        # Minimal baseline policy for guardrails
        baseline = {
            'buy_budget_frac': 0.10,
            'add_max': 4,
            'new_max': 3,
            'sector_bias': {},
            'ticker_bias': {},
        }
        (base / 'config' / 'policy_default.json').write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2), encoding='utf-8'
        )

        # Patch module paths
        gr.BASE_DIR = base
        gr.CONFIG_DIR = base / 'config'
        gr.OUT_DIR = base / 'out'
        gr.STATE_PATH = gr.OUT_DIR / 'orders' / 'ai_override_state.json'
        gr.DEFAULT_POLICY_PATH = gr.CONFIG_DIR / 'policy_default.json'
        gr.METRICS_PATH = gr.OUT_DIR / 'metrics.csv'

        # metrics.csv count for slot bound
        (gr.OUT_DIR / 'metrics.csv').write_text('Ticker\nAAA\nBBB\nCCC\nDDD\nEEE\n', encoding='utf-8')

    def test_news_tilt_mapping_and_rate_limit(self):
        today = date.today()
        sanitized, next_state = gr.enforce_guardrails(
            overrides={'news_risk_tilt': 1.0},
            baseline=json.loads(Path(gr.DEFAULT_POLICY_PATH).read_text(encoding='utf-8')),
            metrics_count=5,
            state={},
            today=today,
        )
        # Expect +0.02 budget (rate-limit 0.04 allows it)
        self.assertAlmostEqual(sanitized.get('buy_budget_frac'), 0.12, places=6)
        # Expect slot delta +1 but capped by ceil(N/6)=1 for N=5 -> 1
        self.assertEqual(sanitized.get('add_max'), 1)
        self.assertEqual(sanitized.get('new_max'), 1)

    def test_bias_decay_and_ttl(self):
        today = date.today()
        state = {
            'ticker_bias': {
                'AAA': {
                    'value': 0.10,
                    'last_updated': (today - timedelta(days=1)).isoformat(),
                    'expiry': (today + timedelta(days=3)).isoformat(),
                }
            }
        }
        sanitized, next_state = gr.enforce_guardrails(
            overrides={},
            baseline=json.loads(Path(gr.DEFAULT_POLICY_PATH).read_text(encoding='utf-8')),
            metrics_count=5,
            state=state,
            today=today,
        )
        # 20% decay
        self.assertIn('ticker_bias', sanitized)
        self.assertAlmostEqual(sanitized['ticker_bias']['AAA'], 0.08, places=5)

    def test_kill_switch(self):
        today = date.today()
        sanitized, next_state = gr.enforce_guardrails(
            overrides={'buy_budget_frac': 0.18},
            baseline=json.loads(Path(gr.DEFAULT_POLICY_PATH).read_text(encoding='utf-8')),
            metrics_count=5,
            state={'kill_switch': True},
            today=today,
        )
        # All overrides suppressed
        self.assertEqual(sanitized, {})

    def test_weekly_positive_cap_applied(self):
        today = date.today()
        # Simulate previous state at 0.14
        prev_state = {
            'buy_budget_frac': {
                'value': 0.14,
                'last_updated': (today - timedelta(days=1)).isoformat(),
                'expiry': (today + timedelta(days=2)).isoformat(),
            }
        }
        # Try to jump to 0.20 (> baseline+0.06) -> should be capped to <= 0.16
        sanitized, _ = gr.enforce_guardrails(
            overrides={'buy_budget_frac': 0.20},
            baseline=json.loads(Path(gr.DEFAULT_POLICY_PATH).read_text(encoding='utf-8')),
            metrics_count=5,
            state=prev_state,
            today=today,
        )
        self.assertLessEqual(sanitized.get('buy_budget_frac', 0.0), 0.16 + 1e-6)


class TestAllowedOverridePaths(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        (base / 'config').mkdir(parents=True, exist_ok=True)
        (base / 'out' / 'orders').mkdir(parents=True, exist_ok=True)

        # Patch config_io paths
        config_io.BASE_DIR = base
        config_io.OUT_DIR = base / 'out'
        config_io.OUT_ORDERS_DIR = base / 'out' / 'orders'
        config_io.OVERRIDE_SRC = base / 'config' / 'policy_overrides.json'

        # Baseline default with multiple keys
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
            json.dumps(default_obj, ensure_ascii=False, indent=2), encoding='utf-8'
        )

    def test_runtime_deep_merge_overrides(self):
        base = Path(config_io.BASE_DIR)
        # Provide an override that includes disallowed fields
        override_obj = {
            'buy_budget_frac': 0.12,
            'thresholds': {'base_add': 0.99},
            'sizing': {'add_share': 0.9},
            'sector_bias': {'Energy': 0.1},
            'ticker_bias': {'AAA': 0.05},
        }
        (base / 'config' / 'policy_overrides.json').write_text(
            json.dumps(override_obj, ensure_ascii=False, indent=2), encoding='utf-8'
        )

        dest = config_io.ensure_policy_override_file()
        merged = json.loads(dest.read_text(encoding='utf-8'))
        # All overrides are applied at runtime (deep-merge). CLI guardrails limit keys upstream.
        self.assertEqual(merged['buy_budget_frac'], 0.12)
        self.assertEqual(merged['sector_bias'], {'Energy': 0.1})
        self.assertEqual(merged['ticker_bias'], {'AAA': 0.05})
        # Calibrator-style fields in thresholds/sizing are also honored at runtime
        self.assertEqual(merged['thresholds']['base_add'], 0.99)
        self.assertEqual(merged['sizing']['add_share'], 0.9)


if __name__ == '__main__':
    unittest.main()
