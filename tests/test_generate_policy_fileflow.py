import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import scripts.ai.codex_policy_budget_bias_tuner as budget_tuner
import scripts.ai.guardrails as guardrails


class TestGeneratePolicyFileflow(unittest.TestCase):
    def _seed_base(self, base: Path) -> None:
        # Minimal sample file content (not parsed, only embedded into prompt)
        (base / 'config').mkdir(parents=True, exist_ok=True)
        (base / 'out' / 'debug').mkdir(parents=True, exist_ok=True)
        (base / 'config' / 'policy_overrides.sample.json').write_text('{"schema":"minimal"}', encoding='utf-8')
        # Baseline policy required for guardrails
        baseline = {
            "buy_budget_frac": 0.10,
            "add_max": 4,
            "new_max": 3,
            "thresholds": {
                "base_add": 0.35,
                "base_new": 0.40,
                "trim_th": -0.05,
            },
            "sizing": {
                "add_share": 0.5,
                "new_share": 0.5,
                "reuse_sell_proceeds_frac": 0.1,
            },
            "sector_bias": {},
            "ticker_bias": {},
        }
        (base / 'config' / 'policy_default.json').write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2), encoding='utf-8'
        )

    def test_generate_policy_copies_file_on_END(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._seed_base(base)
            # Patch module paths
            budget_tuner.BASE_DIR = base
            budget_tuner.CONFIG_DIR = base / 'config'
            budget_tuner.OUT_DIR = base / 'out'
            guardrails.BASE_DIR = base
            guardrails.CONFIG_DIR = base / 'config'
            guardrails.OUT_DIR = base / 'out'
            guardrails.STATE_PATH = guardrails.OUT_DIR / 'orders' / 'ai_override_state.json'
            guardrails.DEFAULT_POLICY_PATH = guardrails.CONFIG_DIR / 'policy_default.json'
            guardrails.METRICS_PATH = guardrails.OUT_DIR / 'metrics.csv'

            def fake_run(cmd, input=None, stdout=None, stderr=None, check=None, cwd=None):
                # Simulate model writing analysis and output JSON file in cwd
                Path(cwd, budget_tuner.ANALYSIS_FILENAME).write_text('round analysis', encoding='utf-8')
                gen_path = Path(cwd, budget_tuner.OUTPUT_FILENAME)
                gen_path.write_text('{"buy_budget_frac": 0.12}', encoding='utf-8')
                # STDOUT contains END marker
                return SimpleNamespace(stdout=b"END\n")

            with patch('subprocess.run', fake_run), patch.dict(os.environ, {'BROKER_CX_GEN_ROUNDS': '1'}, clear=False):
                budget_tuner.main()

            # Expect copied file exists with correct content
            dest = base / 'config' / 'policy_overrides.json'
            self.assertTrue(dest.exists())
            self.assertEqual(json.loads(dest.read_text(encoding='utf-8')), {"buy_budget_frac": 0.12})

    def test_generate_policy_missing_file_raises(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._seed_base(base)
            budget_tuner.BASE_DIR = base
            budget_tuner.CONFIG_DIR = base / 'config'
            budget_tuner.OUT_DIR = base / 'out'
            guardrails.BASE_DIR = base
            guardrails.CONFIG_DIR = base / 'config'
            guardrails.OUT_DIR = base / 'out'
            guardrails.STATE_PATH = guardrails.OUT_DIR / 'orders' / 'ai_override_state.json'
            guardrails.DEFAULT_POLICY_PATH = guardrails.CONFIG_DIR / 'policy_default.json'
            guardrails.METRICS_PATH = guardrails.OUT_DIR / 'metrics.csv'

            def fake_run(cmd, input=None, stdout=None, stderr=None, check=None, cwd=None):
                Path(cwd, budget_tuner.ANALYSIS_FILENAME).write_text('some analysis', encoding='utf-8')
                # Do NOT write output file here
                return SimpleNamespace(stdout=b"END\n")

            with patch('subprocess.run', fake_run), self.assertRaises(SystemExit) as ctx:
                budget_tuner.main()
            self.assertIn('Missing generated file', str(ctx.exception))
            # Raw output should be saved for inspection
            files = list((base / 'out' / 'debug').glob('codex_policy_raw_END_*.txt'))
            self.assertTrue(files, 'raw output should be written when END without file')

    def test_generate_policy_continue_flow(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._seed_base(base)
            budget_tuner.BASE_DIR = base
            budget_tuner.CONFIG_DIR = base / 'config'
            budget_tuner.OUT_DIR = base / 'out'
            guardrails.BASE_DIR = base
            guardrails.CONFIG_DIR = base / 'config'
            guardrails.OUT_DIR = base / 'out'
            guardrails.STATE_PATH = guardrails.OUT_DIR / 'orders' / 'ai_override_state.json'
            guardrails.DEFAULT_POLICY_PATH = guardrails.CONFIG_DIR / 'policy_default.json'
            guardrails.METRICS_PATH = guardrails.OUT_DIR / 'metrics.csv'

            def fake_run(cmd, input=None, stdout=None, stderr=None, check=None, cwd=None):
                Path(cwd, budget_tuner.ANALYSIS_FILENAME).write_text('first round analysis', encoding='utf-8')
                return SimpleNamespace(stdout=b"CONTINUE\n")

            with patch('subprocess.run', fake_run), patch.dict(os.environ, {'BROKER_CX_GEN_ROUNDS': '1'}, clear=False):
                with self.assertRaises(SystemExit) as ctx:
                    budget_tuner.main()
            self.assertIn('Exceeded maximum analysis rounds', str(ctx.exception))
            latest = base / 'out' / 'debug' / 'codex_analysis_latest.txt'
            self.assertTrue(latest.exists())
            self.assertIn('first round analysis', latest.read_text(encoding='utf-8'))
