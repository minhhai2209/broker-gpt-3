import json
import os
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import scripts.ai.generate_policy_overrides as gpo
import scripts.ai.guardrails as guardrails
import scripts.engine.config_io as config_io
import scripts.engine.pipeline as pipeline
import scripts.order_engine as oe


@contextmanager
def change_cwd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class TestIntegrationTuneAndOrders(unittest.TestCase):
    def test_tune_and_generate_orders_in_test_mode(self) -> None:
        baseline_path = Path('config/policy_default.json')
        if not baseline_path.exists():
            self.skipTest('baseline policy_default.json missing')

        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            (base / 'config').mkdir(parents=True, exist_ok=True)
            (base / 'out' / 'orders').mkdir(parents=True, exist_ok=True)
            (base / 'data').mkdir(parents=True, exist_ok=True)
            # Seed baseline policy for prompt + guardrails
            (base / 'config' / 'policy_default.json').write_text(
                baseline_path.read_text(encoding='utf-8'),
                encoding='utf-8',
            )
            prices_hist = pd.DataFrame(
                [
                    {'Date': '2023-01-01', 'Ticker': 'VNINDEX', 'Close': 1000.0, 'Open': 995.0, 'High': 1005.0, 'Low': 990.0},
                    {'Date': '2023-01-02', 'Ticker': 'VNINDEX', 'Close': 1002.0, 'Open': 1000.0, 'High': 1006.0, 'Low': 998.0},
                ]
            )
            prices_hist.to_csv(base / 'out' / 'prices_history.csv', index=False)

            cmd_calls: list[list[str]] = []

            def fake_run(cmd, input=None, stdout=None, stderr=None, check=None, cwd=None):  # type: ignore[override]
                cmd_calls.append(cmd)
                self.assertIn('reasoning_effort=low', cmd)
                Path(cwd, gpo.ANALYSIS_FILENAME).write_text('analysis', encoding='utf-8')
                gen_payload = {
                    'buy_budget_frac': 0.12,
                    'add_max': 3,
                    'new_max': 3,
                    'sector_bias': {},
                    'ticker_bias': {},
                    'rationale': 'integration test',
                }
                Path(cwd, gpo.OUTPUT_FILENAME).write_text(json.dumps(gen_payload), encoding='utf-8')
                return SimpleNamespace(stdout=b'END\n')

            def fake_pipeline():
                return (
                    pd.DataFrame([{'Ticker': 'AAA', 'Quantity': 0, 'AvgCost': 10.0}]),
                    pd.DataFrame([{'Date': pd.Timestamp('2023-01-01'), 'Ticker': 'AAA', 'Close': 10.0}]),
                    pd.DataFrame([{'Ticker': 'AAA', 'Price': 20.0}]),
                    pd.DataFrame([{'Ticker': 'AAA', 'RSI14': 60.0}]),
                    pd.DataFrame([{'sector': 'All', 'breadth_above_ma50_pct': 60.0}]),
                    pd.DataFrame([{'Ticker': 'AAA'}]),
                    pd.DataFrame([{'SessionPhase': 'pre', 'InVNSession': 0, 'IndexChangePct': 0.1}]),
                )

            build_called = {}

            def fake_decide_actions(portfolio, snapshot, metrics, presets, industry, sector_strength, session_summary, tuning, prices_history=None):  # noqa: E501
                regime_stub = SimpleNamespace(
                    orders_ui={},
                    pricing={'tc_roundtrip_frac': 0.0},
                    thresholds={},
                    market_filter={'index_atr_soft_pct': 0.5, 'index_atr_hard_pct': 0.8},
                    index_atr_percentile=0.4,
                    diag_warnings=[],
                    filtered_records=[],
                    debug_filters={},
                    ttl_overrides={},
                )
                actions = {'AAA': 'new'}
                scores = {'AAA': 0.9}
                feats = {'AAA': {'atr_pct': 0.01, 'adtv20_k': 50.0}}
                return actions, scores, feats, regime_stub

            def fake_build_orders(actions, portfolio, snapshot, metrics, presets, pnl_summary, scores, regime, prices_history):
                build_called['ok'] = True
                order = oe.Order(ticker='AAA', side='BUY', quantity=100, limit_price=20.0, note='Test order')
                regime_stub = SimpleNamespace(
                    risk_on=True,
                    diag_warnings=[],
                    filtered_records=[],
                    debug_filters={},
                    ttl_overrides={},
                )
                return [order], {'AAA': 'Test order'}, regime_stub

            with ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, {'BROKER_TEST_MODE': '1'}, clear=False))
                stack.enter_context(change_cwd(base))
                stack.enter_context(patch.object(gpo, 'BASE_DIR', base))
                stack.enter_context(patch.object(gpo, 'CONFIG_DIR', base / 'config'))
                stack.enter_context(patch.object(gpo, 'OUT_DIR', base / 'out'))
                stack.enter_context(patch.object(guardrails, 'BASE_DIR', base))
                stack.enter_context(patch.object(guardrails, 'CONFIG_DIR', base / 'config'))
                stack.enter_context(patch.object(guardrails, 'OUT_DIR', base / 'out'))
                stack.enter_context(patch.object(guardrails, 'STATE_PATH', base / 'out' / 'orders' / 'ai_override_state.json'))
                stack.enter_context(patch.object(guardrails, 'DEFAULT_POLICY_PATH', base / 'config' / 'policy_default.json'))
                stack.enter_context(patch.object(guardrails, 'METRICS_PATH', base / 'out' / 'metrics.csv'))
                stack.enter_context(patch.object(guardrails, 'AUDIT_CSV_PATH', base / 'out' / 'orders' / 'ai_overrides_audit.csv'))
                stack.enter_context(patch.object(guardrails, 'AUDIT_JSONL_PATH', base / 'out' / 'orders' / 'ai_overrides_audit.jsonl'))
                stack.enter_context(patch.object(config_io, 'BASE_DIR', base))
                stack.enter_context(patch.object(config_io, 'OUT_DIR', base / 'out'))
                stack.enter_context(patch.object(config_io, 'OUT_ORDERS_DIR', base / 'out' / 'orders'))
                stack.enter_context(patch.object(config_io, 'OVERRIDE_SRC', base / 'config' / 'policy_overrides.json'))
                stack.enter_context(patch.object(oe, 'BASE_DIR', base))
                stack.enter_context(patch.object(oe, 'OUT_DIR', base / 'out'))
                stack.enter_context(patch.object(oe, 'OUT_ORDERS_DIR', base / 'out' / 'orders'))
                stack.enter_context(patch.object(oe, 'DATA_DIR', base / 'data'))
                stack.enter_context(patch.object(pipeline, 'BASE_DIR', base))
                stack.enter_context(patch.object(pipeline, 'OUT_DIR', base / 'out'))
                stack.enter_context(patch.object(pipeline, 'DATA_DIR', base / 'data'))
                stack.enter_context(patch('subprocess.run', fake_run))
                stack.enter_context(patch('scripts.order_engine.ensure_pipeline_artifacts', fake_pipeline))
                stack.enter_context(patch('scripts.order_engine.decide_actions', fake_decide_actions))
                stack.enter_context(patch('scripts.order_engine.build_orders', fake_build_orders))

                gpo.main()
                oe.run()

            self.assertTrue(cmd_calls, 'Codex CLI should be invoked')
            self.assertIn('reasoning_effort=low', cmd_calls[0])
            generated = base / 'config' / 'policy_overrides.json'
            self.assertTrue(generated.exists(), 'policy overrides should be written in config')
            self.assertTrue(build_called.get('ok'), 'order_engine.build_orders should be invoked')

            orders_dir = base / 'out' / 'orders'
            forbidden = [
                'orders_final.csv',
                'orders_print.txt',
                'orders_reasoning.csv',
                'orders_quality.csv',
                'orders_watchlist.csv',
                'orders_filtered.csv',
                'trade_suggestions.txt',
                'last_actions.csv',
                'position_state.csv',
            ]
            for name in forbidden:
                self.assertFalse((orders_dir / name).exists(), f'{name} should not be persisted in test mode')


if __name__ == '__main__':
    unittest.main()
