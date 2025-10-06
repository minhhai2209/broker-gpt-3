import json
import os
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from scripts.tuning import codex_policy_budget_bias_tuner as budget_tuner
import scripts.engine.config_io as config_io
import scripts.engine.pipeline as pipeline
import scripts.orders.order_engine as oe


@contextmanager
def change_cwd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class TestIntegrationTuneAndOrders(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline_path = Path('config/policy_default.json')
        if not self.baseline_path.exists():
            self.skipTest('baseline policy_default.json missing')

    def _init_temp_repo(self, base: Path) -> None:
        (base / 'config').mkdir(parents=True, exist_ok=True)
        (base / 'out' / 'orders').mkdir(parents=True, exist_ok=True)
        (base / 'out' / 'debug').mkdir(parents=True, exist_ok=True)
        (base / 'data').mkdir(parents=True, exist_ok=True)
        (base / 'config' / 'policy_default.json').write_text(
            self.baseline_path.read_text(encoding='utf-8'),
            encoding='utf-8',
        )

    def test_budget_tuner_writes_ai_overrides_in_test_mode(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            self._init_temp_repo(base)

            def fake_run(cmd, input=None, stdout=None, stderr=None, check=None, cwd=None):  # type: ignore[override]
                Path(cwd, budget_tuner.ANALYSIS_FILENAME).write_text('analysis', encoding='utf-8')
                gen_payload = {
                    'buy_budget_frac': 0.12,
                    'sector_bias': {'Finance': 0.1},
                    'ticker_bias': {'AAA': -0.1},
                    'news_risk_tilt': -0.5,
                    'rationale': 'integration test',
                }
                Path(cwd, budget_tuner.OUTPUT_FILENAME).write_text(json.dumps(gen_payload), encoding='utf-8')
                return SimpleNamespace(stdout=b'END\n')

            with ExitStack() as stack:
                stack.enter_context(change_cwd(base))
                stack.enter_context(patch.object(budget_tuner, 'BASE_DIR', base))
                stack.enter_context(patch.object(budget_tuner, 'CONFIG_DIR', base / 'config'))
                stack.enter_context(patch.object(budget_tuner, 'OUT_DIR', base / 'out'))
                stack.enter_context(patch('subprocess.run', fake_run))
                stack.enter_context(patch.dict(os.environ, {'BROKER_CX_REASONING': 'low'}, clear=False))

                budget_tuner.main()

            generated = base / 'config' / 'policy_ai_overrides.json'
            self.assertTrue(generated.exists(), 'policy_ai_overrides.json should be written in test mode')
            payload = json.loads(generated.read_text(encoding='utf-8'))
            self.assertIn('buy_budget_frac', payload)
            self.assertIn('news_risk_tilt', payload, 'raw payload should be preserved without guardrails')

    def test_generate_orders_in_test_mode(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            self._init_temp_repo(base)
            prices_hist = pd.DataFrame(
                [
                    {'Date': '2023-01-01', 'Ticker': 'VNINDEX', 'Close': 1000.0, 'Open': 995.0, 'High': 1005.0, 'Low': 990.0},
                    {'Date': '2023-01-02', 'Ticker': 'VNINDEX', 'Close': 1002.0, 'Open': 1000.0, 'High': 1006.0, 'Low': 998.0},
                ]
            )
            prices_hist.to_csv(base / 'out' / 'prices_history.csv', index=False)

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
                    pricing={
                        'tc_roundtrip_frac': 0.0,
                        'fill_prob': {
                            'base': 0.3,
                            'cross': 0.9,
                            'near_ceiling': 0.05,
                            'min': 0.05,
                            'decay_scale_min_ticks': 5.0,
                        },
                        'slippage_model': {
                            'alpha_bps': 5.0,
                            'beta_dist_per_tick': 1.0,
                            'beta_size': 45.0,
                            'beta_vol': 8.0,
                            'mae_bps': 15.0,
                        },
                    },
                    thresholds={'near_ceiling_pct': 0.98},
                    diag_warnings=[],
                    filtered_records=[],
                    debug_filters={},
                    ttl_overrides={},
                )
                return [order], {'AAA': 'Test order'}, regime_stub

            with ExitStack() as stack:
                stack.enter_context(change_cwd(base))
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
                stack.enter_context(patch('scripts.orders.order_engine.ensure_pipeline_artifacts', fake_pipeline))
                stack.enter_context(patch('scripts.orders.order_engine.decide_actions', fake_decide_actions))
                stack.enter_context(patch('scripts.orders.order_engine.build_orders', fake_build_orders))

                oe.run()

            self.assertTrue(build_called.get('ok'), 'order_engine.build_orders should be invoked')

            orders_dir = base / 'out' / 'orders'
            # With test mode removed, outputs should be written
            expected_any = [
                'orders_final.csv',
                'orders_analysis.txt',
            ]
            self.assertTrue(any((orders_dir / name).exists() for name in expected_any), 'expected outputs not found')


if __name__ == '__main__':
    unittest.main()
