import json
import os
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from scripts.tuning.calibrators import calibrate_tilts as ai_cal
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

    # Minimal seed for Codex fileflow tests (does not parse; just a context payload)
    def _seed_base(self, base: Path) -> None:
        (base / 'config').mkdir(parents=True, exist_ok=True)
        (base / 'out' / 'debug').mkdir(parents=True, exist_ok=True)
        # Baseline policy to provide schema context
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

    def test_ai_calibrator_merges_overrides(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            self._init_temp_repo(base)

            def fake_run(cmd, input=None, stdout=None, stderr=None, check=None, cwd=None):  # type: ignore[override]
                Path(cwd, ai_cal.ANALYSIS_FILENAME).write_text('analysis', encoding='utf-8')
                gen_payload = {
                    'buy_budget_frac': 0.12,
                    'sector_bias': {'Finance': 0.1},
                    'ticker_bias': {'AAA': -0.1},
                    'news_risk_tilt': -0.5,
                    'rationale': 'integration test',
                }
                Path(cwd, ai_cal.OUTPUT_FILENAME).write_text(json.dumps(gen_payload), encoding='utf-8')
                return SimpleNamespace(stdout=b'END\n')

            with ExitStack() as stack:
                stack.enter_context(change_cwd(base))
                stack.enter_context(patch.object(ai_cal, 'BASE_DIR', base))
                stack.enter_context(patch.object(ai_cal, 'CONFIG_DIR', base / 'config'))
                stack.enter_context(patch.object(ai_cal, 'OUT_DIR', base / 'out'))
                stack.enter_context(patch.object(ai_cal, 'ORDERS_DIR', base / 'out' / 'orders'))
                stack.enter_context(patch('subprocess.run', fake_run))
                stack.enter_context(patch.dict(os.environ, {'BROKER_CX_REASONING': 'low'}, clear=False))
                # Seed runtime policy for merge
                (base / 'out' / 'orders').mkdir(parents=True, exist_ok=True)
                (base / 'out' / 'orders' / 'policy_overrides.json').write_text('{}', encoding='utf-8')
                ai_cal.calibrate(write=True)

            merged = json.loads((base / 'out' / 'orders' / 'policy_overrides.json').read_text(encoding='utf-8'))
            self.assertIn('sector_bias', merged)
            self.assertIn('ticker_bias', merged)

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
                    risk_on=True,
                    orders_ui={'ttl_minutes': {'base': 12, 'soft': 9, 'hard': 7}},
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
                    market_filter={'index_atr_soft_pct': 0.5, 'index_atr_hard_pct': 0.8},
                    buy_budget_frac=0.10,
                    buy_budget_frac_effective=0.10,
                    market_score=0.55,
                    risk_on_probability=0.60,
                    top_sectors=['Technology', 'Finance'],
                    trend_strength=0.02,
                    breadth_hint=0.55,
                    model_components={'breadth_long': 0.50, 'ma200_slope': 0.01, 'uptrend': 1.0},
                    drawdown_pct=0.05,
                    turnover_percentile=0.50,
                    index_atr_percentile=0.4,
                    index_change_pct=0.001,
                    index_change_pct_smoothed=0.001,
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
                return [order], {'AAA': 'Test order'}, regime

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

    # Unified: bring Codex fileflow tests into this integration suite
    def test_generate_policy_copies_file_on_END(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._seed_base(base)
            # Patch module paths and seed runtime policy
            ai_cal.BASE_DIR = base
            ai_cal.CONFIG_DIR = base / 'config'
            ai_cal.OUT_DIR = base / 'out'
            ai_cal.ORDERS_DIR = base / 'out' / 'orders'
            (base / 'out' / 'orders').mkdir(parents=True, exist_ok=True)
            (base / 'out' / 'orders' / 'policy_overrides.json').write_text('{}', encoding='utf-8')

            def fake_run(cmd, input=None, stdout=None, stderr=None, check=None, cwd=None):  # type: ignore[override]
                Path(cwd, ai_cal.ANALYSIS_FILENAME).write_text('round analysis', encoding='utf-8')
                gen_path = Path(cwd, ai_cal.OUTPUT_FILENAME)
                gen_path.write_text('{"buy_budget_frac": 0.12}', encoding='utf-8')
                return SimpleNamespace(stdout=b"END\n")

            with patch('subprocess.run', fake_run), patch.dict(os.environ, {'BROKER_CX_GEN_ROUNDS': '1', 'BROKER_CX_REASONING': 'low'}, clear=False):
                ai_cal.calibrate(write=True)
            merged = json.loads((base / 'out' / 'orders' / 'policy_overrides.json').read_text(encoding='utf-8'))
            self.assertNotIn('buy_budget_frac', merged)

    def test_generate_policy_missing_file_raises(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._seed_base(base)
            ai_cal.BASE_DIR = base
            ai_cal.CONFIG_DIR = base / 'config'
            ai_cal.OUT_DIR = base / 'out'
            ai_cal.ORDERS_DIR = base / 'out' / 'orders'
            (base / 'out' / 'orders').mkdir(parents=True, exist_ok=True)
            (base / 'out' / 'orders' / 'policy_overrides.json').write_text('{}', encoding='utf-8')

            def fake_run(cmd, input=None, stdout=None, stderr=None, check=None, cwd=None):  # type: ignore[override]
                Path(cwd, ai_cal.ANALYSIS_FILENAME).write_text('some analysis', encoding='utf-8')
                return SimpleNamespace(stdout=b"END\n")

            with patch('subprocess.run', fake_run), self.assertRaises(SystemExit) as ctx:
                ai_cal.calibrate(write=True)
            self.assertIn('Missing generated file', str(ctx.exception))
            files = list((base / 'out' / 'debug').glob('codex_policy_raw_END_*.txt'))
            self.assertTrue(files)

    def test_generate_policy_continue_flow(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._seed_base(base)
            ai_cal.BASE_DIR = base
            ai_cal.CONFIG_DIR = base / 'config'
            ai_cal.OUT_DIR = base / 'out'
            (base / 'out' / 'orders').mkdir(parents=True, exist_ok=True)
            (base / 'out' / 'orders' / 'policy_overrides.json').write_text('{}', encoding='utf-8')

            def fake_run(cmd, input=None, stdout=None, stderr=None, check=None, cwd=None):  # type: ignore[override]
                Path(cwd, ai_cal.ANALYSIS_FILENAME).write_text('first round analysis', encoding='utf-8')
                return SimpleNamespace(stdout=b"CONTINUE\n")

            with patch('subprocess.run', fake_run), patch.dict(os.environ, {'BROKER_CX_GEN_ROUNDS': '1', 'BROKER_CX_REASONING': 'low'}, clear=False):
                with self.assertRaises(SystemExit) as ctx:
                    ai_cal.calibrate(write=True)
            self.assertIn('Exceeded maximum analysis rounds', str(ctx.exception))
            latest = base / 'out' / 'debug' / 'codex_analysis_latest.txt'
            self.assertTrue(latest.exists())
            self.assertIn('first round analysis', latest.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
