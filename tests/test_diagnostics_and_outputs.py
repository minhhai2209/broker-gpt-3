import json
import os
import shutil
import unittest
from pathlib import Path

import pandas as pd

from scripts.orders.order_engine import build_orders, decide_actions, run


class TestDiagnosticsAndOutputs(unittest.TestCase):
    def setUp(self):
        self.out_dir = Path('out')
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.orders_dir = self.out_dir / 'orders'
        self.orders_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_orders_dir()
        self.hist_path = self.out_dir / 'prices_history.csv'
        self._hist_backup = self.hist_path.read_bytes() if self.hist_path.exists() else None
        hist = pd.DataFrame({
            'Date': pd.date_range(end=pd.Timestamp.today(), periods=30, freq='D'),
            'Ticker': ['VNINDEX'] * 30,
            'Close': [1000.0 + i for i in range(30)],
        })
        hist['High'] = hist['Close'] * 1.01
        hist['Low'] = hist['Close'] * 0.99
        hist.to_csv(self.hist_path, index=False)

    def tearDown(self):
        self._cleanup_orders_dir()
        if self.hist_path.exists():
            self.hist_path.unlink()
        if self._hist_backup is not None:
            self.hist_path.write_bytes(self._hist_backup)

    def _cleanup_orders_dir(self):
        keep = {'policy_overrides.json'}
        if not self.orders_dir.exists():
            return
        for path in self.orders_dir.iterdir():
            if path.name in keep:
                continue
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)

    def _base_inputs(self):
        portfolio = pd.DataFrame([{'Ticker': 'AAA', 'Quantity': 0, 'AvgCost': 10.0}], columns=['Ticker','Quantity','AvgCost'])
        snapshot = pd.DataFrame([
            {'Ticker': 'AAA', 'Price': 20.0},
            {'Ticker': 'BBB', 'Price': 20.0},
            {'Ticker': 'VNINDEX', 'Price': 1100.0},
        ])
        metrics = pd.DataFrame([
            {'Ticker': 'AAA', 'RSI14': 65.0, 'Sector': 'X', 'MomRetNorm': 0.9, 'LiqNorm': 0.8, 'ATR14_Pct': 1.0},
            {'Ticker': 'BBB', 'RSI14': 62.0, 'Sector': 'X', 'MomRetNorm': 0.8, 'LiqNorm': 0.5, 'ATR14_Pct': 4.0},
            {'Ticker': 'VNINDEX', 'RSI14': 55.0},
        ])
        industry = pd.DataFrame({'Ticker': ['AAA','BBB'], 'Sector': ['X','X']})
        presets = pd.DataFrame([
            {'Ticker': 'AAA', 'MA20': 19.0, 'MA50': 18.0, 'BandFloor_Tick': 18.0, 'BandCeiling_Tick': 22.0},
            {'Ticker': 'BBB', 'MA20': 19.0, 'MA50': 18.0, 'BandFloor_Tick': 18.0, 'BandCeiling_Tick': 22.0},
        ])
        sector_strength = pd.DataFrame([
            {'sector': 'Tất cả', 'breadth_above_ma50_pct': 60.0, 'avg_rsi14': 55.0}
        ])
        session_summary = pd.DataFrame([
            {'SessionPhase': 'morning', 'InVNSession': 1, 'IndexChangePct': 0.4}
        ])
        tuning = {
            'buy_budget_frac': 0.20,
            'add_max': 10,
            'new_max': 10,
            'weights': {
                'w_trend': 0.2,
                'w_momo': 0.4,
                'w_mom_ret': 0.0,
                'w_liq': 0.2,
                'w_vol_guard': 0.0,
                'w_beta': 0.0,
                'w_sector': 0.0,
                'w_sector_sent': 0.0,
                'w_ticker_sent': 0.0,
                'w_roe': 0.0,
                'w_earnings_yield': 0.0,
                'w_rs': 0.0,
            },
            'regime_model': {
                'intercept': 0.0,
                'threshold': 0.5,
                'components': {
                    'trend': {'mean': 0.5, 'std': 0.2, 'weight': 1.0},
                    'breadth': {'mean': 0.5, 'std': 0.2, 'weight': 1.0},
                    'index_return': {'mean': 0.0, 'std': 0.6, 'weight': 0.8},
                    'volatility': {'mean': 0.5, 'std': 0.2, 'weight': 0.6},
                }
            },
            'thresholds': {
                'base_add': 0.10,
                'base_new': 0.10,
                'trim_th': -0.10,
                'q_add': 0.0,
                'q_new': 0.0,
                'min_liq_norm': 0.0,
                'near_ceiling_pct': 0.99,
                'tp_pct': 0.0,
                'sl_pct': 1.0,
                'tp_atr_mult': None,
                'sl_atr_mult': None,
                'tp_floor_pct': 0.0,
                'sl_floor_pct': 0.0,
                'tp_trim_frac': 0.30,
                'exit_on_ma_break': 0,
                'exit_ma_break_rsi': 45.0,
                'trim_rsi_below_ma20': 45.0,
                'trim_rsi_macdh_neg': 40.0,
                'cooldown_days': 0,
                'exit_ma_break_score_gate': 0.0,
                'tilt_exit_downgrade_min': 0.05,
            },
            'sector_bias': {},
            'ticker_bias': {},
            'market_filter': {
                'risk_off_index_drop_pct': 0.5,
                'risk_off_trend_floor': -0.015,
                'risk_off_breadth_floor': 0.4,
                'breadth_relax_margin': 0.0,
                'market_score_soft_floor': 0.55,
                'market_score_hard_floor': 0.35,
                'leader_min_rsi': 50.0,
                'leader_min_mom_norm': 0.5,
                'leader_require_ma20': 0,
                'leader_require_ma50': 0,
                'leader_max': 10,
                'risk_off_drawdown_floor': 0.40,
                'index_atr_soft_pct': 0.90,
                'index_atr_hard_pct': 0.99,
                'idx_chg_smoothed_hard_drop': 0.5,
                'trend_norm_hard_floor': -0.25,
                'vol_ann_hard_ceiling': 0.60,
                'guard_new_scale_cap': 0.40,
                'atr_soft_scale_cap': 0.50,
                'severe_drop_mult': 1.50,
            },
            'pricing': {
                'risk_on_buy': ["Aggr","Bal","Cons","MR","Break"],
                'risk_on_sell': ["Cons","Bal","Break","MR","Aggr"],
                'risk_off_buy': ["Cons","MR","Bal","Aggr","Break"],
                'risk_off_sell': ["MR","Cons","Bal","Aggr","Break"],
                'atr_fallback_buy_mult': 0.25,
                'atr_fallback_sell_mult': 0.25,
                'tc_roundtrip_frac': 0.0,
                'tc_sell_tax_frac': 0.001,
                'fill_prob': {
                    'base': 0.30,
                    'cross': 0.90,
                    'near_ceiling': 0.05,
                    'min': 0.05,
                    'decay_scale_min_ticks': 5.0,
                    'partial_fill_kappa': 0.65,
                    'min_fill_notional_vnd': 5000000.0,
                },
                'slippage_model': {
                    'alpha_bps': 5.0,
                    'beta_dist_per_tick': 1.0,
                    'beta_size': 40.0,
                    'beta_vol': 8.0,
                    'mae_bps': 10.0,
                    'last_fit_date': None,
                },
            },
            'orders_ui': {
                'ttl_minutes': { 'base': 12, 'soft': 7, 'hard': 5 },
                'buy_ttl_floor_minutes': 60,
                'buy_ttl_reversal_minutes': 10,
                'ttl_bucket_minutes': {
                    'low': { 'base': 14, 'soft': 11, 'hard': 8 },
                    'medium': { 'base': 11, 'soft': 9, 'hard': 7 },
                    'high': { 'base': 8, 'soft': 6, 'hard': 5 },
                },
                'watchlist': { 'enable': 1, 'min_priority': 0.25, 'micro_window': 3 },
                'suggestions_top_n': 3,
            },
            'sizing': {
                'softmax_tau': 0.6,
                'add_share': 1.0,
                'new_share': 1.0,
                'min_lot': 100,
                'risk_weighting': 'score_softmax',
                'risk_alpha': 1.0,
                'max_pos_frac': 1.0,
                'max_sector_frac': 1.0,
                'reuse_sell_proceeds_frac': 0.0,
                'leftover_redistribute': 1,
                'min_ticket_k': 0.0,
                'risk_blend': 1.0,
                'cov_lookback_days': 90,
                'cov_reg': 0.0005,
                'risk_parity_floor': 0.2,
                'dynamic_caps': {
                    'enable': 0,
                    'pos_min': 0.10,
                    'pos_max': 0.16,
                    'sector_min': 0.28,
                    'sector_max': 0.40,
                    'blend': 1.0,
                    'override_static': 0,
                },
            },
        }
        return portfolio, snapshot, metrics, industry, presets, sector_strength, session_summary, tuning

    def _prices_history(self):
        dates = pd.date_range('2024-01-02', periods=10, freq='B')
        vnindex = pd.DataFrame({
            'Date': dates,
            'Ticker': ['VNINDEX'] * len(dates),
            'Close': 1100.0 + pd.Series(range(len(dates))) * 0.5,
            'High': 1100.5 + pd.Series(range(len(dates))) * 0.5,
            'Low': 1099.5 + pd.Series(range(len(dates))) * 0.5,
        })
        extra = pd.DataFrame({
            'Date': dates,
            'Ticker': ['AAA'] * len(dates),
            'Close': 20.0 + pd.Series(range(len(dates))) * 0.1,
            'High': 20.1 + pd.Series(range(len(dates))) * 0.1,
            'Low': 19.9 + pd.Series(range(len(dates))) * 0.1,
        })
        return pd.concat([vnindex, extra], ignore_index=True)

    def test_regime_components_snapshot_fields(self):
        portfolio, snapshot, metrics, industry, presets, sector_strength, session_summary, tuning = self._base_inputs()
        prices_history = pd.DataFrame({'Date': ['2024-01-02'], 'Ticker': ['AAA'], 'Close': [20.0]})
        actions, scores, feats, regime = decide_actions(
            portfolio, snapshot, metrics, presets, industry, sector_strength, session_summary, tuning
        )
        pnl_summary = pd.DataFrame([[50000.0, 50000.0, 0.0, 0.0]], columns=['TotalCost','TotalMarket','TotalPnL','ReturnPct'])
        build_orders(actions, portfolio, snapshot, metrics, presets, pnl_summary, scores, regime, prices_history)
        comp_path = self.orders_dir / 'regime_components.json'
        self.assertTrue(comp_path.exists(), 'regime_components.json should be written')
        payload = json.loads(comp_path.read_text(encoding='utf-8'))
        required = {
            'market_score',
            'risk_on_probability',
            'index_vol_annualized',
            'index_atr14_pct',
            'index_atr_percentile',
            'turnover_percentile',
            'diag_warnings',
        }
        self.assertTrue(required.issubset(payload.keys()))
        self.assertIsInstance(payload['diag_warnings'], list)

    def test_orders_analysis_contains_filters_and_hint(self):
        portfolio, snapshot, metrics, industry, presets, sector_strength, session_summary, tuning = self._base_inputs()
        presets.loc[presets['Ticker'] == 'AAA', 'BandCeiling_Tick'] = 20.0
        prices_history = self._prices_history()

        from unittest.mock import patch

        def fake_pipeline():
            return (
                portfolio,
                prices_history,
                snapshot,
                metrics,
                sector_strength,
                presets,
                session_summary,
            )

        env_vars = {
            'CALIBRATE_REGIME_COMPONENTS': os.environ.get('CALIBRATE_REGIME_COMPONENTS'),
            'CALIBRATE_REGIME': os.environ.get('CALIBRATE_REGIME'),
            'CALIBRATE_MARKET_FILTER': os.environ.get('CALIBRATE_MARKET_FILTER'),
            'CALIBRATE_BREADTH_FLOOR': os.environ.get('CALIBRATE_BREADTH_FLOOR'),
            'CALIBRATE_LEADER_GATES': os.environ.get('CALIBRATE_LEADER_GATES'),
            'CALIBRATE_LIQUIDITY': os.environ.get('CALIBRATE_LIQUIDITY'),
            'CALIBRATE_SIZING': os.environ.get('CALIBRATE_SIZING'),
            'CALIBRATE_THRESHOLDS_TOPK': os.environ.get('CALIBRATE_THRESHOLDS_TOPK'),
            'CALIBRATE_SIZING_TAU': os.environ.get('CALIBRATE_SIZING_TAU'),
            'CALIBRATE_RISK_LIMITS': os.environ.get('CALIBRATE_RISK_LIMITS'),
            'CALIBRATE_DYNAMIC_CAPS': os.environ.get('CALIBRATE_DYNAMIC_CAPS'),
        }
        os.environ.update({
            'CALIBRATE_REGIME_COMPONENTS': '0',
            'CALIBRATE_REGIME': '0',
            'CALIBRATE_MARKET_FILTER': '0',
            'CALIBRATE_BREADTH_FLOOR': '0',
            'CALIBRATE_LEADER_GATES': '0',
            'CALIBRATE_LIQUIDITY': '0',
            'CALIBRATE_SIZING': '0',
            'CALIBRATE_THRESHOLDS_TOPK': '0',
            'CALIBRATE_SIZING_TAU': '0',
            'CALIBRATE_RISK_LIMITS': '0',
            'CALIBRATE_DYNAMIC_CAPS': '0',
        })
        try:
            with patch('scripts.orders.order_engine.ensure_pipeline_artifacts', side_effect=fake_pipeline), \
                 patch('scripts.engine.config_io.ensure_policy_override_file', return_value=None), \
                 patch('scripts.engine.config_io.suggest_tuning', return_value=tuning), \
                 patch('scripts.orders.order_engine.suggest_tuning', return_value=tuning):
                run()
        finally:
            for key, val in env_vars.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

        analysis_path = self.orders_dir / 'orders_analysis.txt'
        self.assertTrue(analysis_path.exists())
        lines = analysis_path.read_text(encoding='utf-8').splitlines()
        self.assertTrue(any('Execution window hint:' in line for line in lines))
        self.assertTrue(any('Filtered (near ceiling): AAA' in line for line in lines))
