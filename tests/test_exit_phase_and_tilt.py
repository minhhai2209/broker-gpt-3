import unittest
from pathlib import Path
from datetime import datetime

import pandas as pd

from scripts.orders.order_engine import decide_actions


class TestExitPhaseAndTilt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Seed VNINDEX history for regime
        out_dir = Path('out')
        out_dir.mkdir(parents=True, exist_ok=True)
        dates = pd.date_range('2023-01-02', periods=240, freq='B')
        df = pd.DataFrame({
            'Date': dates.strftime('%Y-%m-%d'),
            'Ticker': ['VNINDEX'] * len(dates),
            'Close': [1000.0 + idx for idx in range(len(dates))],
        })
        df.to_csv(out_dir / 'prices_history.csv', index=False)

    def _base_tuning(self, **overrides):
        t = {
            'buy_budget_frac': 0.1,
            'add_max': 5,
            'new_max': 5,
            'weights': {
                'w_trend': 0.2, 'w_momo': 0.5, 'w_mom_ret': 0.0, 'w_liq': 0.0,
                'w_vol_guard': 0.0, 'w_beta': 0.0, 'w_sector': 0.0, 'w_sector_sent': 0.0,
                'w_ticker_sent': 0.2, 'w_roe': 0.0, 'w_earnings_yield': 0.0, 'w_rs': 0.1,
            },
            'regime_model': {
                'intercept': 0.0, 'threshold': 0.5,
                'components': {
                    'trend': {'mean': 0.5, 'std': 0.2, 'weight': 1.0},
                    'breadth': {'mean': 0.5, 'std': 0.2, 'weight': 1.0},
                    'index_return': {'mean': 0.0, 'std': 0.6, 'weight': 0.5},
                    'volatility': {'mean': 0.5, 'std': 0.2, 'weight': 0.5},
                }
            },
            'thresholds': {
                'base_add': 0.4, 'base_new': 0.5, 'trim_th': -0.05,
                'q_add': 0.0, 'q_new': 0.0,
                'min_liq_norm': 0.0, 'near_ceiling_pct': 0.98,
                'tp_pct': 0.0, 'sl_pct': 1.0,
                'tp_atr_mult': None, 'sl_atr_mult': None,
                'tp_floor_pct': 0.0, 'sl_floor_pct': 0.0,
                'tp_trim_frac': 0.3,
                'exit_on_ma_break': 1,
                'exit_ma_break_min_phase': 'afternoon',
                'exit_ma_break_score_gate': 0.0,
                'tilt_exit_downgrade_min': 0.05,
                'cooldown_days': 0,
                'exit_ma_break_rsi': 45.0,
                'trim_rsi_below_ma20': 45.0,
                'trim_rsi_macdh_neg': 40.0,
            },
            'sector_bias': {},
            'ticker_bias': {},
            'pricing': {
                'risk_on_buy': ["Aggr"],
                'risk_on_sell': ["Cons"],
                'risk_off_buy': ["Cons"],
                'risk_off_sell': ["MR"],
                'atr_fallback_buy_mult': 0.25,
                'tc_roundtrip_frac': 0.0,
                'tc_sell_tax_frac': 0.001,
                'atr_fallback_sell_mult': 0.25,
                'fill_prob': {
                    'base': 0.3,
                    'cross': 0.9,
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
            'sizing': {
                'softmax_tau': 0.6, 'add_share': 0.6, 'new_share': 0.4,
                'min_lot': 100, 'risk_weighting': 'score_softmax', 'risk_alpha': 1.0,
                'max_pos_frac': 0.5, 'max_sector_frac': 0.8,
                'reuse_sell_proceeds_frac': 0.0, 'risk_blend': 1.0,
                'cov_lookback_days': 90, 'cov_reg': 0.0005, 'risk_parity_floor': 0.2,
                'leftover_redistribute': 1, 'min_ticket_k': 0.0,
                'dynamic_caps': {'enable': 0, 'pos_min': 0.1, 'pos_max': 0.2, 'sector_min': 0.3, 'sector_max': 0.4, 'blend': 1.0, 'override_static': 0},
            },
            'market_filter': {
                'risk_off_index_drop_pct': 0.5,
                'risk_off_trend_floor': -0.015,
                'risk_off_breadth_floor': 0.0,
                'breadth_relax_margin': 0.0,
                'market_score_soft_floor': 0.55,
                'market_score_hard_floor': 0.35,
                'leader_min_rsi': 0.0,
                'leader_min_mom_norm': 0.0,
                'leader_require_ma20': 0, 'leader_require_ma50': 0, 'leader_max': 10,
                'risk_off_drawdown_floor': 0.40,
                'idx_chg_smoothed_hard_drop': 0.5,
                'trend_norm_hard_floor': -0.25,
                'vol_ann_hard_ceiling': 0.60,
                'index_atr_soft_pct': 0.80,
                'index_atr_hard_pct': 0.95,
                'guard_new_scale_cap': 0.40,
                'atr_soft_scale_cap': 0.50,
                'severe_drop_mult': 1.50,
            },
        }
        t.update(overrides)
        return t

    def test_exit_ma_break_min_phase_defers(self):
        portfolio = pd.DataFrame([{'Ticker': 'AAA', 'Quantity': 100, 'AvgCost': 10.0}])
        snapshot = pd.DataFrame([{'Ticker': 'AAA', 'Price': 20.0}])
        # Price < MA50 to trigger below MA50; RSI < exit threshold
        presets = pd.DataFrame([{'Ticker': 'AAA', 'MA50': 25.0, 'MA20': 24.0}])
        metrics = pd.DataFrame([{'Ticker': 'AAA', 'RSI14': 40.0}])
        industry = pd.DataFrame({'Ticker': ['AAA'], 'Sector': ['X']})
        sector_strength = pd.DataFrame([])
        # Morning -> TRIM
        session_summary = pd.DataFrame([{'SessionPhase': 'morning', 'InVNSession': 1, 'IndexChangePct': 0.0}])
        tuning = self._base_tuning()
        act, scores, feats, regime = decide_actions(portfolio, snapshot, metrics, presets, industry, sector_strength, session_summary, tuning)
        self.assertEqual(act.get('AAA'), 'trim')
        # Afternoon -> EXIT
        session_summary2 = pd.DataFrame([{'SessionPhase': 'afternoon', 'InVNSession': 1, 'IndexChangePct': 0.0}])
        act2, *_ = decide_actions(portfolio, snapshot, metrics, presets, industry, sector_strength, session_summary2, tuning)
        self.assertEqual(act2.get('AAA'), 'exit')

    def test_tilt_exit_downgrade_min_applies(self):
        portfolio = pd.DataFrame([{'Ticker': 'AAA', 'Quantity': 100, 'AvgCost': 10.0}])
        snapshot = pd.DataFrame([{'Ticker': 'AAA', 'Price': 20.0}])
        presets = pd.DataFrame([{'Ticker': 'AAA', 'MA50': 25.0, 'MA20': 24.0}])
        metrics = pd.DataFrame([{'Ticker': 'AAA', 'RSI14': 40.0}])
        industry = pd.DataFrame({'Ticker': ['AAA'], 'Sector': ['X']})
        sector_strength = pd.DataFrame([])
        session_summary = pd.DataFrame([{'SessionPhase': 'afternoon', 'InVNSession': 1, 'IndexChangePct': 0.0}])
        t = self._base_tuning()
        # Allow tilt to downgrade EXIT->TRIM
        t['thresholds']['exit_ma_break_min_phase'] = 'morning'
        t['thresholds']['exit_ma_break_score_gate'] = 0.0
        t['thresholds']['tilt_exit_downgrade_min'] = 0.05
        t['ticker_bias'] = {'AAA': 0.10}
        act, *_ = decide_actions(portfolio, snapshot, metrics, presets, industry, sector_strength, session_summary, t)
        self.assertEqual(act.get('AAA'), 'trim')

    def test_exit_score_gate_downgrade(self):
        portfolio = pd.DataFrame([{'Ticker': 'AAA', 'Quantity': 100, 'AvgCost': 10.0}])
        snapshot = pd.DataFrame([{'Ticker': 'AAA', 'Price': 20.0}])
        presets = pd.DataFrame([{'Ticker': 'AAA', 'MA50': 25.0, 'MA20': 24.0}])
        # RSI low to trigger MA-break path; use MACD/ticker_sent weight to push score above gate
        metrics = pd.DataFrame([{'Ticker': 'AAA', 'RSI14': 40.0, 'MACDHist': 0.5}])
        industry = pd.DataFrame({'Ticker': ['AAA'], 'Sector': ['X']})
        sector_strength = pd.DataFrame([])
        session_summary = pd.DataFrame([{'SessionPhase': 'afternoon', 'InVNSession': 1, 'IndexChangePct': 0.0}])
        t = self._base_tuning()
        t['thresholds']['exit_ma_break_min_phase'] = 'morning'
        t['thresholds']['exit_ma_break_score_gate'] = 0.2
        t['thresholds']['tilt_exit_downgrade_min'] = 1.0  # disable tilt downgrade path
        # Boost score via ticker sentiment channel without triggering tilt downgrade
        t['weights']['w_ticker_sent'] = 0.8
        t['ticker_bias'] = {'AAA': 1.0}
        act, scores, feats, regime = decide_actions(portfolio, snapshot, metrics, presets, industry, sector_strength, session_summary, t)
        self.assertIn('AAA', scores)
        # Because score >= gate, downgrade to TRIM (despite MA-break)
        self.assertEqual(act.get('AAA'), 'trim')
