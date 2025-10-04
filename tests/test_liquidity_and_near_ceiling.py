import unittest
from pathlib import Path
import pandas as pd

from scripts.order_engine import decide_actions


class TestLiquidityAndNearCeiling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Minimal VNINDEX history to satisfy regime
        out_dir = Path('out')
        out_dir.mkdir(parents=True, exist_ok=True)
        dates = pd.date_range('2023-01-02', periods=220, freq='B')
        df = pd.DataFrame({
            'Date': dates.strftime('%Y-%m-%d'),
            'Ticker': ['VNINDEX'] * len(dates),
            'Close': [1000.0 + idx for idx in range(len(dates))],
        })
        df.to_csv(out_dir / 'prices_history.csv', index=False)

    def _base_tuning(self):
        return {
            'buy_budget_frac': 0.10,
            'add_max': 10,
            'new_max': 10,
            'weights': {
                'w_trend': 0.0, 'w_momo': 1.0, 'w_mom_ret': 0.0, 'w_liq': 0.0,
                'w_vol_guard': 0.0, 'w_beta': 0.0, 'w_sector': 0.0, 'w_sector_sent': 0.0,
                'w_ticker_sent': 0.0, 'w_roe': 0.0, 'w_earnings_yield': 0.0, 'w_rs': 0.0,
            },
            'regime_model': {
                'intercept': 0.0, 'threshold': 0.5,
                'components': {
                    'trend': {'mean': 0.5, 'std': 0.2, 'weight': 1.0},
                    'breadth': {'mean': 0.5, 'std': 0.2, 'weight': 1.0},
                    'index_return': {'mean': 0.0, 'std': 0.6, 'weight': 0.8},
                    'volatility': {'mean': 0.5, 'std': 0.2, 'weight': 0.6},
                }
            },
            'thresholds': {
                'base_add': 0.10, 'base_new': 0.10, 'trim_th': -0.10,
                'q_add': 0.0, 'q_new': 0.0,
                'min_liq_norm': 0.50, 'near_ceiling_pct': 0.98,
                'tp_pct': 10.0, 'sl_pct': 0.50,
                'tp_atr_mult': None, 'sl_atr_mult': None,
                'tp_floor_pct': 0.0, 'sl_floor_pct': 0.0,
                'tp_trim_frac': 0.30,
                'exit_on_ma_break': 0,
                'exit_ma_break_rsi': 45.0,
                'trim_rsi_below_ma20': 45.0,
                'trim_rsi_macdh_neg': 40.0,
                'exit_ma_break_score_gate': 0.0,
                'tilt_exit_downgrade_min': 0.05,
                'cooldown_days': 0,
            },
            'sector_bias': {},
            'ticker_bias': {},
            'pricing': {
                'risk_on_buy': ["Aggr","Bal","Cons","MR","Break"],
                'risk_on_sell': ["Cons","Bal","Break","MR","Aggr"],
                'risk_off_buy': ["Cons","MR","Bal","Aggr","Break"],
                'risk_off_sell': ["MR","Cons","Bal","Aggr","Break"],
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
                'dynamic_caps': {'enable': 0, 'pos_min': 0.10, 'pos_max': 0.20, 'sector_min': 0.28, 'sector_max': 0.40, 'blend': 1.0, 'override_static': 0},
            },
            'market_filter': {
                'risk_off_index_drop_pct': 0.5,
                'risk_off_trend_floor': 0.0,
                'risk_off_breadth_floor': 0.0,
                'market_score_soft_floor': 0.55,
                'market_score_hard_floor': 0.35,
                'leader_min_rsi': 0.0,
                'leader_min_mom_norm': 0.0,
                'leader_require_ma20': 0,
                'leader_require_ma50': 0,
                'leader_max': 10,
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

    def test_min_liq_norm_blocks_new(self):
        portfolio = pd.DataFrame([{'Ticker': 'AAA', 'Quantity': 100, 'AvgCost': 10.0}])
        snapshot = pd.DataFrame([
            {'Ticker': 'AAA', 'Price': 20.0},
            {'Ticker': 'CCC', 'Price': 50.0},
        ])
        metrics = pd.DataFrame([
            {'Ticker': 'AAA', 'RSI14': 60.0, 'LiqNorm': 0.9},
            {'Ticker': 'CCC', 'RSI14': 60.0, 'LiqNorm': 0.10},  # below min_liq_norm=0.50
        ])
        presets = pd.DataFrame([])
        industry = pd.DataFrame({'Ticker': ['AAA','CCC'], 'Sector': ['X','X']})
        sector_strength = pd.DataFrame([])
        session_summary = pd.DataFrame([{'SessionPhase': 'morning', 'InVNSession': 1, 'IndexChangePct': 0.0}])
        tuning = self._base_tuning()
        actions, scores, feats, regime = decide_actions(portfolio, snapshot, metrics, presets, industry, sector_strength, session_summary, tuning)
        self.assertNotEqual(actions.get('CCC'), 'new')
        self.assertIn('CCC', (regime.debug_filters.get('liquidity') or []))

    def test_near_ceiling_blocks_add_with_ref_fallback(self):
        # AAA is held and would be 'add' but near ceiling should downgrade to 'hold'
        portfolio = pd.DataFrame([{'Ticker': 'AAA', 'Quantity': 100, 'AvgCost': 10.0}])
        # RefPrice=100 => ceil≈107; near_ceiling_pct=0.98 => 104.86; set price 105
        snapshot = pd.DataFrame([{'Ticker': 'AAA', 'Price': 105.0}])
        metrics = pd.DataFrame([{'Ticker': 'AAA', 'RSI14': 65.0, 'LiqNorm': 1.0}])
        presets = pd.DataFrame([{'Ticker': 'AAA', 'RefPrice': 100.0}])  # no BandCeiling_Tick -> fallback to Ref*1.07
        industry = pd.DataFrame({'Ticker': ['AAA'], 'Sector': ['X']})
        sector_strength = pd.DataFrame([])
        session_summary = pd.DataFrame([{'SessionPhase': 'morning', 'InVNSession': 1, 'IndexChangePct': 0.0}])
        tuning = self._base_tuning()
        tuning['thresholds']['base_add'] = 0.2  # ensure qualifies add by score
        actions, scores, feats, regime = decide_actions(portfolio, snapshot, metrics, presets, industry, sector_strength, session_summary, tuning)
        self.assertEqual(actions.get('AAA'), 'hold')
        self.assertIn('AAA', (regime.debug_filters.get('near_ceiling') or []))
