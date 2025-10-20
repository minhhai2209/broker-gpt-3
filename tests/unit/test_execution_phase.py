import unittest
import pandas as pd

from scripts.orders.order_engine import MarketRegime, build_orders


class TestExecutionPhaseRules(unittest.TestCase):
    def test_atc_cross_override(self):
        # Minimal data
        portfolio = pd.DataFrame([], columns=['Ticker','Quantity','AvgCost'])
        snapshot = pd.DataFrame([
            {'Ticker': 'AAA', 'Price': 100.0},
        ])
        metrics = pd.DataFrame([
            {'Ticker': 'AAA', 'RSI14': 60.0, 'ATR14_Pct': 0.5, 'AvgTurnover20D_k': 5000.0},
        ])
        presets = pd.DataFrame([
            {'Ticker': 'AAA', 'BandFloor_Tick': 95.0, 'BandCeiling_Tick': 107.0},
        ])
        pnl_summary = pd.DataFrame([[10000000.0, 10000000.0, 0.0, 0.0]], columns=['TotalCost','TotalMarket','TotalPnL','ReturnPct'])
        prices_history = pd.DataFrame({'Date': ['2024-01-02','2024-01-03'], 'Ticker': ['VNINDEX','VNINDEX'], 'Close': [1000.0, 1001.0]})

        tuning = {
            'buy_budget_frac': 0.10,
            'add_max': 10,
            'new_max': 10,
            'weights': {
                'w_trend': 0.0, 'w_momo': 0.0, 'w_mom_ret': 0.0, 'w_liq': 0.0,
                'w_vol_guard': 0.0, 'w_beta': 0.0, 'w_sector': 0.0, 'w_sector_sent': 0.0,
                'w_ticker_sent': 0.0, 'w_roe': 0.0, 'w_earnings_yield': 0.0, 'w_rs': 0.0,
            },
            'thresholds': {
                'base_add': 0.0, 'base_new': 0.0, 'trim_th': -1.0,
                'q_add': 0.0, 'q_new': 0.0, 'min_liq_norm': 0.0,
                'near_ceiling_pct': 0.989,
                'tp_pct': 0.0, 'sl_pct': 1.0, 'tp_trim_frac': 0.3,
                'exit_on_ma_break': 0,
                'exit_ma_break_rsi': 45.0,
                'trim_rsi_below_ma20': 45.0,
                'trim_rsi_macdh_neg': 40.0,
                'cooldown_days': 0,
                'exit_ma_break_score_gate': 0.0,
                'tilt_exit_downgrade_min': 0.05,
            },
            'sector_bias': {}, 'ticker_bias': {},
            'pricing': {
                'risk_on_buy': ["Aggr","Bal","Cons","MR","Break"],
                'risk_on_sell': ["Cons","Bal","Break","MR","Aggr"],
                'risk_off_buy': ["Cons","MR","Bal","Aggr","Break"],
                'risk_off_sell': ["MR","Cons","Bal","Aggr","Break"],
                'atr_fallback_buy_mult': 0.25,
                'atr_fallback_sell_mult': 0.25,
                'tc_roundtrip_frac': 0.0,
                'tc_sell_tax_frac': 0.001,
            },
            'sizing': {
                'softmax_tau': 0.6, 'add_share': 1.0, 'new_share': 1.0,
                'min_lot': 100, 'risk_weighting': 'score_softmax', 'risk_alpha': 1.0,
                'max_pos_frac': 1.0, 'max_sector_frac': 1.0, 'reuse_sell_proceeds_frac': 0.0,
                'leftover_redistribute': 0, 'min_ticket_k': 0.0, 'risk_blend': 1.0,
                'cov_lookback_days': 60, 'cov_reg': 0.0003, 'risk_parity_floor': 0.0,
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
                'leader_require_ma20': 0,
                'leader_require_ma50': 0,
                'leader_max': 10,
                'risk_off_drawdown_floor': 0.40,
                'index_atr_soft_pct': 0.80,
                'index_atr_hard_pct': 0.95,
                'guard_new_scale_cap': 0.40,
                'atr_soft_scale_cap': 0.50,
                'severe_drop_mult': 1.50,
            },
            'execution': {
                'fill': {
                    'horizon_s': 60, 'window_sigma_s': 45, 'window_vol_s': 90,
                    'target_prob': 0.85, 'max_chase_ticks': 1, 'cancel_ratio_per_min': 0.5,
                    'joiner_factor': 0.05, 'no_cross': True,
                },
                'time_of_day': {
                    'phase_rules': {
                        'ATC': { 'allow_cross_if_target_prob_gte': 0.80 }
                    }
                }
            }
        }

        regime = MarketRegime(
            phase='ATC', in_session=True,
            index_change_pct=0.0, breadth_hint=0.5, risk_on=True,
            buy_budget_frac=tuning['buy_budget_frac'], top_sectors=[], add_max=10, new_max=10,
            weights=tuning['weights'], thresholds=tuning['thresholds'], sector_bias={}, ticker_bias={},
            pricing=tuning['pricing'], sizing=tuning['sizing'], execution=tuning['execution']
        )

        # One NEW order; ensure it is accepted (not skipped) due to cross override
        actions = {'AAA': 'new'}
        scores = {'AAA': 1.0}
        orders, _, _ = build_orders(actions, portfolio, snapshot, metrics, presets, pnl_summary, scores, regime, prices_history)
        self.assertGreaterEqual(len(orders), 1)
