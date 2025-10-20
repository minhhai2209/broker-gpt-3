import unittest
import pandas as pd

from scripts.orders.order_engine import decide_actions, build_orders


def base_tuning_auto():
    return {
        'buy_budget_frac': 0.0,  # baseline 0 – rely on auto budget
        'add_max': 5,
        'new_max': 5,
        'weights': {
            'w_trend': 0.2, 'w_momo': 0.4, 'w_mom_ret': 0.0,
            'w_liq': 0.2, 'w_vol_guard': 0.0, 'w_beta': 0.0,
            'w_sector': 0.0, 'w_sector_sent': 0.0, 'w_ticker_sent': 0.0,
            'w_roe': 0.0, 'w_earnings_yield': 0.0, 'w_rs': 0.0,
        },
        'regime_model': {
            'intercept': 0.0, 'threshold': 0.5,
            'components': {
                'trend': {'mean': 0.5, 'std': 0.2, 'weight': 1.0},
                'breadth': {'mean': 0.5, 'std': 0.2, 'weight': 1.0},
                'index_return': {'mean': 0.0, 'std': 0.6, 'weight': 0.8},
                'volatility': {'mean': 0.5, 'std': 0.2, 'weight': 0.6},
            },
        },
        'thresholds': {
            'base_add': 0.0, 'base_new': 0.0, 'trim_th': -0.10,
            'q_add': 0.0, 'q_new': 0.0, 'min_liq_norm': 0.0,
            'near_ceiling_pct': 0.999,
            'tp_pct': 0.0, 'sl_pct': 0.10,  # 10% SL to ensure finite stop distance
            'tp_atr_mult': None, 'sl_atr_mult': None,
            'tp_floor_pct': 0.0, 'sl_floor_pct': 0.0,
            'tp_trim_frac': 0.30, 'exit_on_ma_break': 0,
            'exit_ma_break_rsi': 45.0, 'trim_rsi_below_ma20': 45.0, 'trim_rsi_macdh_neg': 40.0,
            'cooldown_days': 0, 'exit_ma_break_score_gate': 0.0, 'tilt_exit_downgrade_min': 0.05,
        },
        'sector_bias': {}, 'ticker_bias': {},
        'market_filter': {
            'risk_off_index_drop_pct': 0.5, 'risk_off_trend_floor': -0.15,
            'risk_off_breadth_floor': 0.2, 'breadth_relax_margin': 0.0,
            'market_score_soft_floor': 0.55, 'market_score_hard_floor': 0.20,
            'leader_min_rsi': 0.0, 'leader_min_mom_norm': 0.0,
            'leader_require_ma20': 0, 'leader_require_ma50': 0, 'leader_max': 10,
            'risk_off_drawdown_floor': 0.99, 'index_atr_soft_pct': 0.95, 'index_atr_hard_pct': 0.99,
            'idx_chg_smoothed_hard_drop': 0.05, 'trend_norm_hard_floor': -0.25,
            'vol_ann_hard_ceiling': 0.60, 'guard_new_scale_cap': 1.0, 'atr_soft_scale_cap': 1.0,
            'severe_drop_mult': 1.5,
        },
        'pricing': {
            'risk_on_buy': ["Aggr"], 'risk_on_sell': ["Cons"],
            'risk_off_buy': ["Cons"], 'risk_off_sell': ["Cons"],
            'atr_fallback_buy_mult': 0.25,
            'tc_roundtrip_frac': 0.0, 'tc_sell_tax_frac': 0.001,
            'atr_fallback_sell_mult': 0.25,
            'fill_prob': {
                'base': 0.3, 'cross': 0.9, 'near_ceiling': 0.05,
                'min': 0.05, 'decay_scale_min_ticks': 5.0,
                'partial_fill_kappa': 0.65, 'min_fill_notional_vnd': 2_000_000.0,
            },
            'slippage_model': {
                'alpha_bps': 5.0, 'beta_dist_per_tick': 1.0, 'beta_size': 40.0,
                'beta_vol': 8.0, 'mae_bps': 10.0, 'last_fit_date': None,
            },
        },
        'sizing': {
            'softmax_tau': 0.6, 'add_share': 1.0, 'new_share': 1.0,
            'min_lot': 100, 'risk_weighting': 'score_softmax', 'risk_alpha': 1.0,
            'max_pos_frac': 1.0, 'max_sector_frac': 1.0,
            'reuse_sell_proceeds_frac': 0.0,
            'leftover_redistribute': 1, 'min_ticket_k': 0.0, 'risk_blend': 1.0,
            'cov_lookback_days': 60, 'cov_reg': 0.0005, 'risk_parity_floor': 0.2,
            'dynamic_caps': {'enable': 0, 'pos_min': 0.10, 'pos_max': 0.16, 'sector_min': 0.28, 'sector_max': 0.40, 'blend': 1.0, 'override_static': 0},
            'allocation_model': 'softmax',
            'market_index_symbol': 'VNINDEX',
            # Risk-per-trade inputs
            'risk_per_trade_frac': 0.01, 'default_stop_atr_mult': 2.0,
            'tranche_frac': 1.0, 'qty_min_lot': 100, 'min_notional_per_order': 2_000_000.0,
            'new_first_tranche_lots': 10,
            # Auto budget toggles
            'auto_budget_enable': 1, 'auto_budget_cap_frac': 0.20, 'auto_budget_min_k': 0.0, 'auto_budget_mode': 'rpt',
        },
    }


class TestAutoBudget(unittest.TestCase):
    def test_auto_budget_produces_buys_with_zero_base_budget(self):
        portfolio = pd.DataFrame([], columns=['Ticker','Quantity','AvgCost'])
        snapshot = pd.DataFrame([
            {'Ticker': 'AAA', 'Price': 50.0},
            {'Ticker': 'VNINDEX', 'Price': 1100.0},
        ])
        metrics = pd.DataFrame([
            {'Ticker': 'AAA', 'RSI14': 60.0, 'Sector': 'X', 'MomRetNorm': 0.8, 'LiqNorm': 0.5, 'ATR14_Pct': 4.0},
            {'Ticker': 'VNINDEX', 'RSI14': 55.0},
        ])
        presets = pd.DataFrame([
            {'Ticker': 'AAA', 'MA20': 49.0, 'MA50': 45.0, 'BandFloor_Tick': 45.0, 'BandCeiling_Tick': 55.0},
        ])
        industry = pd.DataFrame({'Ticker': ['AAA'], 'Sector': ['X']})
        sector_strength = pd.DataFrame([
            {'sector': 'Tất cả', 'breadth_above_ma50_pct': 60.0, 'avg_rsi14': 55.0}
        ])
        session_summary = pd.DataFrame([{'SessionPhase': 'morning', 'InVNSession': 1, 'IndexChangePct': 0.4}])
        tuning = base_tuning_auto()
        actions, scores, feats, regime = decide_actions(portfolio, snapshot, metrics, presets, industry, sector_strength, session_summary, tuning)
        # Large NAV to make budget room
        pnl_summary = pd.DataFrame([[100000.0, 100000.0, 0.0, 0.0]], columns=['TotalCost','TotalMarket','TotalPnL','ReturnPct'])
        prices_history = pd.DataFrame({'Date': ['2024-01-02'], 'Ticker': ['AAA'], 'Close': [50.0]})
        orders, notes, regime2 = build_orders(actions, portfolio, snapshot, metrics, presets, pnl_summary, scores, regime, prices_history)
        buys = [o for o in orders if o.side == 'BUY']
        self.assertTrue(buys, 'Expected BUY orders under auto budget mode with base budget 0')
        # Effective fraction should reflect cap (<= auto_budget_cap_frac)
        eff_frac = float(getattr(regime2, 'buy_budget_frac', 0.0) or 0.0)
        self.assertGreater(eff_frac, 0.0)
        self.assertLessEqual(eff_frac, tuning['sizing']['auto_budget_cap_frac'] + 1e-6)


if __name__ == '__main__':
    unittest.main()
