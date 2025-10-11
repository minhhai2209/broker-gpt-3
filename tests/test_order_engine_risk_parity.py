import unittest
import pandas as pd
import numpy as np

from scripts.orders.order_engine import decide_actions, build_orders


def make_history(n=120, sigma_a=0.01, sigma_b=0.03, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    ret_a = rng.normal(0.0005, sigma_a, size=n)
    ret_b = rng.normal(0.0005, sigma_b, size=n)
    pa = 20.0 * (1 + pd.Series(ret_a)).cumprod()
    pb = 20.0 * (1 + pd.Series(ret_b)).cumprod()
    df = pd.DataFrame({
        'Date': list(dates) * 2,
        'Ticker': ['AAA'] * n + ['BBB'] * n,
        'Close': list(pa) + list(pb),
    })
    return df


def tuning_rp():
    return {
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
        'risk_off_trend_floor': 0.0,
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
            'softmax_tau': 0.6,
            'add_share': 1.0,
            'new_share': 1.0,
            'min_lot': 100,
            'risk_weighting': 'risk_parity',
            'risk_alpha': 1.0,
            'max_pos_frac': 1.0,
            'max_sector_frac': 1.0,
            'reuse_sell_proceeds_frac': 0.0,
            'leftover_redistribute': 1,
            'min_ticket_k': 0.0,
            'risk_blend': 1.0,
            'cov_lookback_days': 60,
            'cov_reg': 0.0000,
            'risk_parity_floor': 0.0,
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


class TestRiskParity(unittest.TestCase):
    def test_risk_parity_prefers_lower_vol(self):
        portfolio = pd.DataFrame([], columns=['Ticker','Quantity','AvgCost'])
        snapshot = pd.DataFrame([
            {'Ticker': 'AAA', 'Price': 20.0},
            {'Ticker': 'BBB', 'Price': 20.0},
        ])
        metrics = pd.DataFrame([
            {'Ticker': 'AAA', 'RSI14': 60.0, 'Sector': 'X', 'MomRetNorm': 0.8, 'LiqNorm': 0.5},
            {'Ticker': 'BBB', 'RSI14': 60.0, 'Sector': 'X', 'MomRetNorm': 0.8, 'LiqNorm': 0.5},
        ])
        presets = pd.DataFrame([
            {'Ticker': 'AAA', 'MA20': 19.0, 'MA50': 18.0, 'BandFloor_Tick': 18.0, 'BandCeiling_Tick': 22.0},
            {'Ticker': 'BBB', 'MA20': 19.0, 'MA50': 18.0, 'BandFloor_Tick': 18.0, 'BandCeiling_Tick': 22.0},
        ])
        industry = pd.DataFrame({'Ticker': ['AAA','BBB'], 'Sector': ['X','X']})
        sector_strength = pd.DataFrame([
            {'sector': 'Tất cả', 'breadth_above_ma50_pct': 60.0, 'avg_rsi14': 55.0}
        ])
        session_summary = pd.DataFrame([{'SessionPhase': 'morning', 'InVNSession': 1, 'IndexChangePct': 0.4}])
        tuning = tuning_rp()
        actions, scores, feats, regime = decide_actions(portfolio, snapshot, metrics, presets, industry, sector_strength, session_summary, tuning)
        pnl_summary = pd.DataFrame([[100000.0, 100000.0, 0.0, 0.0]], columns=['TotalCost','TotalMarket','TotalPnL','ReturnPct'])
        ph = make_history(n=120, sigma_a=0.01, sigma_b=0.04)
        regime.market_score = 0.7
        regime.index_atr_percentile = 0.20
        regime.trend_strength = 0.05
        orders, notes, _ = build_orders(actions, portfolio, snapshot, metrics, presets, pnl_summary, scores, regime, ph)
        buys = [o for o in orders if o.side == 'BUY']
        self.assertTrue(buys and len({o.ticker for o in buys}) >= 2, 'Expected buys for both AAA and BBB under risk parity')
        for order in buys:
            self.assertEqual(order.quantity % 100, 0)
        q = {o.ticker: o.quantity for o in buys}
        self.assertGreaterEqual(q.get('AAA', 0), 100)
        self.assertGreaterEqual(q.get('BBB', 0), 100)


if __name__ == '__main__':
    unittest.main()
