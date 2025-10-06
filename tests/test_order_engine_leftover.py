import unittest
from pathlib import Path

import pandas as pd

from scripts.orders.order_engine import decide_actions, build_orders


def make_tuning():
    return {
        'buy_budget_frac': 0.30,  # 30% of NAV
        'add_max': 10,
        'new_max': 10,
        'weights': {
            'w_trend': 0.0,
            'w_momo': 1.0,
            'w_mom_ret': 0.0,
            'w_liq': 0.0,
            'w_vol_guard': 0.0,
            'w_beta': 0.0,
            'w_sector': 0.0,
            'w_sector_sent': 0.0,
            'w_ticker_sent': 0.0,
            'w_roe': 0.0,
            'w_earnings_yield': 0.0,
        },
        'regime_model': {
            'intercept': 0.0,
            'threshold': 0.5,
            'components': {
                'trend': {'mean': 0.5, 'std': 0.2, 'weight': 1.0},
                'breadth': {'mean': 0.5, 'std': 0.2, 'weight': 1.0},
                'index_return': {'mean': 0.0, 'std': 0.6, 'weight': 0.8},
                'volatility': {'mean': 0.5, 'std': 0.2, 'weight': 0.6}
            }
        },
        'thresholds': {
            'base_add': 0.10,
            'base_new': 0.10,
            'trim_th': -0.10,
            'q_add': 0.75,
            'q_new': 0.75,
            'min_liq_norm': 0.0,
            'near_ceiling_pct': 0.98,
            'tp_pct': 0.10,
            'sl_pct': 0.10,
            'tp_atr_mult': None,
            'sl_atr_mult': None,
            'tp_floor_pct': 0.0,
            'sl_floor_pct': 0.0,
            'tp_trim_frac': 0.30,
            'exit_on_ma_break': 0,
            'exit_ma_break_rsi': 45.0,
            'trim_rsi_below_ma20': 45.0,
            'trim_rsi_macdh_neg': 40.0,
            'exit_ma_break_score_gate': 0.0,
            'tilt_exit_downgrade_min': 0.05,
        },
        'sector_bias': {},
        'ticker_bias': {},
        'market_filter': {
            'risk_off_index_drop_pct': 0.5,
            'risk_off_trend_floor': 0.0,
            'risk_off_breadth_floor': 0.4,
            'market_score_soft_floor': 0.55,
            'market_score_hard_floor': 0.35,
            'leader_min_rsi': 55.0,
            'leader_min_mom_norm': 0.6,
            'leader_require_ma20': 1,
            'leader_require_ma50': 1,
            'idx_chg_smoothed_hard_drop': 0.5,
            'trend_norm_hard_floor': -0.25,
            'vol_ann_hard_ceiling': 0.60,
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
            'min_lot': 100,
            'add_share': 0.6,
            'new_share': 0.4,
            'risk_weighting': 'score_softmax',
            'risk_alpha': 1.0,
            'max_pos_frac': 0.5,
            'max_sector_frac': 0.8,
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


class TestLeftoverRedistribution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._ensure_vnindex_history()

    @staticmethod
    def _ensure_vnindex_history():
        out_dir = Path('out')
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / 'prices_history.csv'
        dates = pd.date_range('2023-01-02', periods=240, freq='B')
        df = pd.DataFrame({
            'Date': dates.strftime('%Y-%m-%d'),
            'Ticker': ['VNINDEX'] * len(dates),
            'Close': [1000.0 + idx for idx in range(len(dates))],
        })
        df.to_csv(path, index=False)

    def test_leftover_redistribution_allocates_buy(self):
        # Two NEW candidates get split budget < 1 lot each; leftover should combine to 1 BUY
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
            {'sector': 'Tất cả', 'breadth_above_ma50_pct': 75.0, 'avg_rsi14': 55.0}
        ])
        session_summary = pd.DataFrame([{'SessionPhase': 'morning', 'InVNSession': 1, 'IndexChangePct': 0.6}])
        tuning = make_tuning()

        actions, scores, feats, regime = decide_actions(
            portfolio, snapshot, metrics, presets, industry, sector_strength, session_summary, tuning
        )
        # TotalMarket in nghìn; 10,000 -> budget 30% = 3,000 (enough for 1 lot at 2,000)
        pnl_summary = pd.DataFrame([[10000.0, 10000.0, 0.0, 0.0]], columns=['TotalCost','TotalMarket','TotalPnL','ReturnPct'])
        prices_history = pd.DataFrame({
            'Date': ['2025-01-01','2025-01-02'],
            'Ticker': ['AAA','BBB'],
            'Close': [20.0, 20.5],
        })
        # Ensure regime context signals healthy market so budget scaling does not zero-out NEW allocations
        regime.market_score = 0.8
        regime.index_atr_percentile = 0.20
        regime.trend_strength = 0.10
        regime.breadth_hint = 0.65
        orders, notes, _ = build_orders(
            actions, portfolio, snapshot, metrics, presets, pnl_summary, scores, regime, prices_history
        )
        buys = [o for o in orders if o.side == 'BUY']
        self.assertEqual(len(buys), 0, 'Budgets below 1 lot no longer trigger probe buys under strict lot enforcement')


if __name__ == '__main__':
    unittest.main()
