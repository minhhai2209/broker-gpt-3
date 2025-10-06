import unittest
from pathlib import Path

import pandas as pd

from scripts.orders.order_engine import decide_actions, build_orders


class TestMarketFilterScalingParams(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Minimal VNINDEX history; no High/Low to keep ATR percentile at fallback 0.5
        out_dir = Path('out')
        out_dir.mkdir(parents=True, exist_ok=True)
        dates = pd.date_range('2024-01-02', periods=60, freq='B')
        df = pd.DataFrame({
            'Date': dates.strftime('%Y-%m-%d'),
            'Ticker': ['VNINDEX'] * len(dates),
            'Close': [1000.0 + i for i in range(len(dates))],
        })
        (out_dir / 'prices_history.csv').write_text(df.to_csv(index=False))

    def _base_frames(self):
        portfolio = pd.DataFrame([], columns=['Ticker','Quantity','AvgCost'])
        snapshot = pd.DataFrame([{'Ticker': 'VNINDEX', 'Price': 1100.0}])
        metrics = pd.DataFrame([], columns=['Ticker'])
        presets = pd.DataFrame([])
        industry = pd.DataFrame([], columns=['Ticker','Sector'])
        return portfolio, snapshot, metrics, presets, industry

    def _tuning_with_regime(self, market_filter: dict, buy_budget: float = 0.12):
        return {
            'buy_budget_frac': buy_budget,
            'add_max': 5,
            'new_max': 5,
            'weights': {
                'w_trend': 0,'w_momo': 0,'w_mom_ret': 0,'w_liq': 0,'w_vol_guard': 0,
                'w_beta': 0,'w_sector': 0,'w_sector_sent': 0,'w_ticker_sent': 0,
                'w_roe': 0,'w_earnings_yield': 0,'w_rs': 0,
            },
            # Force high market_score via large positive intercept
            'regime_model': {
                'intercept': 5.0,
                'threshold': 0.5,
                'components': {
                    'trend': {'mean': 0.5, 'std': 0.2, 'weight': 0.0},
                    'breadth': {'mean': 0.5, 'std': 0.2, 'weight': 0.0},
                    'index_return': {'mean': 0.0, 'std': 0.6, 'weight': 0.0},
                    'volatility': {'mean': 0.5, 'std': 0.2, 'weight': 0.0},
                }
            },
            'thresholds': {
                'base_add': 0.0,
                'base_new': 0.0,
                'trim_th': -1.0,
                'q_add': 0.0,
                'q_new': 0.0,
                'min_liq_norm': 0.0,
                'near_ceiling_pct': 0.98,
                'tp_pct': 0.0,
                'sl_pct': 1.0,
                'tp_atr_mult': None,
                'sl_atr_mult': None,
                'tp_floor_pct': 0.0,
                'sl_floor_pct': 0.0,
                'tp_trim_frac': 0.3,
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
                'softmax_tau': 0.6,
                'add_share': 0.6,
                'new_share': 0.4,
                'min_lot': 100,
                'risk_weighting': 'score_softmax',
                'risk_alpha': 1.0,
                'max_pos_frac': 0.5,
                'max_sector_frac': 0.8,
                'reuse_sell_proceeds_frac': 0.0,
                'leftover_redistribute': 1,
                'min_ticket_k': 0.0,
                'risk_blend': 1.0,
                'cov_lookback_days': 60,
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
            'market_filter': market_filter,
        }

    def test_atr_hard_sets_scale_zero(self):
        portfolio, snapshot, metrics, presets, industry = self._base_frames()
        sector_strength = pd.DataFrame([
            {'sector': 'Tất cả', 'breadth_above_ma50_pct': 80.0, 'avg_rsi14': 60.0}
        ])
        session_summary = pd.DataFrame([{'SessionPhase': 'morning', 'InVNSession': 1, 'IndexChangePct': 0.0}])
        # With no High/Low, index_atr_percentile≈0.5. Set hard below that to force scale=0
        mf = {
            'risk_off_index_drop_pct': 0.5,
            'risk_off_trend_floor': -1.0,
            'risk_off_breadth_floor': 0.2,
            'market_score_soft_floor': 0.55,
            'market_score_hard_floor': 0.35,
            'leader_min_rsi': 30.0,
            'leader_min_mom_norm': 0.0,
            'leader_require_ma20': 0,
            'leader_require_ma50': 0,
            'leader_max': 0,
            'guard_new_scale_cap': 0.90,
            'atr_soft_scale_cap': 0.70,
            'severe_drop_mult': 1.50,
            'index_atr_soft_pct': 0.10,
            'index_atr_hard_pct': 0.40,
        }
        tuning = self._tuning_with_regime(mf, buy_budget=0.12)
        actions, scores, feats, regime = decide_actions(
            portfolio, snapshot, metrics, presets, industry, sector_strength, session_summary, tuning
        )
        pnl_summary = pd.DataFrame([[0.0, 0.0, 0.0, 0.0]], columns=['TotalCost','TotalMarket','TotalPnL','ReturnPct'])
        prices_history = pd.DataFrame({'Date':['2024-01-02'],'Ticker':['AAA'],'Close':[20.0]})
        _orders, _notes, regime = build_orders(actions, portfolio, snapshot, metrics, presets, pnl_summary, scores, regime, prices_history)
        self.assertAlmostEqual(float(regime.buy_budget_frac_effective), 0.0, places=6)

    def test_atr_soft_scale_cap_applies(self):
        portfolio, snapshot, metrics, presets, industry = self._base_frames()
        # Strong breadth; no guard_new. ATR soft cap should apply since percentile ~0.5 >= 0.4
        sector_strength = pd.DataFrame([
            {'sector': 'Tất cả', 'breadth_above_ma50_pct': 80.0, 'avg_rsi14': 60.0}
        ])
        session_summary = pd.DataFrame([{'SessionPhase': 'morning', 'InVNSession': 1, 'IndexChangePct': 0.0}])
        mf = {
            'risk_off_index_drop_pct': 0.5,
            'risk_off_trend_floor': -1.0,
            'risk_off_breadth_floor': 0.2,
            'market_score_soft_floor': 0.55,
            'market_score_hard_floor': 0.35,
            'leader_min_rsi': 30.0,
            'leader_min_mom_norm': 0.0,
            'leader_require_ma20': 0,
            'leader_require_ma50': 0,
            'leader_max': 0,
            'guard_new_scale_cap': 0.90,
            'atr_soft_scale_cap': 0.70,
            'severe_drop_mult': 1.50,
            'index_atr_soft_pct': 0.40,
            'index_atr_hard_pct': 0.90,
        }
        tuning = self._tuning_with_regime(mf, buy_budget=0.12)
        actions, scores, feats, regime = decide_actions(
            portfolio, snapshot, metrics, presets, industry, sector_strength, session_summary, tuning
        )
        pnl_summary = pd.DataFrame([[0.0, 0.0, 0.0, 0.0]], columns=['TotalCost','TotalMarket','TotalPnL','ReturnPct'])
        prices_history = pd.DataFrame({'Date':['2024-01-02'],'Ticker':['AAA'],'Close':[20.0]})
        _orders, _notes, regime = build_orders(actions, portfolio, snapshot, metrics, presets, pnl_summary, scores, regime, prices_history)
        self.assertAlmostEqual(float(regime.buy_budget_frac_effective), 0.12 * 0.70, places=6)


if __name__ == '__main__':
    unittest.main()
