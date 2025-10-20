import unittest
from pathlib import Path

import pandas as pd

from scripts.orders.order_engine import decide_actions, MarketRegime, build_orders


def make_tuning():
    return {
        'buy_budget_frac': 0.1,
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
            'sl_pct': 0.07,
            'tp_atr_mult': None,
            'sl_atr_mult': None,
            'tp_floor_pct': 0.0,
            'tp_cap_pct': 0.15,
            'sl_floor_pct': 0.0,
            'sl_cap_pct': 0.10,
            'tp_trim_frac': 0.30,
            'exit_on_ma_break': 0,
            'tp_sl_mode': 'legacy',
            'tp1_frac': 0.50,
            'tp1_hh_lookback': 10,
            'trail_hh_lookback': 22,
            'trail_atr_mult': 2.5,
            'be_buffer_pct': 0.0,
            'sl_trim_step_1_trigger': 0.5,
            'sl_trim_step_1_frac': 0.25,
            'sl_trim_step_2_trigger': 0.8,
            'sl_trim_step_2_frac': 0.35,
            # New thresholds used by engine; keep values benign for tests
            'exit_ma_break_rsi': 45.0,
            'trim_rsi_below_ma20': 45.0,
            'trim_rsi_macdh_neg': 40.0,
            'exit_ma_break_score_gate': 0.0,
            'tilt_exit_downgrade_min': 0.05,
        },
        'neutral_adaptive': {
            'neutral_enable': 0,
            'neutral_risk_on_prob_low': 0.35,
            'neutral_risk_on_prob_high': 0.65,
            'neutral_index_atr_soft_cap': None,
            'neutral_breadth_band': 0.05,
            'neutral_breadth_center': None,
            'neutral_base_new_scale': 0.90,
            'neutral_base_new_floor': 0.80,
            'neutral_base_add_scale': 0.90,
            'neutral_base_add_floor': 0.80,
            'partial_threshold_ratio': 0.75,
            'partial_entry_frac': 0.30,
            'partial_allow_leftover': 0,
            'min_new_per_day': 1,
            'max_new_overrides_per_day': 1,
            'add_max_neutral_cap': 1,
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
            'leader_min_rsi': 55.0,
            'leader_min_mom_norm': 0.6,
            'leader_require_ma20': 1,
            'leader_require_ma50': 1,
            'leader_max': 2,
            'risk_off_drawdown_floor': 0.40,
            # New hard-guard params (defaults aligned with prior behavior)
            'idx_chg_smoothed_hard_drop': 0.5,
            'trend_norm_hard_floor': -0.25,
            'vol_ann_hard_ceiling': 0.60,
            'index_atr_soft_pct': 0.80,
            'index_atr_hard_pct': 0.95,
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
        'execution': {
            'stop_ttl_min': 3,
            'slip_pct_min': 0.002,
            'slip_atr_mult': 0.5,
            'slip_ticks_min': 1,
            'flash_k_atr': 1.5,
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
            'cov_lookback_days': 90,
            'cov_reg': 0.0005,
            'risk_parity_floor': 0.2,
            'risk_per_trade_frac': 0.005,
            'default_stop_atr_mult': 2.15,
            'tranche_frac': 0.25,
            'dynamic_caps': {
                'enable': 0,
                'pos_min': 0.10,
                'pos_max': 0.16,
                'sector_min': 0.28,
                'sector_max': 0.40,
                'blend': 1.0,
                'override_static': 0,
            }
        },
    }


class TestDecideActions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._ensure_vnindex_history()

    @staticmethod
    def _ensure_vnindex_history():
        out_dir = Path('out')
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / 'prices_history.csv'
        # Provide >=200 sessions so regime trend/vol computations have robust data.
        dates = pd.date_range('2023-01-02', periods=240, freq='B')
        df = pd.DataFrame({
            'Date': dates.strftime('%Y-%m-%d'),
            'Ticker': ['VNINDEX'] * len(dates),
            'Close': [1000.0 + idx for idx in range(len(dates))],
        })
        df.to_csv(path, index=False)

    @staticmethod
    def _make_prices_history(highs: list[float], ticker: str = 'AAA') -> pd.DataFrame:
        dates = pd.date_range('2023-03-01', periods=len(highs), freq='B')
        rows = []
        for idx, (d, h) in enumerate(zip(dates, highs)):
            close = h - 0.1 if h > 0.1 else h
            rows.append({
                'Date': d.strftime('%Y-%m-%d'),
                'Ticker': ticker,
                'High': float(h),
                'Close': float(close),
            })
        for idx, d in enumerate(dates):
            rows.append({
                'Date': d.strftime('%Y-%m-%d'),
                'Ticker': 'VNINDEX',
                'High': 1000.0 + idx * 0.5 + 5.0,
                'Close': 1000.0 + idx * 0.5,
            })
        return pd.DataFrame(rows)

    def test_quantile_gate_filters_low_adds(self):
        # Two holdings AAA (low RSI -> low score), BBB (RSI 60 -> high score)
        portfolio = pd.DataFrame([
            {'Ticker': 'AAA', 'Quantity': 100, 'AvgCost': 10.0},
            {'Ticker': 'BBB', 'Quantity': 100, 'AvgCost': 10.0},
        ])
        snapshot = pd.DataFrame([
            {'Ticker': 'AAA', 'Price': 10.9},
            {'Ticker': 'BBB', 'Price': 10.9},
        ])
        metrics = pd.DataFrame([
            {'Ticker': 'AAA', 'RSI14': 20.0, 'Sector': 'X'},
            {'Ticker': 'BBB', 'RSI14': 60.0, 'Sector': 'X'},
        ])
        presets = pd.DataFrame([])
        industry = pd.DataFrame({'Ticker': ['AAA','BBB'], 'Sector': ['X','X']})
        sector_strength = pd.DataFrame([
            {'sector': 'Tất cả', 'breadth_above_ma50_pct': 75.0, 'avg_rsi14': 55.0}
        ])
        session_summary = pd.DataFrame([{'SessionPhase': 'morning', 'InVNSession': 1, 'IndexChangePct': 0.0}])
        tuning = make_tuning()

        actions, scores, feats, regime = decide_actions(
            portfolio, snapshot, metrics, presets, industry, sector_strength, session_summary, tuning
        )
        # Expect only BBB remains 'add' after quantile gate
        self.assertEqual(actions.get('BBB'), 'add')
        self.assertIn('AAA', actions)
        # AAA should not be 'add' (could be 'hold' or 'trim' due to low RSI rules)
        self.assertNotEqual(actions.get('AAA'), 'add')

    def test_guard_near_ceiling_removes_new(self):
        # CCC is not held; will be considered new but near ceiling -> removed
        portfolio = pd.DataFrame([
            {'Ticker': 'AAA', 'Quantity': 100, 'AvgCost': 10.0},
        ])
        snapshot = pd.DataFrame([
            {'Ticker': 'AAA', 'Price': 20.0},
            # Ceiling ~ RefPrice*1.07 = 107, so 106 is within 98% of ceiling
            {'Ticker': 'CCC', 'Price': 106.0},
        ])
        metrics = pd.DataFrame([
            {'Ticker': 'AAA', 'RSI14': 60.0},
            {'Ticker': 'CCC', 'RSI14': 60.0},
        ])
        presets = pd.DataFrame([
            {'Ticker': 'CCC', 'RefPrice': 100.0},
        ])
        industry = pd.DataFrame({'Ticker': ['AAA','CCC'], 'Sector': ['X','X']})
        sector_strength = pd.DataFrame([])
        session_summary = pd.DataFrame([{'SessionPhase': 'morning', 'InVNSession': 1, 'IndexChangePct': 0.0}])
        tuning = make_tuning()

        actions, scores, feats, regime = decide_actions(
            portfolio, snapshot, metrics, presets, industry, sector_strength, session_summary, tuning
        )
        # If near ceiling guard works, CCC should not be in actions as 'new'
        self.assertNotEqual(actions.get('CCC'), 'new')

    def test_stop_final_exit_generates_stop_order(self):
        portfolio = pd.DataFrame([
            {'Ticker': 'AAA', 'Quantity': 1000, 'AvgCost': 10.0},
        ])
        snapshot = pd.DataFrame([
            {'Ticker': 'AAA', 'Price': 9.0, 'Open': 9.2, 'High': 9.5},
        ])
        metrics = pd.DataFrame([
            {
                'Ticker': 'AAA',
                'RSI14': 55.0,
                'ATR14_Pct': 2.0,
                'LiqNorm': 0.8,
                'Beta60D': 1.0,
                'MomRetNorm': 0.6,
                'AvgTurnover20D_k': 5000.0,
                'Sector': 'X',
            }
        ])
        presets = pd.DataFrame([
            {
                'Ticker': 'AAA',
                'BandFloor_Tick': 8.0,
                'BandCeiling_Tick': 12.0,
                'ATR14': 0.2,
                'MA20': 10.0,
                'MA50': 10.5,
            }
        ])
        industry = pd.DataFrame({'Ticker': ['AAA'], 'Sector': ['X']})
        sector_strength = pd.DataFrame([])
        session_summary = pd.DataFrame([{'SessionPhase': 'morning', 'InVNSession': 1, 'IndexChangePct': 0.0}])
        tuning = make_tuning()
        prices_history = self._make_prices_history([10.5 + 0.05 * i for i in range(30)])

        actions, scores, feats, regime = decide_actions(
            portfolio, snapshot, metrics, presets, industry, sector_strength, session_summary, tuning, prices_history=prices_history
        )
        self.assertEqual(actions.get('AAA'), 'exit')
        meta = getattr(regime, 'stateless_sell_meta', {}).get('AAA', {})
        self.assertTrue(meta.get('stop_order'))
        regime_pricing = dict(getattr(regime, 'pricing', {}) or {})
        regime_pricing.setdefault('tc_sell_tax_frac', 0.0)
        regime.pricing = regime_pricing
        pnl_summary = pd.DataFrame([{'TotalMarket': 9000.0}])
        orders, notes, regime_out = build_orders(
            actions,
            portfolio,
            snapshot,
            metrics,
            presets,
            pnl_summary,
            scores,
            regime,
            prices_history,
        )
        sell_orders = [o for o in orders if o.ticker == 'AAA' and o.side == 'SELL']
        self.assertEqual(len(sell_orders), 1)
        # Engine clamps SELL limit up to market when below market
        mkt = float(snapshot.loc[snapshot['Ticker'] == 'AAA', 'Price'].iloc[0])
        self.assertAlmostEqual(sell_orders[0].limit_price, mkt, places=2)
        self.assertIn('STOP_FINAL', sell_orders[0].note)
        self.assertEqual(regime_out.ttl_overrides.get('AAA'), 3)

    def test_tp1_partial_sell_uses_tp_fraction(self):
        portfolio = pd.DataFrame([
            {'Ticker': 'AAA', 'Quantity': 1000, 'AvgCost': 10.0},
        ])
        snapshot = pd.DataFrame([
            {'Ticker': 'AAA', 'Price': 11.0, 'Open': 10.8, 'High': 11.2},
        ])
        metrics = pd.DataFrame([
            {
                'Ticker': 'AAA',
                'RSI14': 65.0,
                'ATR14_Pct': 2.0,
                'LiqNorm': 0.8,
                'Beta60D': 1.0,
                'MomRetNorm': 0.7,
                'AvgTurnover20D_k': 6000.0,
                'Sector': 'X',
            }
        ])
        presets = pd.DataFrame([
            {
                'Ticker': 'AAA',
                'BandFloor_Tick': 9.0,
                'BandCeiling_Tick': 13.0,
                'ATR14': 0.2,
                'MA20': 10.5,
                'MA50': 10.0,
            }
        ])
        industry = pd.DataFrame({'Ticker': ['AAA'], 'Sector': ['X']})
        sector_strength = pd.DataFrame([])
        session_summary = pd.DataFrame([{'SessionPhase': 'morning', 'InVNSession': 1, 'IndexChangePct': 0.0}])
        tuning = make_tuning()
        prices_history = self._make_prices_history([9.5 + 0.05 * i for i in range(29)] + [11.2])

        actions, scores, feats, regime = decide_actions(
            portfolio, snapshot, metrics, presets, industry, sector_strength, session_summary, tuning, prices_history=prices_history
        )
        self.assertEqual(actions.get('AAA'), 'take_profit')
        regime_pricing = dict(getattr(regime, 'pricing', {}) or {})
        regime_pricing.setdefault('tc_sell_tax_frac', 0.0)
        regime.pricing = regime_pricing
        pnl_summary = pd.DataFrame([{'TotalMarket': 11000.0}])
        orders, notes, regime_out = build_orders(
            actions,
            portfolio,
            snapshot,
            metrics,
            presets,
            pnl_summary,
            scores,
            regime,
            prices_history,
        )
        sell_orders = [o for o in orders if o.ticker == 'AAA' and o.side == 'SELL']
        self.assertEqual(len(sell_orders), 1)
        self.assertEqual(sell_orders[0].quantity, 500)
        self.assertTrue(any(tag in sell_orders[0].note for tag in ('TP1_BREAKOUT', 'TP1_ATR')))

    def test_trim_when_momentum_weak(self):
        portfolio = pd.DataFrame([
            {'Ticker': 'AAA', 'Quantity': 1000, 'AvgCost': 10.0},
        ])
        snapshot = pd.DataFrame([
            {'Ticker': 'AAA', 'Price': 9.8, 'Open': 9.9, 'High': 9.92},
        ])
        metrics = pd.DataFrame([
            {
                'Ticker': 'AAA',
                'RSI14': 40.0,
                'ATR14_Pct': 1.0,
                'LiqNorm': 0.8,
                'Beta60D': 1.0,
                'MomRetNorm': 0.3,
                'AvgTurnover20D_k': 4000.0,
                'Sector': 'X',
            }
        ])
        presets = pd.DataFrame([
            {
                'Ticker': 'AAA',
                'BandFloor_Tick': 9.0,
                'BandCeiling_Tick': 12.0,
                'ATR14': 0.15,
                'MA20': 10.2,
                'MA50': 10.5,
            }
        ])
        industry = pd.DataFrame({'Ticker': ['AAA'], 'Sector': ['X']})
        sector_strength = pd.DataFrame([])
        session_summary = pd.DataFrame([{'SessionPhase': 'morning', 'InVNSession': 1, 'IndexChangePct': 0.0}])
        tuning = make_tuning()
        prices_history = self._make_prices_history([9.4 + 0.02 * i for i in range(30)])

        actions, scores, feats, regime = decide_actions(
            portfolio, snapshot, metrics, presets, industry, sector_strength, session_summary, tuning, prices_history=prices_history
        )
        self.assertEqual(actions.get('AAA'), 'trim')
        regime_pricing = dict(getattr(regime, 'pricing', {}) or {})
        regime_pricing.setdefault('tc_sell_tax_frac', 0.0)
        regime.pricing = regime_pricing
        pnl_summary = pd.DataFrame([{'TotalMarket': 9800.0}])
        orders, notes, regime_out = build_orders(
            actions,
            portfolio,
            snapshot,
            metrics,
            presets,
            pnl_summary,
            scores,
            regime,
            prices_history,
        )
        sell_orders = [o for o in orders if o.ticker == 'AAA' and o.side == 'SELL']
        self.assertEqual(len(sell_orders), 1)
        self.assertEqual(sell_orders[0].quantity, 300)
        self.assertIn('MOM_WEAK', sell_orders[0].note)

    def test_market_filter_blocks_non_leader_when_index_drops(self):
        portfolio = pd.DataFrame([], columns=['Ticker', 'Quantity', 'AvgCost'])
        snapshot = pd.DataFrame([
            {'Ticker': 'LEAD', 'Price': 25.0},
            {'Ticker': 'LAG', 'Price': 18.0},
        ])
        metrics = pd.DataFrame([
            {'Ticker': 'LEAD', 'RSI14': 60.0, 'MomRetNorm': 0.8},
            {'Ticker': 'LAG', 'RSI14': 45.0, 'MomRetNorm': 0.3},
        ])
        presets = pd.DataFrame([
            {'Ticker': 'LEAD', 'MA20': 24.0, 'MA50': 23.0},
            {'Ticker': 'LAG', 'MA20': 19.0, 'MA50': 19.5},
        ])
        industry = pd.DataFrame({'Ticker': ['LEAD', 'LAG'], 'Sector': ['X', 'X']})
        sector_strength = pd.DataFrame([
            {'sector': 'Tất cả', 'breadth_above_ma50_pct': 10.0, 'avg_rsi14': 40.0}
        ])
        session_summary = pd.DataFrame([
            {'SessionPhase': 'morning', 'InVNSession': 1, 'IndexChangePct': -1.0}
        ])
        tuning = make_tuning()
        tuning['thresholds']['q_new'] = 0.0

        actions, scores, feats, regime = decide_actions(
            portfolio, snapshot, metrics, presets, industry, sector_strength, session_summary, tuning
        )
        self.assertNotEqual(actions.get('LEAD'), 'new')
        self.assertNotEqual(actions.get('LAG'), 'new')
        self.assertIn('LEAD', regime.debug_filters.get('market', []))
        self.assertIn('LAG', regime.debug_filters.get('market', []))
        self.assertEqual(regime.buy_budget_frac, 0.0)
        self.assertEqual(regime.new_max, 0)
        self.assertEqual(regime.add_max, 0)


if __name__ == '__main__':
    unittest.main()
