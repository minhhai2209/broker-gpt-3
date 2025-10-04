import unittest
from pathlib import Path

import pandas as pd

from scripts.order_engine import MarketRegime, build_orders


def base_tuning():
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


class TestDynamicCapsEnable(unittest.TestCase):
    def _setup_common(self):
        portfolio = pd.DataFrame([], columns=['Ticker','Quantity','AvgCost'])
        snapshot = pd.DataFrame([
            {'Ticker': 'AAA', 'Price': 20.0},
            {'Ticker': 'BBB', 'Price': 20.0},
            {'Ticker': 'VNINDEX', 'Price': 1100.0},
        ])
        metrics = pd.DataFrame([
            {'Ticker': 'AAA', 'RSI14': 65.0, 'Sector': 'X', 'MomRetNorm': 0.9, 'LiqNorm': 0.8, 'ATR14_Pct': 10.0},
            {'Ticker': 'BBB', 'RSI14': 63.0, 'Sector': 'Y', 'MomRetNorm': 0.85, 'LiqNorm': 0.7, 'ATR14_Pct': 4.0},
            {'Ticker': 'VNINDEX', 'RSI14': 55.0},
        ])
        industry = pd.DataFrame({'Ticker': ['AAA','BBB'], 'Sector': ['X','Y']})
        presets = pd.DataFrame([
            {'Ticker': 'AAA', 'MA20': 19.0, 'MA50': 18.0, 'BandFloor_Tick': 18.0, 'BandCeiling_Tick': 22.0},
            {'Ticker': 'BBB', 'MA20': 19.0, 'MA50': 18.0, 'BandFloor_Tick': 18.0, 'BandCeiling_Tick': 22.0},
        ])
        sector_strength = pd.DataFrame([
            {'sector': 'Tất cả', 'breadth_above_ma50_pct': 60.0, 'avg_rsi14': 55.0}
        ])
        session_summary = pd.DataFrame([
            {'SessionPhase': 'morning', 'InVNSession': 1, 'IndexChangePct': -0.4}
        ])
        pnl_summary = pd.DataFrame([[50000.0, 50000.0, 0.0, 0.0]], columns=['TotalCost','TotalMarket','TotalPnL','ReturnPct'])
        prices_history = pd.DataFrame({'Date': ['2024-01-02','2024-01-03'], 'Ticker': ['AAA','BBB'], 'Close': [20.0, 20.5]})

        return portfolio, snapshot, metrics, industry, presets, sector_strength, session_summary, pnl_summary, prices_history

    def _make_regime(self, tuning, **overrides):
        regime = MarketRegime(
            phase='morning',
            in_session=True,
            index_change_pct=overrides.get('index_change_pct', 0.0),
            breadth_hint=overrides.get('breadth_hint', 0.6),
            risk_on=overrides.get('risk_on', True),
            buy_budget_frac=tuning['buy_budget_frac'],
            top_sectors=[],
            add_max=tuning['add_max'],
            new_max=tuning['new_max'],
            weights=tuning['weights'],
            thresholds=tuning['thresholds'],
            sector_bias=tuning['sector_bias'],
            ticker_bias=tuning['ticker_bias'],
            pricing=tuning['pricing'],
            sizing=tuning['sizing'],
            execution=tuning.get('execution', {}),
        )
        regime.market_filter = tuning['market_filter']
        regime.market_score = overrides.get('market_score', 0.8)
        regime.trend_strength = overrides.get('trend_strength', 0.1)
        regime.index_atr_percentile = overrides.get('index_atr_percentile', 0.2)
        return regime

    def test_override_static_blend_behavior(self):
        portfolio, snapshot, metrics, industry, presets, sector_strength, session_summary, pnl_summary, prices_history = self._setup_common()
        tune_static = base_tuning()
        regime_static = self._make_regime(tune_static, risk_on=True, index_change_pct=0.5, breadth_hint=0.7)
        actions = {'AAA': 'new', 'BBB': 'new'}
        scores = {'AAA': 0.9, 'BBB': 0.8}
        orders_static, _, _ = build_orders(actions, portfolio, snapshot, metrics, presets, pnl_summary, scores, regime_static, prices_history)
        qty_static = next(o.quantity for o in orders_static if o.ticker == 'AAA')

        tune_dyn = base_tuning()
        dyn_conf = tune_dyn['sizing']['dynamic_caps']
        dyn_conf.update({'enable': 1, 'pos_min': 0.05, 'pos_max': 0.12, 'sector_min': 0.10, 'sector_max': 0.25, 'blend': 1.0, 'override_static': 1})
        regime_dyn = self._make_regime(tune_dyn, risk_on=True, index_change_pct=-0.3, breadth_hint=0.3)
        orders_dyn, _, regime_dyn_out = build_orders(actions, portfolio, snapshot, metrics, presets, pnl_summary, scores, regime_dyn, prices_history)
        if not orders_dyn:
            self.assertLess(regime_dyn_out.buy_budget_frac_effective, regime_static.buy_budget_frac)
        else:
            order_dyn = next(o for o in orders_dyn if o.ticker == 'AAA')
            qty_dyn = order_dyn.quantity
            self.assertLess(qty_dyn, qty_static)
            nav = float(pnl_summary.loc[0, 'TotalMarket'])
            risk_score = 0.5 * ((regime_dyn.index_change_pct / 1.0 + 1.0) / 2.0) + 0.5 * regime_dyn.breadth_hint - 0.1
            risk_score = max(0.0, min(1.0, risk_score))
            dyn_pos = dyn_conf['pos_min'] + (dyn_conf['pos_max'] - dyn_conf['pos_min']) * risk_score
            expected_qty = int((dyn_pos * nav) / (order_dyn.limit_price * tune_dyn['sizing']['min_lot'])) * tune_dyn['sizing']['min_lot']
            self.assertEqual(qty_dyn, expected_qty)

    def test_risk_per_trade_caps_buy_qty(self):
        portfolio, snapshot, metrics, industry, presets, sector_strength, session_summary, pnl_summary, prices_history = self._setup_common()
        tune_base = base_tuning()
        tune_base['sizing']['leftover_redistribute'] = 0
        regime_base = self._make_regime(tune_base, risk_on=True, index_change_pct=0.4, breadth_hint=0.7)
        actions = {'AAA': 'new', 'BBB': 'new'}
        scores = {'AAA': 0.9, 'BBB': 0.8}
        orders0, _, _ = build_orders(actions, portfolio, snapshot, metrics, presets, pnl_summary, scores, regime_base, prices_history)
        base_qty = next(o.quantity for o in orders0 if o.ticker == 'AAA')

        tune = base_tuning()
        tune['sizing']['risk_per_trade_frac'] = 0.01
        tune['sizing']['default_stop_atr_mult'] = 2.0
        tune['sizing']['leftover_redistribute'] = 0
        regime = self._make_regime(tune, risk_on=True, index_change_pct=0.4, breadth_hint=0.7)
        orders, _, _ = build_orders(actions, portfolio, snapshot, metrics, presets, pnl_summary, scores, regime, prices_history)
        order = next(o for o in orders if o.ticker == 'AAA')
        qty = order.quantity
        nav = float(pnl_summary.loc[0, 'TotalMarket'])
        atr_pct = metrics.loc[metrics['Ticker'] == 'AAA', 'ATR14_Pct'].iloc[0]
        price_now = snapshot.loc[snapshot['Ticker'] == 'AAA', 'Price'].iloc[0]
        atr_k = (atr_pct / 100.0) * price_now
        stop_dist = tune['sizing']['default_stop_atr_mult'] * atr_k
        allowed = tune['sizing']['risk_per_trade_frac'] * nav
        lot = tune['sizing']['min_lot']
        risk_cap_qty = int(max(allowed / stop_dist / lot, 0)) * lot
        self.assertEqual(base_qty, 100)
        self.assertGreaterEqual(qty, 0)
        self.assertLessEqual(qty, max(risk_cap_qty, 0))
        self.assertEqual(qty % 100, 0)

    def test_min_ticket_k_leftover_behavior(self):
        portfolio, snapshot, metrics, industry, presets, sector_strength, session_summary, pnl_summary, prices_history = self._setup_common()
        presets.loc[presets['Ticker'] == 'BBB', 'Break_Buy1_Tick'] = 22.0
        presets.loc[presets['Ticker'] == 'BBB', 'Break_Buy2_Tick'] = 21.8
        tune_base = base_tuning()
        tune_base['market_filter']['risk_off_buy'] = ["Break"]

        actions = {'AAA': 'new', 'BBB': 'new'}
        scores = {'AAA': 0.9, 'BBB': 0.8}
        regime = self._make_regime(tune_base)
        orders_open, _, reg_state = build_orders(actions, portfolio, snapshot, metrics, presets, pnl_summary, scores, regime, prices_history)
        qty_loose = next(o.quantity for o in orders_open if o.ticker == 'AAA')
        filters = getattr(reg_state, 'debug_filters', {})
        self.assertIn('BBB', filters.get('limit_gt_market', []))

        tune_strict = base_tuning()
        tune_strict['market_filter']['risk_off_buy'] = ["Break"]
        tune_strict['sizing']['min_ticket_k'] = 6000.0
        regime_strict = self._make_regime(tune_strict)
        orders_strict, _, _ = build_orders(actions, portfolio, snapshot, metrics, presets, pnl_summary, scores, regime_strict, prices_history)
        qty_strict = next(o.quantity for o in orders_strict if o.ticker == 'AAA')
        self.assertLessEqual(qty_strict, qty_loose)
