import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.orders.order_engine import decide_actions, build_orders


class TestNeutralAdaptiveEntry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        out_dir = Path('out')
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / 'prices_history.csv'
        dates = pd.date_range('2022-01-03', periods=260, freq='B')
        df = pd.DataFrame({
            'Date': dates.strftime('%Y-%m-%d'),
            'Ticker': ['VNINDEX'] * len(dates),
            'Close': 1000.0 + 0.5 * pd.RangeIndex(len(dates)),
        })
        df.to_csv(path, index=False)

    def make_tuning(self, *, neutral_enable: bool = True, q_mode: str = 'subset') -> dict:
        tuning = {
            'buy_budget_frac': 0.1,
            'add_max': 3,
            'new_max': 3,
            'weights': {
                'w_trend': 0.0,
                'w_momo': 0.0,
                'w_mom_ret': 0.0,
                'w_liq': 0.0,
                'w_vol_guard': 0.0,
                'w_beta': 0.0,
                'w_sector': 0.0,
                'w_sector_sent': 0.0,
                'w_ticker_sent': 1.0,
                'w_roe': 0.0,
                'w_earnings_yield': 0.0,
                'w_rs': 0.0,
                'fund_scale': 1.0,
            },
            'regime_model': {
                'intercept': 0.0,
                'threshold': 0.5,
                'components': {
                    'trend': {'mean': 0.5, 'std': 0.2, 'weight': 0.0},
                    'breadth': {'mean': 0.5, 'std': 0.2, 'weight': 0.0},
                    'index_return': {'mean': 0.0, 'std': 0.5, 'weight': 0.0},
                },
            },
            'thresholds': {
                'base_add': 0.35,
                'base_new': 0.40,
                'trim_th': -0.05,
                'q_add': 0.7,
                'q_new': 0.8,
                'min_liq_norm': 0.2,
                'near_ceiling_pct': 0.98,
                'tp_pct': 0.1,
                'sl_pct': 0.07,
                'tp_atr_mult': None,
                'sl_atr_mult': None,
                'tp_floor_pct': 0.05,
                'sl_floor_pct': 0.04,
                'tp_trim_frac': 0.4,
                'exit_on_ma_break': 0,
                'cooldown_days': 0,
                'exit_ma_break_rsi': 45.0,
                'trim_rsi_below_ma20': 45.0,
                'trim_rsi_macdh_neg': 40.0,
                'exit_ma_break_score_gate': 0.0,
                'tilt_exit_downgrade_min': 0.05,
                'exit_downgrade_min_r': 0.0,
                'tc_gate_scale': 0.0,
                'exit_ma_break_min_phase': 'morning',
                'quantile_pool': q_mode,
            },
            'neutral_adaptive': {
                'neutral_enable': 1 if neutral_enable else 0,
                'neutral_risk_on_prob_low': 0.35,
                'neutral_risk_on_prob_high': 0.65,
                'neutral_index_atr_soft_cap': None,
                'neutral_breadth_band': 0.05,
                'neutral_breadth_center': 0.5,
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
                'risk_off_index_drop_pct': 0.6,
                'risk_off_trend_floor': -1.0,
                'risk_off_breadth_floor': 0.0,
                'market_score_soft_floor': 0.2,
                'market_score_hard_floor': 0.05,
                'leader_min_rsi': 0.0,
                'leader_min_mom_norm': 0.0,
                'leader_require_ma20': 0,
                'leader_require_ma50': 0,
                'leader_max': 2,
                'risk_off_drawdown_floor': 1.0,
                'index_atr_soft_pct': 0.9,
                'index_atr_hard_pct': 0.95,
                'guard_new_scale_cap': 0.4,
                'atr_soft_scale_cap': 0.5,
                'severe_drop_mult': 1.5,
                'idx_chg_smoothed_hard_drop': 1.0,
                'trend_norm_hard_floor': -1.0,
                'vol_ann_hard_ceiling': 10.0,
            },
            'pricing': {
                'risk_on_buy': ["Aggr", "Bal", "Cons"],
                'risk_on_sell': ["Cons", "Bal", "Aggr"],
                'risk_off_buy': ["Cons", "MR"],
                'risk_off_sell': ["MR", "Cons"],
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
                'add_share': 0.5,
                'new_share': 0.5,
                'min_lot': 100,
                'risk_weighting': 'score_softmax',
                'risk_alpha': 1.0,
                'max_pos_frac': 1.0,
                'max_sector_frac': 1.0,
                'reuse_sell_proceeds_frac': 0.0,
                'leftover_redistribute': 1,
                'min_ticket_k': 0.0,
                'risk_blend': 0.0,
                'cov_lookback_days': 90,
                'cov_reg': 0.0005,
                'risk_parity_floor': 0.0,
                'dynamic_caps': {
                    'enable': 0,
                    'pos_min': 0.1,
                    'pos_max': 0.2,
                    'sector_min': 0.1,
                    'sector_max': 0.3,
                    'blend': 1.0,
                    'override_static': 0,
                },
            },
        }
        return tuning

    def _make_frames(self, tickers: list[str], prices: list[float], ticker_bias: dict[str, float], holdings: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        price_map = {t: p for t, p in zip(tickers, prices)}
        portfolio_rows = [{'Ticker': t, 'Quantity': qty, 'AvgCost': price_map.get(t, 1.0)} for t, qty in holdings.items()]
        if portfolio_rows:
            portfolio = pd.DataFrame(portfolio_rows)
        else:
            portfolio = pd.DataFrame(columns=['Ticker', 'Quantity', 'AvgCost'])
        snapshot = pd.DataFrame({'Ticker': tickers, 'Price': prices})
        metrics = pd.DataFrame({
            'Ticker': tickers,
            'RSI14': [55.0] * len(tickers),
            'LiqNorm': [0.8] * len(tickers),
            'MomRetNorm': [0.5] * len(tickers),
            'ATR14_Pct': [2.0] * len(tickers),
            'Beta60D': [1.0] * len(tickers),
            'Fund_ROE': [0.1] * len(tickers),
            'Fund_EarningsYield': [0.1] * len(tickers),
            'RS_Trend50': [0.5] * len(tickers),
            'Sector': ['Tech'] * len(tickers),
        })
        presets = pd.DataFrame([])
        industry = pd.DataFrame({'Ticker': tickers, 'Sector': ['Tech'] * len(tickers)})
        sector_strength = pd.DataFrame([])
        return portfolio, snapshot, metrics, presets, industry, sector_strength

    def _session_summary(self) -> pd.DataFrame:
        return pd.DataFrame([
            {'SessionPhase': 'morning', 'InVNSession': 1, 'IndexChangePct': 0.0, 'BreadthPct': 50.0}
        ])

    def test_neutral_detection_scales_thresholds(self):
        tuning = self.make_tuning()
        tuning['ticker_bias'] = {'AAA': 0.9, 'BBB': 0.5}
        portfolio, snapshot, metrics, presets, industry, sector_strength = self._make_frames(
            ['AAA', 'BBB'], [10.0, 10.0], tuning['ticker_bias'], {'AAA': 100}
        )
        actions, scores, feats, regime = decide_actions(
            portfolio,
            snapshot,
            metrics,
            presets,
            industry,
            sector_strength,
            self._session_summary(),
            tuning,
        )
        self.assertTrue(regime.is_neutral)
        expected_base_new = 0.40 * 0.90
        self.assertAlmostEqual(regime.thresholds['base_new'], expected_base_new, places=6)
        self.assertAlmostEqual(regime.thresholds['base_add'], 0.35 * 0.90, places=6)

    def test_neutral_disabled_when_guard_hits(self):
        tuning = self.make_tuning()
        tuning['ticker_bias'] = {'AAA': 0.9}
        session_summary = pd.DataFrame([
            {'SessionPhase': 'morning', 'InVNSession': 1, 'IndexChangePct': -1.0}
        ])
        portfolio, snapshot, metrics, presets, industry, sector_strength = self._make_frames(
            ['AAA'], [10.0], tuning['ticker_bias'], {'AAA': 100}
        )
        actions, scores, feats, regime = decide_actions(
            portfolio,
            snapshot,
            metrics,
            presets,
            industry,
            sector_strength,
            session_summary,
            tuning,
        )
        self.assertFalse(regime.is_neutral)
        self.assertEqual(regime.neutral_thresholds, {})
        self.assertAlmostEqual(regime.thresholds['base_new'], 0.40, places=6)

    def test_partial_entry_created_for_near_threshold(self):
        tuning = self.make_tuning()
        tuning['ticker_bias'] = {'AAA': 0.9, 'BBB': 0.33}
        portfolio, snapshot, metrics, presets, industry, sector_strength = self._make_frames(
            ['AAA', 'BBB', 'CCC'], [10.0, 10.0, 10.0], tuning['ticker_bias'], {'AAA': 100}
        )
        actions, scores, feats, regime = decide_actions(
            portfolio,
            snapshot,
            metrics,
            presets,
            industry,
            sector_strength,
            self._session_summary(),
            tuning,
        )
        self.assertEqual(actions.get('BBB'), 'new_partial')
        self.assertIn('BBB', regime.neutral_partial_tickers)

    def test_quantile_override_restores_candidate(self):
        tuning = self.make_tuning(q_mode='full')
        tuning['ticker_bias'] = {'AAA': 0.95, 'BBB': 0.45}
        portfolio, snapshot, metrics, presets, industry, sector_strength = self._make_frames(
            ['AAA', 'BBB'], [10.0, 10.0], tuning['ticker_bias'], {'AAA': 100}
        )
        actions, scores, feats, regime = decide_actions(
            portfolio,
            snapshot,
            metrics,
            presets,
            industry,
            sector_strength,
            self._session_summary(),
            tuning,
        )
        self.assertEqual(actions.get('BBB'), 'new')
        self.assertIn('BBB', regime.neutral_override_tickers)

    def test_add_cap_limits_to_one(self):
        tuning = self.make_tuning()
        tuning['ticker_bias'] = {'AAA': 0.9, 'BBB': 0.8, 'CCC': 0.7}
        portfolio, snapshot, metrics, presets, industry, sector_strength = self._make_frames(
            ['AAA', 'BBB', 'CCC'], [10.0, 10.0, 10.0], tuning['ticker_bias'], {'AAA': 100, 'BBB': 100}
        )
        actions, scores, feats, regime = decide_actions(
            portfolio,
            snapshot,
            metrics,
            presets,
            industry,
            sector_strength,
            self._session_summary(),
            tuning,
        )
        adds = [t for t, a in actions.items() if a == 'add']
        self.assertEqual(len(adds), 1)
        self.assertEqual(adds[0], regime.neutral_accum_tickers[0])

    def test_partial_sizing_and_notes(self):
        tuning = self.make_tuning()
        tuning['ticker_bias'] = {'BBB': 0.33}
        portfolio, snapshot, metrics, presets, industry, sector_strength = self._make_frames(
            ['BBB'], [1.0], tuning['ticker_bias'], {}
        )
        actions, scores, feats, regime = decide_actions(
            portfolio,
            snapshot,
            metrics,
            presets,
            industry,
            sector_strength,
            self._session_summary(),
            tuning,
        )
        pnl_summary = pd.DataFrame([{'TotalMarket': 10000.0}])
        orders, notes, _ = build_orders(
            actions,
            portfolio,
            snapshot,
            metrics,
            presets,
            pnl_summary,
            scores,
            regime,
            prices_history=pd.DataFrame(),
        )
        self.assertEqual(len(orders), 1)
        order = orders[0]
        self.assertEqual(order.ticker, 'BBB')
        self.assertEqual(order.side, 'BUY')
        # Base budget: 0.1 * 10000 = 1000 -> NEW probes default to 100 shares (lot size)
        self.assertEqual(order.quantity, 100)
        self.assertIn('PARTIAL_ENTRY', order.note)
        self.assertIn('NEUTRAL_ADAPT', order.note)


if __name__ == '__main__':
    unittest.main()
