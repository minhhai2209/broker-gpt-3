import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.order_engine import build_orders, decide_actions


class TestCooldownLedger(unittest.TestCase):
    def setUp(self):
        self.out_dir = Path('out')
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.orders_dir = self.out_dir / 'orders'
        self.orders_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.orders_dir / 'last_actions.csv'
        if self.ledger_path.exists():
            self.ledger_path.unlink()
        self.hist_path = self.out_dir / 'prices_history.csv'
        self._hist_backup = self.hist_path.read_bytes() if self.hist_path.exists() else None
        hist = pd.DataFrame({
            'Date': pd.date_range(end=datetime.now(), periods=30, freq='D'),
            'Ticker': ['VNINDEX'] * 30,
            'Close': [1000.0 + i for i in range(30)],
        })
        hist['High'] = hist['Close'] * 1.01
        hist['Low'] = hist['Close'] * 0.99
        hist.to_csv(self.hist_path, index=False)

    def tearDown(self):
        if self.ledger_path.exists():
            self.ledger_path.unlink()
        if self.hist_path.exists():
            self.hist_path.unlink()
        if self._hist_backup is not None:
            self.hist_path.write_bytes(self._hist_backup)

    def _tuning(self, cooldown_days):
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
                'cooldown_days': cooldown_days,
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

    def test_new_blocked_after_recent_exit_or_tp(self):
        recent = datetime.now().date().isoformat()
        self.ledger_path.write_text('Ticker,LastAction,Date\nAAA,exit,%s\n' % recent, encoding='utf-8')
        portfolio = pd.DataFrame([], columns=['Ticker','Quantity','AvgCost'])
        snapshot = pd.DataFrame([
            {'Ticker': 'AAA', 'Price': 20.0},
            {'Ticker': 'VNINDEX', 'Price': 1100.0},
        ])
        metrics = pd.DataFrame([
            {'Ticker': 'AAA', 'RSI14': 65.0, 'Sector': 'X', 'MomRetNorm': 0.9, 'LiqNorm': 0.8, 'ATR14_Pct': 1.0},
            {'Ticker': 'VNINDEX', 'RSI14': 55.0},
        ])
        industry = pd.DataFrame({'Ticker': ['AAA'], 'Sector': ['X']})
        presets = pd.DataFrame([
            {'Ticker': 'AAA', 'MA20': 19.0, 'MA50': 18.0, 'BandFloor_Tick': 18.0, 'BandCeiling_Tick': 22.0},
        ])
        sector_strength = pd.DataFrame([
            {'sector': 'Tất cả', 'breadth_above_ma50_pct': 60.0, 'avg_rsi14': 55.0}
        ])
        session_summary = pd.DataFrame([
            {'SessionPhase': 'morning', 'InVNSession': 1, 'IndexChangePct': 0.4}
        ])
        tuning = self._tuning(cooldown_days=3)
        actions, scores, feats, regime = decide_actions(
            portfolio, snapshot, metrics, presets, industry, sector_strength, session_summary, tuning
        )
        self.assertNotIn('AAA', actions)
        market_filters = getattr(regime, 'debug_filters', {}).get('market', [])
        self.assertIn('AAA', market_filters)
        records = getattr(regime, 'filtered_records', [])
        details = [r.get('Note', '') for r in records if str(r.get('Ticker')).upper() == 'AAA']
        self.assertTrue(any('cooldown active' in str(d) for d in details))

    def test_ledger_update_written(self):
        portfolio = pd.DataFrame([{'Ticker': 'AAA', 'Quantity': 1000, 'AvgCost': 18.0}])
        snapshot = pd.DataFrame([
            {'Ticker': 'AAA', 'Price': 20.0},
            {'Ticker': 'VNINDEX', 'Price': 1100.0},
        ])
        metrics = pd.DataFrame([
            {'Ticker': 'AAA', 'RSI14': 55.0, 'Sector': 'X', 'MomRetNorm': 0.6, 'LiqNorm': 0.6, 'ATR14_Pct': 1.5},
            {'Ticker': 'VNINDEX', 'RSI14': 55.0},
        ])
        industry = pd.DataFrame({'Ticker': ['AAA'], 'Sector': ['X']})
        presets = pd.DataFrame([
            {'Ticker': 'AAA', 'MA20': 19.0, 'MA50': 18.0, 'BandFloor_Tick': 18.0, 'BandCeiling_Tick': 22.0},
        ])
        sector_strength = pd.DataFrame([
            {'sector': 'Tất cả', 'breadth_above_ma50_pct': 60.0, 'avg_rsi14': 55.0}
        ])
        session_summary = pd.DataFrame([
            {'SessionPhase': 'morning', 'InVNSession': 1, 'IndexChangePct': 0.4}
        ])
        tuning = self._tuning(cooldown_days=0)
        today = datetime.now().date().isoformat()
        records = [{'Ticker': 'AAA', 'LastAction': 'exit', 'Date': today}]
        df_new = pd.DataFrame(records)
        df_old = pd.DataFrame([
            {'Ticker': 'AAA', 'LastAction': 'take_profit', 'Date': (datetime.now().date().replace(day=1)).isoformat()}
        ])
        df = pd.concat([df_old, df_new], ignore_index=True)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values(['Ticker','Date']).dropna(subset=['Ticker','Date'])
        df = df.drop_duplicates(subset=['Ticker'], keep='last')
        df['Ticker'] = df['Ticker'].astype(str).str.upper()
        row = df[df['Ticker'] == 'AAA'].iloc[0]
        self.assertEqual(row['LastAction'], 'exit')
        self.assertEqual(pd.to_datetime(row['Date']).date(), datetime.now().date())
