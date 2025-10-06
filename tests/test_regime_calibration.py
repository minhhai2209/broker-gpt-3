import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from scripts.tuning.calibrators import calibrate_regime as cal_reg
from scripts.orders.order_engine import get_market_regime


class TestRegimeCalibration(unittest.TestCase):
    def test_updates_intercept_threshold(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            out_dir = base / 'out'
            orders_dir = out_dir / 'orders'
            config_dir = base / 'config'
            out_dir.mkdir(parents=True, exist_ok=True)
            orders_dir.mkdir(parents=True, exist_ok=True)
            config_dir.mkdir(parents=True, exist_ok=True)

            prices = self._prices_history()
            prices.to_csv(out_dir / 'prices_history.csv', index=False)

            policy = {
                'regime_model': {
                    'intercept': 0.0,
                    'threshold': 0.5,
                    'components': {
                        'trend': {'mean': 0.5, 'std': 0.2, 'weight': 1.0},
                        'index_return': {'mean': 0.0, 'std': 0.6, 'weight': 0.8},
                        'volatility': {'mean': 0.5, 'std': 0.2, 'weight': 0.6},
                        'drawdown': {'mean': 0.5, 'std': 0.2, 'weight': 0.5},
                    }
                }
            }
            policy_path = orders_dir / 'policy_overrides.json'
            policy_path.write_text(json.dumps(policy), encoding='utf-8')

            orig_out_dir, orig_orders_dir, orig_config_path = cal_reg.OUT_DIR, cal_reg.ORDERS_DIR, cal_reg.CONFIG_PATH
            cal_reg.OUT_DIR = out_dir
            cal_reg.ORDERS_DIR = orders_dir
            cal_reg.CONFIG_PATH = config_dir / 'policy_overrides.json'
            try:
                b, thr, n, pos_rate, names = cal_reg.calibrate(horizon=5, write=True)
            finally:
                cal_reg.OUT_DIR = orig_out_dir
                cal_reg.ORDERS_DIR = orig_orders_dir
                cal_reg.CONFIG_PATH = orig_config_path

            self.assertGreater(n, 0)
            self.assertGreater(pos_rate, 0.0)
            self.assertTrue(names)
            self.assertTrue(math.isfinite(b))
            self.assertTrue(math.isfinite(thr))

            written = json.loads(policy_path.read_text(encoding='utf-8'))
            rm = written['regime_model']
            self.assertEqual(rm['intercept'], b)
            self.assertEqual(rm['threshold'], thr)

    def test_unknown_component_failfast(self):
        session_summary = pd.DataFrame([
            {'SessionPhase': 'morning', 'InVNSession': 1, 'IndexChangePct': 0.2}
        ])
        sector_strength = pd.DataFrame([
            {'sector': 'Tất cả', 'breadth_above_ma50_pct': 55.0, 'avg_rsi14': 55.0}
        ])
        tuning = {
            'buy_budget_frac': 0.20,
            'add_max': 5,
            'new_max': 5,
            'weights': {
                'w_trend': 0.2,
                'w_momo': 0.2,
                'w_liq': 0.2,
                'w_vol_guard': 0.2,
                'w_beta': 0.2,
                'w_sector': 0.0,
                'w_sector_sent': 0.0,
                'w_ticker_sent': 0.0,
                'w_roe': 0.0,
                'w_earnings_yield': 0.0,
                'w_rs': 0.0,
            },
            'regime_model': {
                'intercept': 0.1,
                'threshold': 0.5,
                'components': {
                    'trend': {'mean': 0.5, 'std': 0.2, 'weight': 1.0},
                    'unknown_factor': {'mean': 0.0, 'std': 0.3, 'weight': 0.5},
                }
            },
            'thresholds': {
                'base_add': 0.10,
                'base_new': 0.10,
                'trim_th': -0.10,
                'q_add': 0.5,
                'q_new': 0.5,
                'min_liq_norm': 0.0,
                'near_ceiling_pct': 0.98,
                'tp_pct': 0.0,
                'sl_pct': 1.0,
                'tp_trim_frac': 0.30,
                'exit_on_ma_break': 0,
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
                'risk_on_buy': ["Aggr","Bal"],
                'risk_on_sell': ["Cons","Bal"],
                'risk_off_buy': ["Cons","MR"],
                'risk_off_sell': ["MR","Cons"],
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
        }
        with self.assertRaises(SystemExit) as ctx:
            get_market_regime(session_summary, sector_strength, tuning)
        self.assertIn('unknown component', str(ctx.exception))

    def _prices_history(self):
        dates = pd.date_range('2022-01-03', periods=280, freq='B')
        steps = pd.Series(range(len(dates)))
        close = 1000.0 + 3.0 * steps + 20.0 * pd.Series(np.sin(steps / 20.0))
        high = close + 3.0
        low = close - 3.0
        df = pd.DataFrame({
            'Date': dates.strftime('%Y-%m-%d'),
            'Ticker': ['VNINDEX'] * len(dates),
            'Close': close,
            'High': high,
            'Low': low,
        })
        return df
