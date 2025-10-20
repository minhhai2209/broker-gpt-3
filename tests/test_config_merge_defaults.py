import os
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.engine import config_io


class TestConfigMergeDefaults(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        (base / "config").mkdir(parents=True, exist_ok=True)
        (base / "out" / "orders").mkdir(parents=True, exist_ok=True)
        # Patch paths in module
        self.patchers = [
            patch.object(config_io, "BASE_DIR", base),
            patch.object(config_io, "OUT_DIR", base / "out"),
            patch.object(config_io, "OUT_ORDERS_DIR", base / "out" / "orders"),
            patch.object(config_io, "OVERRIDE_SRC", base / "config" / "policy_overrides.json"),
        ]
        for p in self.patchers:
            p.start()
            self.addCleanup(p.stop)

    def _dest(self) -> Path:
        return config_io.OUT_ORDERS_DIR / "policy_overrides.json"

    def test_merge_overlays_in_order(self):
        base = config_io.BASE_DIR / 'config'
        base.mkdir(parents=True, exist_ok=True)
        (base / 'policy_default.json').write_text(json.dumps({
            "buy_budget_frac": 0.10,
            "add_max": 1,
            "new_max": 1,
            "weights": {},
            "thresholds": {"base_add": 0.1, "base_new": 0.1, "trim_th": 0, "q_add": 0.5, "q_new": 0.5, "min_liq_norm": 0.0, "near_ceiling_pct": 0.98, "tp_pct": 0.0, "sl_pct": 1.0, "tp_trim_frac": 0.3, "exit_on_ma_break": 0, "cooldown_days": 0},
            "pricing": {"risk_on_buy": [], "risk_on_sell": [], "risk_off_buy": [], "risk_off_sell": [], "atr_fallback_buy_mult": 0.25, "atr_fallback_sell_mult": 0.25},
            "sizing": {"add_share": 0.5, "new_share": 0.5, "cov_lookback_days": 60, "cov_reg": 0.0005, "risk_parity_floor": 0.2, "market_index_symbol": "VNINDEX", "default_stop_atr_mult": 2.0, "risk_per_trade_frac": 0.0},
            "market_filter": {
                "risk_off_trend_floor": -0.015,
                "risk_off_breadth_floor": 0.4,
                "breadth_relax_margin": 0.0,
                "risk_off_drawdown_floor": 0.2,
                "market_score_soft_floor": 0.55,
                "market_score_hard_floor": 0.35,
                "leader_min_rsi": 55.0,
                "leader_min_mom_norm": 0.6,
                "leader_require_ma20": 1,
                "leader_require_ma50": 1,
                "leader_max": 2,
                "guard_new_scale_cap": 0.4,
                "atr_soft_scale_cap": 0.5,
                "severe_drop_mult": 1.5,
                "idx_chg_smoothed_hard_drop": 0.5,
                "trend_norm_hard_floor": -0.25,
                "vol_ann_hard_ceiling": 0.6
            },
            "regime_model": {"intercept": 0.0, "threshold": 0.5, "components": {}},
            "sector_bias": {},
            "ticker_bias": {}
        }), encoding='utf-8')
        (base / 'policy_nightly_overrides.json').write_text('{"buy_budget_frac": 0.11}', encoding='utf-8')
        (base / 'policy_ai_overrides.json').write_text('{"buy_budget_frac": 0.12}', encoding='utf-8')

        path = config_io.ensure_policy_override_file()
        self.assertEqual(path, self._dest())
        obj = json.loads(self._dest().read_text(encoding='utf-8'))
        self.assertEqual(obj["buy_budget_frac"], 0.12)

    def test_merge_defaults_from_config_dir(self):
        # Write config base and defaults
        base = config_io.OVERRIDE_SRC
        defaults = config_io.BASE_DIR / "config" / "policy_for_calibration.json"
        base.write_text(json.dumps({
            "buy_budget_frac": 0.1,
            "add_max": 1,
            "new_max": 1,
            "weights": {"w_trend": 0, "w_momo": 0, "w_mom_ret": 0, "w_liq": 0, "w_vol_guard": 0, "w_beta": 0, "w_sector": 0, "w_sector_sent": 0, "w_ticker_sent": 0, "w_roe": 0, "w_earnings_yield": 0, "w_rs": 0},
            "thresholds": {"base_add": 0.1, "base_new": 0.1, "trim_th": -0.1, "q_add": 0.5, "q_new": 0.5, "min_liq_norm": 0.0, "near_ceiling_pct": 0.98, "tp_pct": 0.0, "sl_pct": 1.0, "tp_trim_frac": 0.3, "exit_on_ma_break": 0, "cooldown_days": 0},
            "pricing": {"risk_on_buy": ["Aggr"], "risk_on_sell": ["Cons"], "risk_off_buy": ["Cons"], "risk_off_sell": ["MR"], "atr_fallback_buy_mult": 0.25, "atr_fallback_sell_mult": 0.25},
            "sizing": {
                "softmax_tau": 0.6,
                "add_share": 0.5,
                "new_share": 0.5,
                "min_lot": 100,
                "risk_weighting": "score_softmax",
                "risk_alpha": 1.0,
                "max_pos_frac": 0.5,
                "max_sector_frac": 0.8,
                "reuse_sell_proceeds_frac": 0.0,
                "risk_blend": 1.0,
                "min_ticket_k": 0.0,
                "cov_lookback_days": 60,
                "cov_reg": 0.0005,
                "risk_parity_floor": 0.2,
                "leftover_redistribute": 1,
                "dynamic_caps": {"enable": 0, "pos_min": 0.1, "pos_max": 0.2, "sector_min": 0.3, "sector_max": 0.4, "blend": 1.0, "override_static": 0},
                "allocation_model": "softmax",
                "bl_rf_annual": 0.03,
                "bl_mkt_prem_annual": 0.06,
                "bl_alpha_scale": 0.02,
                "risk_blend_eta": 0.0,
                "min_names_target": 0,
                "market_index_symbol": "VNINDEX",
                "risk_per_trade_frac": 0.0,
                "default_stop_atr_mult": 2.0,
                "mean_variance_calibration": {
                    "enable": 0,
                    "risk_alpha": [1.0],
                    "cov_reg": [0.0005],
                    "bl_alpha_scale": [0.02],
                    "lookback_days": 120,
                    "test_horizon_days": 30,
                    "min_history_days": 180
                }
            },
            "market_filter": {
                "risk_off_trend_floor": -0.015,
                "risk_off_breadth_floor": 0.4,
                "breadth_relax_margin": 0.0,
                "risk_off_drawdown_floor": 0.2,
                "market_score_soft_floor": 0.55,
                "market_score_hard_floor": 0.35,
                "leader_min_rsi": 55.0,
                "leader_min_mom_norm": 0.6,
                "leader_require_ma20": 1,
                "leader_require_ma50": 1,
                "leader_max": 2,
                "guard_new_scale_cap": 0.4,
                "atr_soft_scale_cap": 0.5,
                "severe_drop_mult": 1.5,
                "idx_chg_smoothed_hard_drop": 0.5,
                "trend_norm_hard_floor": -0.25,
                "vol_ann_hard_ceiling": 0.6
            },
            "regime_model": {"intercept": 0.0, "threshold": 0.5, "components": {"trend": {"mean": 0.5, "std": 0.2, "weight": 1.0}}},
            "sector_bias": {},
            "ticker_bias": {}
        }), encoding="utf-8")
        defaults.write_text('{"market_filter": {"index_atr_soft_pct": 0.9, "index_atr_hard_pct": 0.99}}', encoding="utf-8")

        path = config_io.ensure_policy_override_file()
        self.assertEqual(path, self._dest())
        obj = json.loads(self._dest().read_text(encoding="utf-8"))
        self.assertEqual(obj["market_filter"].get("index_atr_soft_pct"), 0.9)
        self.assertEqual(obj["market_filter"].get("index_atr_hard_pct"), 0.99)

    # No fallback to sample in runtime; defaults.json absence simply results in copying base only.
    def test_no_defaults_json_copies_base_only(self):
        base = config_io.OVERRIDE_SRC
        base.write_text('{"buy_budget_frac": 0.1, "add_max": 1, "new_max": 1, "weights": {"w_trend": 0, "w_momo": 0, "w_mom_ret": 0, "w_liq": 0, "w_vol_guard": 0, "w_beta": 0, "w_sector": 0, "w_sector_sent": 0, "w_ticker_sent": 0, "w_roe": 0, "w_earnings_yield": 0, "w_rs": 0}, "thresholds": {"base_add": 0.1, "base_new": 0.1, "trim_th": -0.1, "near_ceiling_pct": 0.98, "tp_trim_frac": 0.3, "exit_on_ma_break": 0, "cooldown_days": 0, "exit_ma_break_rsi": 45, "trim_rsi_below_ma20": 45, "trim_rsi_macdh_neg": 40}, "market_filter": {"risk_off_trend_floor": -0.015, "risk_off_breadth_floor": 0.4, "breadth_relax_margin": 0.0, "risk_off_drawdown_floor": 0.2, "market_score_soft_floor": 0.55, "market_score_hard_floor": 0.35, "leader_min_rsi": 55.0, "leader_min_mom_norm": 0.6, "leader_require_ma20": 1, "leader_require_ma50": 1, "leader_max": 2, "guard_new_scale_cap": 0.4, "atr_soft_scale_cap": 0.5, "severe_drop_mult": 1.5, "idx_chg_smoothed_hard_drop": 0.5, "trend_norm_hard_floor": -0.25, "vol_ann_hard_ceiling": 0.6}, "regime_model": {"components": {"trend": {"weight": 1.0}}}}', encoding="utf-8")
        path = config_io.ensure_policy_override_file()
        self.assertEqual(path, self._dest())
        obj = json.loads(self._dest().read_text(encoding='utf-8'))
        self.assertNotIn('q_add', obj.get('thresholds', {}))


if __name__ == '__main__':
    unittest.main()
