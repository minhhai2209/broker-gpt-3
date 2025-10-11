import os
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from scripts.engine import config_io


class TestConfigIO(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        (base / "config").mkdir(parents=True, exist_ok=True)
        (base / "out" / "orders").mkdir(parents=True, exist_ok=True)

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

    def test_copy_from_config_when_available(self):
        src = config_io.OVERRIDE_SRC
        dest = self._dest()
        dest.unlink(missing_ok=True)
        sample = '{"buy_budget_frac": 0.10}'.strip()
        src.write_text(sample, encoding="utf-8")

        with StringIO() as buf, redirect_stdout(buf):
            path = config_io.ensure_policy_override_file()
            output = buf.getvalue()

        self.assertEqual(path, dest)
        self.assertTrue(dest.exists())
        self.assertEqual(dest.read_text(encoding="utf-8"), sample)
        self.assertIn("Copied policy overrides", output)

    # POLICY_FILE legacy env removed: explicit env override is no longer supported.
    # The engine always merges baseline + overlays from the repo config.
    def test_no_env_override_supported(self):
        dest = self._dest()
        dest.unlink(missing_ok=True)
        # Provide minimal baseline and an overlay; ensure result comes from config files, not env.
        base = config_io.BASE_DIR / 'config' / 'policy_default.json'
        base.write_text('{"buy_budget_frac": 0.10, "add_max":1, "new_max":1, "weights":{}, "thresholds": {"base_add":0.1, "base_new":0.1, "trim_th":0, "q_add":0.5, "q_new":0.5, "min_liq_norm":0, "near_ceiling_pct":0.98, "tp_pct":0, "sl_pct":1, "tp_trim_frac":0.3, "exit_on_ma_break":0, "cooldown_days":0}, "pricing": {"risk_on_buy":[], "risk_on_sell":[], "risk_off_buy":[], "risk_off_sell":[], "atr_fallback_buy_mult":0.25, "atr_fallback_sell_mult":0.25}, "sizing": {"add_share":0.5, "new_share":0.5, "cov_lookback_days":60, "cov_reg":0.0001, "risk_parity_floor":0.2, "market_index_symbol":"VNINDEX", "default_stop_atr_mult":2.0, "risk_per_trade_frac":0.0}, "market_filter": {"risk_off_trend_floor":0.0, "risk_off_breadth_floor":0.4, "breadth_relax_margin":0.0, "risk_off_drawdown_floor":0.2, "market_score_soft_floor":0.6, "market_score_hard_floor":0.4, "leader_min_rsi":55, "leader_min_mom_norm":0.6, "leader_require_ma20":1, "leader_require_ma50":1, "leader_max":2, "index_atr_soft_pct":0.8, "index_atr_hard_pct":0.95, "guard_new_scale_cap":0.4, "atr_soft_scale_cap":0.5, "severe_drop_mult":1.5, "idx_chg_smoothed_hard_drop":0.5, "trend_norm_hard_floor":-0.25, "vol_ann_hard_ceiling":0.6}, "regime_model": {"intercept":0, "threshold":0.5, "components":{}}, "sector_bias":{}, "ticker_bias":{}}', encoding='utf-8')
        (config_io.BASE_DIR / 'config' / 'policy_ai_overrides.json').write_text('{"buy_budget_frac": 0.12}', encoding='utf-8')
        with patch.dict(os.environ, {"POLICY_FILE": str(config_io.BASE_DIR / 'custom_policy.json')}, clear=False):
            path = config_io.ensure_policy_override_file()
        self.assertEqual(path, dest)
        self.assertIn('"buy_budget_frac": 0.12', dest.read_text(encoding='utf-8'))

    def test_existing_runtime_used_when_config_missing(self):
        dest = self._dest()
        dest.write_text('{"buy_budget_frac": 0.09}', encoding="utf-8")
        src = config_io.OVERRIDE_SRC
        src.unlink(missing_ok=True)

        with patch.dict(os.environ, {}, clear=True):
            path = config_io.ensure_policy_override_file()

        self.assertEqual(path, dest)
        self.assertEqual(dest.read_text(encoding="utf-8"), '{"buy_budget_frac": 0.09}')

    def test_missing_sources_raise(self):
        dest = self._dest()
        dest.unlink(missing_ok=True)
        src = config_io.OVERRIDE_SRC
        src.unlink(missing_ok=True)

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit) as exc:
                config_io.ensure_policy_override_file()

        self.assertIn("Missing policy overrides", str(exc.exception))

    def test_suggest_tuning_requires_runtime_copy(self):
        dest = self._dest()
        dest.unlink(missing_ok=True)

        with self.assertRaises(SystemExit) as exc:
            config_io.suggest_tuning(pd.DataFrame(), pd.DataFrame())

        self.assertIn("Missing out/orders/policy_overrides.json", str(exc.exception))
