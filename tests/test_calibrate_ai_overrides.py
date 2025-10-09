import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from scripts.tuning.calibrators import calibrate_ai_overrides


class TestCalibrateAIOverrides(TestCase):
    def test_normalise_overrides_clamps_values(self):
        raw = {
            "rationale": " Test ",
            "buy_budget_frac": 0.5,
            "add_max": 50,
            "new_max": -5,
            "sector_bias": {"bank": 0.5},
            "ticker_bias": {"aaa": -0.5},
            "calibration_targets": {
                "liquidity": {"adtv_multiple": 5},
                "market_filter": {"idx_drop_q": 0.05},
            },
            "execution": {
                "fill": {
                    "max_chase_ticks": 5,
                    "target_prob": 1.2,
                    "horizon_s": 5,
                    "window_sigma_s": 5,
                    "window_vol_s": 10,
                    "cancel_ratio_per_min": 2.0,
                    "joiner_factor": 1.0,
                    "no_cross": False,
                }
            },
        }
        normalised = calibrate_ai_overrides._normalise_overrides(raw)
        self.assertEqual(normalised["rationale"], "Test")
        self.assertAlmostEqual(normalised["buy_budget_frac"], 0.30)
        self.assertEqual(normalised["add_max"], 20)
        self.assertEqual(normalised["new_max"], 0)
        self.assertEqual(normalised["sector_bias"], {"BANK": 0.2})
        self.assertEqual(normalised["ticker_bias"], {"AAA": -0.2})
        fill = normalised["execution"]["fill"]
        self.assertEqual(fill["max_chase_ticks"], 2)
        self.assertAlmostEqual(fill["target_prob"], 0.95)
        self.assertEqual(fill["horizon_s"], 10)
        self.assertEqual(fill["window_sigma_s"], 15)
        self.assertEqual(fill["window_vol_s"], 30)
        self.assertAlmostEqual(fill["cancel_ratio_per_min"], 0.90)
        self.assertAlmostEqual(fill["joiner_factor"], 0.50)
        self.assertFalse(fill["no_cross"])

    def test_normalise_overrides_rejects_unknown_key(self):
        with self.assertRaises(SystemExit):
            calibrate_ai_overrides._normalise_overrides({"foo": 1})

    def test_normalise_execution_rejects_base_alias(self):
        raw = {
            "execution": {
                "fill": {
                    "base": {
                        "horizon_s": 75,
                        "target_prob": 0.55,
                        "max_chase_ticks": 2,
                    }
                }
            }
        }
        with self.assertRaises(SystemExit):
            calibrate_ai_overrides._normalise_overrides(raw)

    def test_write_ai_overrides_audit(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            overrides_path = base / "policy_ai_overrides.json"
            audit_path = base / "audit.ndjson"
            overrides_path.write_text("{\"buy_budget_frac\": 0.1}", encoding="utf-8")
            with patch.object(calibrate_ai_overrides, "AI_OVERRIDES_PATH", overrides_path), \
                 patch.object(calibrate_ai_overrides, "AUDIT_PATH", audit_path):
                calibrate_ai_overrides._write_ai_overrides({"buy_budget_frac": 0.2})
            saved = json.loads(overrides_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["buy_budget_frac"], 0.2)
            lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertTrue(lines)
            entry = json.loads(lines[-1])
            self.assertIn("keys_changed", entry)
            self.assertIn("buy_budget_frac", entry["changes"])

    def test_calibrate_skips_when_codex_missing(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            config_dir = base / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "policy_default.json").write_text("{}", encoding="utf-8")
            with patch.object(calibrate_ai_overrides, "CONFIG_DIR", config_dir), \
                 patch.object(calibrate_ai_overrides, "BASE_DIR", base), \
                 patch.object(calibrate_ai_overrides, "OUT_DIR", base / "out"), \
                 patch.object(calibrate_ai_overrides, "AI_OVERRIDES_PATH", config_dir / "policy_ai_overrides.json"), \
                 patch.object(calibrate_ai_overrides, "AUDIT_PATH", base / "out" / "debug" / "audit.ndjson"), \
                 patch("scripts.tuning.calibrators.calibrate_ai_overrides.shutil.which", return_value=None):
                result = calibrate_ai_overrides.calibrate(write=True)
                self.assertEqual(result, {})
                self.assertFalse((config_dir / "policy_ai_overrides.json").exists())

    def test_calibrate_fails_when_codex_required(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            config_dir = base / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "policy_default.json").write_text("{}", encoding="utf-8")
            with patch.object(calibrate_ai_overrides, "CONFIG_DIR", config_dir), \
                 patch.object(calibrate_ai_overrides, "BASE_DIR", base), \
                 patch.object(calibrate_ai_overrides, "OUT_DIR", base / "out"), \
                 patch.object(calibrate_ai_overrides, "AI_OVERRIDES_PATH", config_dir / "policy_ai_overrides.json"), \
                 patch.object(calibrate_ai_overrides, "AUDIT_PATH", base / "out" / "debug" / "audit.ndjson"), \
                 patch("scripts.tuning.calibrators.calibrate_ai_overrides.shutil.which", return_value=None), \
                 patch.dict(os.environ, {"BROKER_REQUIRE_CODEX": "1"}, clear=False):
                with self.assertRaises(SystemExit):
                    calibrate_ai_overrides.calibrate(write=True)
