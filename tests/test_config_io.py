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

    def test_policy_file_env_takes_precedence(self):
        dest = self._dest()
        dest.unlink(missing_ok=True)
        override = config_io.BASE_DIR / "custom_policy.json"
        override.write_text('{"buy_budget_frac": 0.14}', encoding="utf-8")

        with patch.dict(os.environ, {"POLICY_FILE": str(override)}, clear=False):
            config_io.ensure_policy_override_file()

        self.assertEqual(dest.read_text(encoding="utf-8"), override.read_text(encoding="utf-8"))

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

