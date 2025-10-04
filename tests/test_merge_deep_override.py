import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.engine import config_io


class TestMergeDeepOverride(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        (base / 'config').mkdir(parents=True, exist_ok=True)
        (base / 'out' / 'orders').mkdir(parents=True, exist_ok=True)
        # Patch paths
        self.patchers = [
            patch.object(config_io, 'BASE_DIR', base),
            patch.object(config_io, 'OUT_DIR', base / 'out'),
            patch.object(config_io, 'OUT_ORDERS_DIR', base / 'out' / 'orders'),
            patch.object(config_io, 'OVERRIDE_SRC', base / 'config' / 'policy_overrides.json'),
        ]
        for p in self.patchers:
            p.start(); self.addCleanup(p.stop)

    def _dest(self) -> Path:
        return config_io.OUT_ORDERS_DIR / 'policy_overrides.json'

    def test_base_overrides_defaults_nested_thresholds(self):
        base_pol = {'thresholds': {'tp_floor_pct': 0.04}}
        defs_pol = {'thresholds': {'tp_floor_pct': 0.06, 'sl_floor_pct': 0.03}}
        config_io.OVERRIDE_SRC.write_text(json.dumps(base_pol), encoding='utf-8')
        (config_io.BASE_DIR / 'config' / 'policy_for_calibration.json').write_text(json.dumps(defs_pol), encoding='utf-8')
        path = config_io.ensure_policy_override_file()
        obj = json.loads(self._dest().read_text(encoding='utf-8'))
        self.assertEqual(path, self._dest())
        self.assertEqual(obj['thresholds']['tp_floor_pct'], 0.04)
        self.assertEqual(obj['thresholds']['sl_floor_pct'], 0.03)

