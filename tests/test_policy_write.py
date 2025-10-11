import json
from pathlib import Path
import unittest

from scripts.tuning.calibrators.policy_write import (
    expect_calibrations,
    get_written_calibrations,
    reset_tracking,
    verify_calibrations,
    write_policy,
)


class TestPolicyWrite(unittest.TestCase):
    def setUp(self) -> None:
        reset_tracking()

    def tearDown(self) -> None:
        reset_tracking()

    def test_write_policy_tracks_and_persists(self):
        target = Path(self._tmp_dir(), 'policy.json')
        expect_calibrations(['foo'])
        payload = {'hello': 'world'}
        write_policy(calibrator='foo', policy=payload, explicit_path=target)
        data = json.loads(target.read_text(encoding='utf-8'))
        self.assertEqual(data, payload)
        self.assertEqual(verify_calibrations(), set())

    def test_verify_reports_missing_calibrator(self):
        target = Path(self._tmp_dir(), 'policy.json')
        expect_calibrations(['foo', 'bar'])
        write_policy(calibrator='foo', policy={}, explicit_path=target)
        self.assertEqual(verify_calibrations(), {'bar'})

    def test_write_policy_requires_target_paths_without_explicit(self):
        with self.assertRaises(ValueError):
            write_policy(
                calibrator='foo',
                policy={},
                orders_path=Path('orders.json'),
                config_path=None,
            )

    def test_write_policy_prefers_existing_orders_path(self):
        tmp_dir = self._tmp_dir()
        orders_target = tmp_dir / 'orders-policy.json'
        config_target = tmp_dir / 'config-policy.json'
        orders_target.write_text('{}', encoding='utf-8')

        path = write_policy(
            calibrator='foo',
            policy={'a': 1},
            orders_path=orders_target,
            config_path=config_target,
        )

        self.assertEqual(path, orders_target)
        self.assertEqual(json.loads(orders_target.read_text(encoding='utf-8')), {'a': 1})

    def test_write_policy_falls_back_to_config_path(self):
        tmp_dir = self._tmp_dir()
        orders_target = tmp_dir / 'orders-policy.json'
        config_target = tmp_dir / 'config-policy.json'

        path = write_policy(
            calibrator='foo',
            policy={'fallback': True},
            orders_path=orders_target,
            config_path=config_target,
        )

        self.assertEqual(path, config_target)
        self.assertFalse(orders_target.exists())
        self.assertEqual(json.loads(config_target.read_text(encoding='utf-8')), {'fallback': True})

    def test_written_calibrations_exposed(self):
        target = Path(self._tmp_dir(), 'policy.json')
        write_policy(calibrator='foo', policy={}, explicit_path=target)
        write_policy(calibrator='bar', policy={}, explicit_path=target)
        self.assertEqual(get_written_calibrations(), {'foo', 'bar'})

    def _tmp_dir(self) -> Path:
        from tempfile import TemporaryDirectory

        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name)
        path.mkdir(parents=True, exist_ok=True)
        return path


if __name__ == '__main__':
    unittest.main()
