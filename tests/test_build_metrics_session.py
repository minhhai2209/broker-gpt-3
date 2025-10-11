from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import pandas as pd

import scripts.build_metrics as bm


ORIGINAL_DATETIME = bm.datetime


class _MorningDateTime(ORIGINAL_DATETIME):
    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        base = ORIGINAL_DATETIME(2025, 9, 23, 9, 5)
        return base if tz is None else ORIGINAL_DATETIME(2025, 9, 23, 9, 5, tzinfo=tz)


class _PreOpenDateTime(ORIGINAL_DATETIME):
    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        base = ORIGINAL_DATETIME(2025, 9, 23, 8, 30)
        return base if tz is None else ORIGINAL_DATETIME(2025, 9, 23, 8, 30, tzinfo=tz)


class _WeekendDateTime(ORIGINAL_DATETIME):
    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        base = ORIGINAL_DATETIME(2025, 9, 27, 13, 30)
        return base if tz is None else ORIGINAL_DATETIME(2025, 9, 27, 13, 30, tzinfo=tz)


class TestBuildMetricsSession(unittest.TestCase):
    def tearDown(self) -> None:
        bm.datetime = ORIGINAL_DATETIME

    @staticmethod
    def _write_latest(dir_path: Path, ts_str: str) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    'Ticker': 'AAA',
                    'Ts': 1758527100,
                    'Price': 10.0,
                    'RSI14': 50.0,
                    'TimeVN': ts_str,
                }
            ]
        ).to_csv(dir_path / 'latest.csv', index=False)

    def test_infer_session_context_raises_when_intraday_stale_during_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / 'intraday'
            self._write_latest(dataset_dir, '2025-09-22 14:45:00')
            bm.datetime = _MorningDateTime
            with self.assertRaises(RuntimeError) as ctx:
                bm.infer_session_context(dataset_dir)
            msg = str(ctx.exception)
            self.assertIn('stale', msg)
            self.assertIn('2025-09-22 14:45:00', msg)
            self.assertIn('2025-09-23 09:05:00', msg)

    def test_infer_session_context_allows_preopen_with_previous_day_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / 'intraday'
            self._write_latest(dataset_dir, '2025-09-22 14:45:00')
            bm.datetime = _PreOpenDateTime
            phase, in_session = bm.infer_session_context(dataset_dir)
            self.assertEqual(phase, 'pre')
            self.assertEqual(in_session, 0)

    def test_infer_session_context_weekend_tolerates_previous_day_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / 'intraday'
            self._write_latest(dataset_dir, '2025-09-26 14:45:00')
            bm.datetime = _WeekendDateTime
            phase, in_session = bm.infer_session_context(dataset_dir)
            self.assertEqual(phase, 'post')
            self.assertEqual(in_session, 0)


if __name__ == '__main__':
    unittest.main()
