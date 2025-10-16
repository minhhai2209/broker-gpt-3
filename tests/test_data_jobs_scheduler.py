from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.data_fetching.job_config import DEFAULT_CONFIG_PATH, load_jobs_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_jobs_config_defaults():
    cfg = load_jobs_config(DEFAULT_CONFIG_PATH)
    assert cfg.version == 1
    expected_log_dir = (REPO_ROOT / "out" / "logs" / "data_jobs").resolve()
    assert cfg.log_dir == expected_log_dir
    nightly = cfg.get_group("nightly")
    assert nightly.default_max_workers == 2
    fundamentals_job = next(j for j in nightly.jobs if j.name == "collect_vietstock_fundamentals")
    assert fundamentals_job.command[0] == sys.executable
    assert fundamentals_job.allow_parallel is False
    real_time = cfg.get_group("real_time")
    assert real_time.default_max_workers == 1
    intraday_job = real_time.jobs[0]
    assert "--mode" in intraday_job.command
    assert "ensure-latest" in intraday_job.command


def test_run_data_jobs_dry_run():
    result = subprocess.run(
        [sys.executable, "scripts/data_fetching/run_data_jobs.py", "--dry-run", "--group", "nightly"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "[dry-run]" in result.stdout


def test_run_data_jobs_single_job():
    result = subprocess.run(
        [sys.executable, "scripts/data_fetching/run_data_jobs.py", "--dry-run", "--job", "collect_global_factors"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "[dry-run] nightly/collect_global_factors" in result.stdout
