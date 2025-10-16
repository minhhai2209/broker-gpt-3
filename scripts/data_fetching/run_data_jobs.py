#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.data_fetching.job_config import (
    DEFAULT_CONFIG_PATH,
    GroupSpec,
    JobSpec,
    load_jobs_config,
)


def _now_utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sanitize_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _build_log_path(base_dir: Path, group_name: str, job_name: str) -> Path:
    ts = _now_utc_ts()
    safe_group = _sanitize_filename(group_name)
    safe_job = _sanitize_filename(job_name)
    return base_dir / safe_group / f"{safe_group}__{safe_job}__{ts}.log"


def run_job(job: JobSpec, group_name: str, log_dir: Path, dry_run: bool = False) -> Optional[Path]:
    log_path = _build_log_path(log_dir, group_name, job.name)
    if dry_run:
        return log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env: Dict[str, str] = os.environ.copy()
    env.update(job.env)
    header = [
        f"# job: {job.name}",
        f"# group: {group_name}",
        f"# command: {' '.join(job.command)}",
        f"# cwd: {job.cwd}",
        f"# launched_at_utc: {_now_utc_ts()}",
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("\n".join(header) + "\n\n")
        log_file.flush()
        try:
            completed = subprocess.run(
                job.command,
                cwd=str(job.cwd),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
                timeout=job.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Job '{job.name}' timed out after {job.timeout_seconds} seconds (log: {log_path})"
            ) from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"Job '{job.name}' exited with code {completed.returncode} (see {log_path})"
        )
    return log_path


def _finalize_future(
    future: concurrent.futures.Future,
    futures: Dict[concurrent.futures.Future, JobSpec],
    group_name: str,
    dry_run: bool,
) -> Tuple[Optional[JobSpec], Optional[Exception]]:
    job = futures.pop(future, None)
    try:
        log_path = future.result()
        if dry_run:
            print(f"[dry-run] {group_name}/{job.name} -> {log_path}")
        else:
            print(f"[ok] {group_name}/{job.name} -> {log_path}")
        return job, None
    except Exception as exc:  # noqa: BLE001
        msg = f"[error] {group_name}/{job.name}: {exc}"
        print(msg, file=sys.stderr)
        return job, exc


def execute_group(
    group: GroupSpec,
    log_dir: Path,
    max_workers: int,
    dry_run: bool = False,
) -> None:
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    print(
        f"Running group '{group.name}' ({len(group.jobs)} jobs) "
        f"with max_workers={min(max_workers, len(group.jobs))}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    errors: List[Tuple[JobSpec, Exception]] = []
    futures: Dict[concurrent.futures.Future, JobSpec] = {}
    executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
    if max_workers > 1:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def wait_for_all() -> None:
        while futures:
            future = next(concurrent.futures.as_completed(list(futures.keys())))
            job, exc = _finalize_future(future, futures, group.name, dry_run)
            if exc is not None:
                errors.append((job, exc))

    for job in group.jobs:
        if executor and job.allow_parallel and not dry_run:
            print(f"[queued] {group.name}/{job.name}")
            future = executor.submit(run_job, job, group.name, log_dir, dry_run)
            futures[future] = job
            continue
        if futures:
            wait_for_all()
            if errors:
                break
        try:
            log_path = run_job(job, group.name, log_dir, dry_run=dry_run)
            if dry_run:
                print(f"[dry-run] {group.name}/{job.name} -> {log_path}")
            else:
                print(f"[ok] {group.name}/{job.name} -> {log_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {group.name}/{job.name}: {exc}", file=sys.stderr)
            errors.append((job, exc))
            break

    if executor:
        if errors:
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=True, cancel_futures=False)
    if not errors and futures:
        wait_for_all()
    if errors:
        details = "; ".join(f"{job.name}: {exc}" for job, exc in errors if job)
        raise RuntimeError(f"Group '{group.name}' failed: {details}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run configured data collection jobs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to data jobs config (default: config/data_jobs.json)",
    )
    parser.add_argument(
        "--group",
        action="append",
        dest="groups",
        help="Group name to run (can be specified multiple times). Defaults to 'nightly'.",
    )
    parser.add_argument(
        "--job",
        action="append",
        dest="jobs",
        help="Specific job name to run (can be specified multiple times). Overrides group selection when provided.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Override concurrency for all selected groups (default: per-group setting).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print planned jobs without executing them.",
    )
    parser.add_argument(
        "--list-groups",
        action="store_true",
        help="List available job groups and exit.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        config = load_jobs_config(args.config)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load data jobs config: {exc}", file=sys.stderr)
        return 2
    if args.list_groups:
        for group in config.groups:
            print(f"{group.name}\t{group.description}\tjobs={len(group.jobs)}")
        return 0

    job_lookup: Dict[str, Tuple[JobSpec, GroupSpec]] = {}
    for group in config.groups:
        for job in group.jobs:
            if job.name in job_lookup:
                # Keep the earliest group association but warn about duplicates
                prev_group = job_lookup[job.name][1].name
                raise RuntimeError(f"Duplicate job name '{job.name}' found in groups '{prev_group}' and '{group.name}'")
            job_lookup[job.name] = (job, group)

    if args.jobs:
        unknown_jobs = [name for name in args.jobs if name not in job_lookup]
        if unknown_jobs:
            print(f"Unknown job(s): {', '.join(unknown_jobs)}", file=sys.stderr)
            return 2
        errors: List[Tuple[str, Exception]] = []
        for name in args.jobs:
            job, group = job_lookup[name]
            try:
                run_job(job, group.name, config.log_dir, dry_run=args.dry_run)
                if args.dry_run:
                    print(f"[dry-run] {group.name}/{job.name}")
                else:
                    print(f"[ok] {group.name}/{job.name}")
            except Exception as exc:  # noqa: BLE001
                print(f"[error] {group.name}/{job.name}: {exc}", file=sys.stderr)
                errors.append((name, exc))
                break
        if errors:
            return 1
        return 0

    selected_groups = args.groups or ["nightly"]
    missing = [name for name in selected_groups if not any(g.name == name for g in config.groups)]
    if missing:
        print(f"Unknown group(s): {', '.join(missing)}", file=sys.stderr)
        return 2
    for name in selected_groups:
        group = config.get_group(name)
        max_workers = args.max_workers or group.default_max_workers
        try:
            execute_group(group, config.log_dir, max_workers=max_workers, dry_run=args.dry_run)
        except Exception as exc:  # noqa: BLE001
            print(str(exc), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
