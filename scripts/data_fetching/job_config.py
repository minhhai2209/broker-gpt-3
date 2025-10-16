from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "data_jobs.json"
PLACEHOLDER_PYTHON = "{python}"


@dataclass(frozen=True)
class JobSpec:
    name: str
    command: List[str]
    allow_parallel: bool = False
    cwd: Path = ROOT_DIR
    timeout_seconds: Optional[int] = None
    env: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class GroupSpec:
    name: str
    description: str
    jobs: List[JobSpec]
    default_max_workers: int = 1


@dataclass(frozen=True)
class JobsConfig:
    version: int
    log_dir: Path
    groups: List[GroupSpec]

    def get_group(self, name: str) -> GroupSpec:
        for group in self.groups:
            if group.name == name:
                return group
        raise KeyError(f"Group '{name}' not found in data jobs config")


def _ensure_command(value: Iterable) -> List[str]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("command must be a list or tuple of strings")
    items = [str(item) for item in value]
    if not items:
        raise ValueError("command cannot be empty")
    out: List[str] = []
    for token in items:
        if token == PLACEHOLDER_PYTHON:
            out.append(sys.executable)
        else:
            out.append(token)
    return out


def _to_path(base: Path, value: str | Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    return candidate


def load_jobs_config(config_path: Optional[Path] = None) -> JobsConfig:
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Data jobs config not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    version = int(raw.get("version", 0))
    if version != 1:
        raise ValueError(f"Unsupported data jobs config version: {version}")
    log_dir_raw = raw.get("log_dir")
    if not isinstance(log_dir_raw, str):
        raise TypeError("log_dir must be a string path")
    log_dir = _to_path(ROOT_DIR, log_dir_raw)
    groups_raw = raw.get("groups")
    if not isinstance(groups_raw, list) or not groups_raw:
        raise ValueError("groups must be a non-empty list")
    groups: List[GroupSpec] = []
    for entry in groups_raw:
        if not isinstance(entry, dict):
            raise TypeError("each group entry must be a dict")
        name = str(entry.get("name") or "").strip()
        if not name:
            raise ValueError("group missing required 'name'")
        desc = str(entry.get("description") or "").strip()
        jobs_raw = entry.get("jobs")
        if not isinstance(jobs_raw, list) or not jobs_raw:
            raise ValueError(f"group '{name}' must define at least one job")
        max_workers = int(entry.get("default_max_workers", 1))
        if max_workers < 1:
            raise ValueError(f"group '{name}' default_max_workers must be >= 1")
        jobs: List[JobSpec] = []
        for job_entry in jobs_raw:
            if not isinstance(job_entry, dict):
                raise TypeError(f"group '{name}' has non-dict job entry")
            job_name = str(job_entry.get("name") or "").strip()
            if not job_name:
                raise ValueError(f"group '{name}' has job missing 'name'")
            command = _ensure_command(job_entry.get("command"))
            allow_parallel = bool(job_entry.get("allow_parallel", False))
            cwd_raw = job_entry.get("cwd")
            cwd = _to_path(ROOT_DIR, cwd_raw) if cwd_raw else ROOT_DIR
            timeout_raw = job_entry.get("timeout_seconds")
            if timeout_raw is not None:
                timeout = int(timeout_raw)
                if timeout <= 0:
                    raise ValueError(f"job '{job_name}' timeout_seconds must be > 0")
            else:
                timeout = None
            env_raw = job_entry.get("env", {})
            if not isinstance(env_raw, dict):
                raise TypeError(f"job '{job_name}' env must be a dict of strings")
            env: Dict[str, str] = {str(k): str(v) for k, v in env_raw.items()}
            jobs.append(
                JobSpec(
                    name=job_name,
                    command=command,
                    allow_parallel=allow_parallel,
                    cwd=cwd,
                    timeout_seconds=timeout,
                    env=env,
                )
            )
        groups.append(GroupSpec(name=name, description=desc, jobs=jobs, default_max_workers=max_workers))
    return JobsConfig(version=version, log_dir=log_dir, groups=groups)
