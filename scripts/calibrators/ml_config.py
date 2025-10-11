"""Utilities for loading ML calibrator configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CFG_PATH = BASE_DIR / "config" / "ml_config.yaml"


@dataclass(frozen=True)
class MLConfig:
    raw: Dict[str, Any]

    @property
    def inference(self) -> Dict[str, Any]:
        return dict(self.raw.get("inference", {}) or {})

    @property
    def execution(self) -> Dict[str, Any]:
        return dict(self.raw.get("execution", {}) or {})

    @property
    def io_paths(self) -> Dict[str, str]:
        paths = ((self.raw.get("io", {}) or {}).get("paths") or {})
        return {str(k): str(v) for k, v in paths.items()}


def load_config(path: Path | None = None) -> MLConfig:
    cfg_path = path or DEFAULT_CFG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing ML config at {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid ML config structure in {cfg_path}; expected mapping")
    return MLConfig(raw=data)
