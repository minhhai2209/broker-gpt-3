from __future__ import annotations

"""Shared helpers for writing calibrated policy overlays with audit tracking."""

from pathlib import Path
from typing import Iterable, Mapping, MutableMapping
import json

_EXPECTED: set[str] = set()
_WRITTEN: set[str] = set()


def reset_tracking() -> None:
    """Clear tracking state before a new tuning session."""
    _EXPECTED.clear()
    _WRITTEN.clear()


def expect_calibrations(calibrators: Iterable[str]) -> None:
    """Register calibrators that are expected to persist policy changes."""
    for name in calibrators:
        if not name:
            continue
        _EXPECTED.add(str(name))


def resolve_policy_path(orders_path: Path, config_path: Path) -> Path:
    """Return the preferred policy path for writing (orders/ if present)."""
    orders_path = Path(orders_path)
    config_path = Path(config_path)
    return orders_path if orders_path.exists() else config_path


def write_policy(
    *,
    calibrator: str,
    policy: Mapping[str, object] | MutableMapping[str, object],
    orders_path: Path | None = None,
    config_path: Path | None = None,
    explicit_path: Path | None = None,
) -> Path:
    """Persist the provided policy object and record the writing calibrator.

    Args:
        calibrator: Fully-qualified module name performing the write.
        policy: JSON-serialisable policy object to persist.
        orders_path: Runtime overrides path (preferred target).
        config_path: Repository overrides path (fallback when runtime copy absent).
        explicit_path: When provided, overrides orders/config resolution entirely.

    Returns:
        Path to which the policy was written.
    """
    if explicit_path is not None:
        path = Path(explicit_path)
    else:
        if orders_path is None or config_path is None:
            raise ValueError("orders_path and config_path must be provided when explicit_path is None")
        path = resolve_policy_path(Path(orders_path), Path(config_path))

    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(policy, ensure_ascii=False, indent=2)
    path.write_text(data, encoding="utf-8")
    if calibrator:
        _WRITTEN.add(str(calibrator))
    return path


def verify_calibrations() -> set[str]:
    """Return the set of expected calibrators that have not written policy."""
    return set(_EXPECTED - _WRITTEN)


def get_written_calibrations() -> set[str]:
    """Expose the set of calibrators that have persisted changes."""
    return set(_WRITTEN)
