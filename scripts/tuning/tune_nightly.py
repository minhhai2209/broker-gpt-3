from __future__ import annotations

"""Deprecated wrapper for backward compatibility. Calls unified tuner."""

from scripts.tuning.tune import main as unified_main  # type: ignore

if __name__ == "__main__":
    raise SystemExit(unified_main())
