"""Utility helpers for the backtest package."""

from __future__ import annotations

import importlib
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha1
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

VN_TICK_SIZE_TABLE: Dict[str, List[Tuple[float, float, float]]] = {
    "HOSE": [
        (0.0, 9_999.0, 10.0),
        (9_999.0, 49_999.0, 50.0),
        (49_999.0, float("inf"), 100.0),
    ],
    "HNX": [
        (0.0, 9_999.0, 10.0),
        (9_999.0, 49_999.0, 50.0),
        (49_999.0, float("inf"), 100.0),
    ],
    "UPCOM": [
        (0.0, 9_999.0, 10.0),
        (9_999.0, 49_999.0, 50.0),
        (49_999.0, float("inf"), 100.0),
    ],
}

DEFAULT_LOT_SIZE = 100


class ConfigError(RuntimeError):
    """Raised when configuration files are invalid or missing mandatory fields."""


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and OS-level randomness for deterministic backtests."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_directory(path: Path) -> Path:
    """Ensure a directory exists and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path: Path | str) -> Dict[str, Any]:
    """Load a YAML or JSON configuration file."""

    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Missing configuration file: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ConfigError(f"Empty configuration file: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - configuration error surfaced to user
            raise ConfigError("PyYAML is required to load YAML configurations") from exc
        data = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ConfigError(f"Unsupported configuration extension: {path.suffix}")
    if not isinstance(data, dict):
        raise ConfigError("Top-level configuration must be an object")
    return data


def load_callable(path: str) -> Callable[..., Any]:
    """Dynamically import a callable from a dotted path."""

    if not path:
        raise ConfigError("Empty callable path provided")
    module_path: str
    attr_name: str
    if ":" in path:
        module_path, attr_name = path.split(":", 1)
    else:
        if "." not in path:
            raise ConfigError(
                "Callable path must include a module and attribute name, e.g. 'pkg.module:func'"
            )
        module_path, attr_name = path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:  # pragma: no cover - surfaced to configuration users
        raise ConfigError(f"Cannot import module '{module_path}' for callable lookup") from exc
    if not hasattr(module, attr_name):
        raise ConfigError(f"Module '{module_path}' does not define '{attr_name}'")
    value = getattr(module, attr_name)
    if not callable(value):
        raise ConfigError(f"Resolved object '{module_path}.{attr_name}' is not callable")
    return value


def params_hash(params: Mapping[str, Any]) -> str:
    """Compute a stable hash for parameter dictionaries."""

    normalized = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return sha1(normalized.encode("utf-8")).hexdigest()


def tick_size(symbol: str, price: float, *, exchange: str | None = None) -> float:
    """Return the tick size for a given price under the Vietnamese schedule."""

    if price is None or price <= 0:
        return 0.0
    exchange = (exchange or "HOSE").upper()
    schedule = VN_TICK_SIZE_TABLE.get(exchange)
    if not schedule:
        raise ConfigError(f"Unknown exchange '{exchange}' for tick size lookup")
    for lower, upper, tick in schedule:
        if lower <= price < upper:
            return tick
    return schedule[-1][2]


def round_to_tick(price: float, tick: float) -> float:
    """Round a price to the nearest tick."""

    if tick <= 0:
        return price
    return round(price / tick) * tick


def round_quantity(qty: int, *, lot_size: int = DEFAULT_LOT_SIZE) -> int:
    """Round quantity to the nearest valid lot size."""

    if lot_size <= 0:
        return qty
    if qty % lot_size == 0:
        return qty
    return int(math.floor(qty / lot_size) * lot_size)


def business_days_between(calendar: Sequence[date], start: date, end: date) -> List[date]:
    """Return trading days between start and end inclusive based on the provided calendar."""

    if start > end:
        return []
    return [d for d in calendar if start <= d <= end]


def ensure_sorted_calendar(calendar: Iterable[date]) -> List[date]:
    dates = sorted(set(calendar))
    return dates


def next_trading_date(calendar: Sequence[date], current: date, offset: int) -> Optional[date]:
    """Return the next trading date offset from current (positive offset only)."""

    if offset <= 0:
        return current
    try:
        idx = calendar.index(current)
    except ValueError:
        return None
    target = idx + offset
    if target >= len(calendar):
        return None
    return calendar[target]


def as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return datetime.strptime(value, "%Y-%m-%d").date()
    raise TypeError(f"Cannot convert {value!r} to date")


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def cumulative_returns(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float((1.0 + series).prod() - 1.0)


def sharpe_ratio(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    mean = returns.mean()
    std = returns.std(ddof=0)
    if std == 0:
        return 0.0
    return float((mean / std) * math.sqrt(252))


def max_drawdown(nav_series: pd.Series) -> float:
    if nav_series.empty:
        return 0.0
    cummax = nav_series.cummax()
    drawdown = (nav_series / cummax) - 1.0
    return float(drawdown.min())


@dataclass(frozen=True)
class LiquidityModel:
    type: str
    participation_cap: float | None = None


@dataclass(frozen=True)
class SlippageModel:
    type: str
    value: float = 0.0


def normalize_liquidity_model(config: Mapping[str, Any] | None) -> LiquidityModel:
    cfg = dict(config or {})
    model_type = str(cfg.get("type", "touch_full")).lower()
    if model_type not in {"touch_full", "participation"}:
        raise ConfigError(f"Unsupported liquidity model: {model_type}")
    cap = cfg.get("participation_cap", 1.0)
    if model_type == "participation":
        cap = float(cap)
        if cap <= 0.0 or cap > 1.0:
            raise ConfigError("participation_cap must be in (0, 1]")
    else:
        cap = 1.0
    return LiquidityModel(type=model_type, participation_cap=cap)


def normalize_slippage_model(config: Mapping[str, Any] | None) -> SlippageModel:
    cfg = dict(config or {})
    model_type = str(cfg.get("type", "none")).lower()
    if model_type not in {"none", "bps", "ticks"}:
        raise ConfigError(f"Unsupported slippage model: {model_type}")
    value = float(cfg.get("value", 0.0) or 0.0)
    if value < 0.0:
        raise ConfigError("Slippage value must be non-negative")
    return SlippageModel(type=model_type, value=value)


def apply_slippage(price: float, side: str, slippage: SlippageModel, *, tick: float) -> float:
    if price is None:
        raise ValueError("Execution price cannot be None when applying slippage")
    if slippage.type == "none" or slippage.value == 0:
        return price
    if slippage.type == "bps":
        adjustment = price * (slippage.value / 10_000.0)
        return price + adjustment if side == "BUY" else price - adjustment
    if slippage.type == "ticks":
        adjustment = slippage.value * tick
        return price + adjustment if side == "BUY" else price - adjustment
    raise ConfigError(f"Unsupported slippage model: {slippage.type}")


def expand_params_grid(grid: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """Expand a parameter grid into a list of dictionaries."""

    keys = list(grid.keys())
    if not keys:
        return [{}]
    values_list = [list(grid[k]) for k in keys]
    combos: List[Dict[str, Any]] = []
    def _recurse(prefix: Dict[str, Any], idx: int) -> None:
        if idx == len(keys):
            combos.append(dict(prefix))
            return
        key = keys[idx]
        for value in values_list[idx]:
            prefix[key] = value
            _recurse(prefix, idx + 1)
        prefix.pop(key, None)
    _recurse({}, 0)
    return combos


def ensure_timezone(value: datetime | date | str, tz: str = "Asia/Ho_Chi_Minh") -> str:
    dt = value
    if isinstance(value, date) and not isinstance(value, datetime):
        dt = datetime.combine(value, datetime.min.time())
    if isinstance(dt, datetime):
        return dt.replace(tzinfo=None).isoformat() + f"[{tz}]"
    if isinstance(value, str):
        return value
    raise TypeError(f"Unsupported timestamp type: {type(value)!r}")
