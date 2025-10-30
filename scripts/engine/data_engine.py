"""Standalone data engine that collects market data and pre-computes metrics."""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import pandas as pd
import yaml

from scripts.data_fetching.collect_intraday import ensure_intraday_latest_df
from scripts.data_fetching.fetch_ticker_data import ensure_and_load_history_df
from scripts.indicators import atr_wilder, ma, rsi_wilder, ema

LOGGER = logging.getLogger(__name__)


class ConfigurationError(RuntimeError):
    """Raised when the engine configuration file is invalid."""


@dataclass
class PresetConfig:
    name: str
    buy_tiers: List[float]
    sell_tiers: List[float]
    description: str | None = None

    @classmethod
    def from_dict(cls, name: str, raw: Dict[str, object]) -> "PresetConfig":
        if not isinstance(raw, dict):
            raise ConfigurationError(f"Preset '{name}' must be a mapping, got {type(raw).__name__}")
        buy = raw.get("buy_tiers", [])
        sell = raw.get("sell_tiers", [])
        if not isinstance(buy, list) or not all(isinstance(x, (int, float)) for x in buy):
            raise ConfigurationError(f"Preset '{name}' requires buy_tiers as a list of numbers")
        if not isinstance(sell, list) or not all(isinstance(x, (int, float)) for x in sell):
            raise ConfigurationError(f"Preset '{name}' requires sell_tiers as a list of numbers")
        desc = raw.get("description")
        if desc is not None and not isinstance(desc, str):
            raise ConfigurationError(f"Preset '{name}' description must be a string")
        return cls(name=name, buy_tiers=[float(x) for x in buy], sell_tiers=[float(x) for x in sell], description=desc)


@dataclass
class ShortlistFilterConfig:
    """Configuration for conservative preset shortlisting.

    The intention is to remove only the *very weak* tickers from preset outputs,
    while keeping everything else for consideration. Users can also force-keep or
    force-exclude specific tickers.
    """

    enabled: bool = False
    # Technical weakness thresholds (all must be met to exclude unless logic says otherwise)
    rsi14_max: Optional[float] = 25.0  # RSI_14 <= rsi14_max
    max_pct_to_lo_252: Optional[float] = 2.0  # PctToLo_252 <= X (near 52w low)
    return20_max: Optional[float] = -15.0  # Return_20 <= X
    return60_max: Optional[float] = -25.0  # Return_60 <= X
    require_below_sma50_and_200: bool = True  # LastPrice < SMA_50 and < SMA_200
    min_adv_20: Optional[float] = None  # ADV_20 <= X means illiquid (optional)
    # Compose: if True, require all active conditions to be true to drop; if False, any.
    drop_logic_all: bool = True
    # Manual overrides
    keep: List[str] | None = None
    exclude: List[str] | None = None

    def normalized_keep(self) -> List[str]:
        return sorted({t.strip().upper() for t in (self.keep or []) if isinstance(t, str) and t.strip()})

    def normalized_exclude(self) -> List[str]:
        return sorted({t.strip().upper() for t in (self.exclude or []) if isinstance(t, str) and t.strip()})

@dataclass
class EngineConfig:
    universe_csv: Path
    include_indices: bool
    moving_averages: List[int]
    rsi_periods: List[int]
    atr_periods: List[int]
    ema_periods: List[int]
    returns_periods: List[int]
    bollinger_windows: List[int]
    bollinger_k: float
    bollinger_include_bands: bool
    range_lookback_days: int
    adv_periods: List[int]
    macd_fast: int
    macd_slow: int
    macd_signal: int
    presets: Dict[str, PresetConfig]
    portfolio_dir: Path
    order_history_dir: Path
    output_base_dir: Path
    market_snapshot_path: Path
    presets_dir: Path
    portfolios_dir: Path
    diagnostics_dir: Path
    market_cache_dir: Path
    history_min_days: int
    intraday_window_minutes: int
    aggressiveness: str
    max_order_pct_adv: float
    slice_adv_ratio: float
    min_lot: int
    max_qty_per_order: int
    shortlist_filter: Optional[ShortlistFilterConfig]

    @classmethod
    def from_yaml(cls, path: Path) -> "EngineConfig":
        if not path.exists():
            raise ConfigurationError(f"Config file not found: {path}")
        config_dir = path.parent.resolve()
        repo_root = _find_repo_root(config_dir)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ConfigurationError("Engine config must be a mapping")
        uni = data.get("universe", {})
        if not isinstance(uni, dict):
            raise ConfigurationError("universe section must be a mapping")
        csv_path = uni.get("csv")
        if not isinstance(csv_path, str):
            raise ConfigurationError("universe.csv must be a string path")
        universe_csv = _resolve_path(csv_path, config_dir, repo_root)
        include_indices = bool(uni.get("include_indices", False))

        technical = data.get("technical_indicators", {}) or {}
        if not isinstance(technical, dict):
            raise ConfigurationError("technical_indicators must be a mapping")
        moving_averages = [int(x) for x in technical.get("moving_averages", [])]
        rsi_periods = [int(x) for x in technical.get("rsi_periods", [])]
        atr_periods = [int(x) for x in technical.get("atr_periods", [])]
        ema_periods = [int(x) for x in technical.get("ema_periods", [])]
        returns_periods = [int(x) for x in technical.get("returns_periods", [])]
        bb_cfg = technical.get("bollinger", {}) or {}
        if not isinstance(bb_cfg, dict):
            raise ConfigurationError("technical_indicators.bollinger must be a mapping if provided")
        bollinger_windows = [int(x) for x in bb_cfg.get("windows", [])]
        bollinger_k = float(bb_cfg.get("k", 2.0))
        bollinger_include_bands = bool(bb_cfg.get("include_bands", False))
        range_lookback_days = int(technical.get("range_lookback_days", 252))
        adv_periods = [int(x) for x in technical.get("adv_periods", [])]
        macd_cfg = technical.get("macd", {}) or {}
        if not isinstance(macd_cfg, dict):
            raise ConfigurationError("technical_indicators.macd must be a mapping")
        macd_fast = int(macd_cfg.get("fast", 12))
        macd_slow = int(macd_cfg.get("slow", 26))
        macd_signal = int(macd_cfg.get("signal", 9))

        raw_presets = data.get("presets", {}) or {}
        if not isinstance(raw_presets, dict):
            raise ConfigurationError("presets section must be a mapping if provided")
        presets = {name: PresetConfig.from_dict(name, cfg) for name, cfg in raw_presets.items()}

        portfolio_cfg = data.get("portfolio", {}) or {}
        if not isinstance(portfolio_cfg, dict):
            raise ConfigurationError("portfolio section must be a mapping")
        portfolio_dir = _resolve_path(
            portfolio_cfg.get("directory", "data/portfolios"), config_dir, repo_root
        )
        order_history_dir = _resolve_path(
            portfolio_cfg.get("order_history_directory", "data/order_history"), config_dir, repo_root
        )

        output_cfg = data.get("output", {}) or {}
        if not isinstance(output_cfg, dict):
            raise ConfigurationError("output section must be a mapping")
        output_base_dir = _resolve_path(output_cfg.get("base_dir", "out"), config_dir, repo_root)
        market_snapshot_rel = output_cfg.get("market_snapshot", "technical_snapshot.csv")
        presets_rel = output_cfg.get("presets_dir", ".")
        portfolios_rel = output_cfg.get("portfolios_dir", ".")
        diagnostics_rel = output_cfg.get("diagnostics_dir", ".")
        market_snapshot_path = (output_base_dir / market_snapshot_rel).resolve()
        presets_dir = (output_base_dir / presets_rel).resolve()
        portfolios_dir = (output_base_dir / portfolios_rel).resolve()
        diagnostics_dir = (output_base_dir / diagnostics_rel).resolve()

        data_cfg = data.get("data", {}) or {}
        if not isinstance(data_cfg, dict):
            raise ConfigurationError("data section must be a mapping")
        market_cache_dir = _resolve_path(data_cfg.get("history_cache", "out/data"), config_dir, repo_root)
        history_min_days = int(data_cfg.get("history_min_days", 400))
        intraday_window_minutes = int(data_cfg.get("intraday_window_minutes", 12 * 60))

        execution_cfg = data.get("execution", {}) or {}
        if not isinstance(execution_cfg, dict):
            raise ConfigurationError("execution section must be a mapping")
        aggressiveness = str(execution_cfg.get("aggressiveness", "med"))
        max_order_pct_adv = float(execution_cfg.get("max_order_pct_adv", 0.1))
        slice_adv_ratio = float(execution_cfg.get("slice_adv_ratio", 0.25))
        min_lot = int(execution_cfg.get("min_lot", 100))
        max_qty_per_order = int(execution_cfg.get("max_qty_per_order", 500_000))
        if min_lot <= 0:
            raise ConfigurationError("execution.min_lot must be positive")
        if max_qty_per_order <= 0 or max_qty_per_order > 500_000:
            raise ConfigurationError("execution.max_qty_per_order must be in (0, 500000]")
        if not 0 < max_order_pct_adv <= 1:
            raise ConfigurationError("execution.max_order_pct_adv must be in (0, 1]")
        if not 0 < slice_adv_ratio <= 1:
            raise ConfigurationError("execution.slice_adv_ratio must be in (0, 1]")

        # Optional shortlist filter
        filters_cfg = data.get("filters", {}) or {}
        if not isinstance(filters_cfg, dict):
            raise ConfigurationError("filters section must be a mapping if provided")
        shortlist_cfg_raw = filters_cfg.get("shortlist", {}) or {}
        if shortlist_cfg_raw is not None and not isinstance(shortlist_cfg_raw, dict):
            raise ConfigurationError("filters.shortlist must be a mapping if provided")
        shortlist_filter: Optional[ShortlistFilterConfig]
        if shortlist_cfg_raw:
            # Map YAML keys to dataclass fields with type coercion
            shortlist_filter = ShortlistFilterConfig(
                enabled=bool(shortlist_cfg_raw.get("enabled", False)),
                rsi14_max=(float(shortlist_cfg_raw["rsi14_max"]) if "rsi14_max" in shortlist_cfg_raw and shortlist_cfg_raw["rsi14_max"] is not None else ShortlistFilterConfig.rsi14_max),
                max_pct_to_lo_252=(float(shortlist_cfg_raw["max_pct_to_lo_252"]) if "max_pct_to_lo_252" in shortlist_cfg_raw and shortlist_cfg_raw["max_pct_to_lo_252"] is not None else ShortlistFilterConfig.max_pct_to_lo_252),
                return20_max=(float(shortlist_cfg_raw["return20_max"]) if "return20_max" in shortlist_cfg_raw and shortlist_cfg_raw["return20_max"] is not None else ShortlistFilterConfig.return20_max),
                return60_max=(float(shortlist_cfg_raw["return60_max"]) if "return60_max" in shortlist_cfg_raw and shortlist_cfg_raw["return60_max"] is not None else ShortlistFilterConfig.return60_max),
                require_below_sma50_and_200=bool(shortlist_cfg_raw.get("require_below_sma50_and_200", True)),
                min_adv_20=(float(shortlist_cfg_raw["min_adv_20"]) if "min_adv_20" in shortlist_cfg_raw and shortlist_cfg_raw["min_adv_20"] is not None else None),
                drop_logic_all=bool(shortlist_cfg_raw.get("drop_logic_all", True)),
                keep=[str(x).upper() for x in shortlist_cfg_raw.get("keep", [])],
                exclude=[str(x).upper() for x in shortlist_cfg_raw.get("exclude", [])],
            )
        else:
            shortlist_filter = None

        return cls(
            universe_csv=universe_csv,
            include_indices=include_indices,
            moving_averages=moving_averages,
            rsi_periods=rsi_periods,
            atr_periods=atr_periods,
            ema_periods=ema_periods,
            returns_periods=returns_periods,
            bollinger_windows=bollinger_windows,
            bollinger_k=bollinger_k,
            bollinger_include_bands=bollinger_include_bands,
            range_lookback_days=range_lookback_days,
            adv_periods=adv_periods,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            presets=presets,
            portfolio_dir=portfolio_dir,
            order_history_dir=order_history_dir,
            output_base_dir=output_base_dir,
            market_snapshot_path=market_snapshot_path,
            presets_dir=presets_dir,
            portfolios_dir=portfolios_dir,
            diagnostics_dir=diagnostics_dir,
            market_cache_dir=market_cache_dir,
            history_min_days=history_min_days,
            intraday_window_minutes=intraday_window_minutes,
            aggressiveness=aggressiveness,
            max_order_pct_adv=max_order_pct_adv,
            slice_adv_ratio=slice_adv_ratio,
            min_lot=min_lot,
            max_qty_per_order=max_qty_per_order,
            shortlist_filter=shortlist_filter,
        )


def _find_repo_root(start: Path) -> Path:
    current = start
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    return start


def _resolve_path(candidate: str, config_dir: Path, repo_root: Path) -> Path:
    if not isinstance(candidate, str):
        raise ConfigurationError(f"Expected string path, got {type(candidate).__name__}")
    path = Path(candidate)
    if path.is_absolute():
        return path.resolve()

    config_candidate = (config_dir / path).resolve()
    try:
        config_candidate.relative_to(repo_root)
    except ValueError:
        raise ConfigurationError(
            f"Path '{candidate}' escapes repository root {repo_root}. Use absolute paths for external locations."
        )
    if config_candidate.exists():
        return config_candidate

    root_candidate = (repo_root / path).resolve()
    try:
        root_candidate.relative_to(repo_root)
    except ValueError:
        raise ConfigurationError(
            f"Path '{candidate}' escapes repository root {repo_root}. Use absolute paths for external locations."
        )
    return root_candidate


def _tick_size(price: float) -> float:
    if price is None or pd.isna(price):
        return float("nan")
    value = float(price)
    if value < 10.0:
        return 0.01
    if value < 50.0:
        return 0.05
    return 0.10


def _as_decimal(value: float | int | str) -> Optional[Decimal]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _quantize_to_tick(value: float, rounding) -> float:
    dec_value = _as_decimal(value)
    if dec_value is None:
        return float("nan")
    tick = _as_decimal(_tick_size(float(dec_value)))
    if tick is None or tick == 0:
        return float("nan")
    steps = (dec_value / tick).to_integral_value(rounding=rounding)
    result = steps * tick
    return float(result)


def round_to_tick(value: float) -> float:
    return _quantize_to_tick(value, ROUND_HALF_UP)


def floor_to_tick(value: float) -> float:
    return _quantize_to_tick(value, ROUND_FLOOR)


def ceil_to_tick(value: float) -> float:
    return _quantize_to_tick(value, ROUND_CEILING)


def clamp_price(value: float, floor_value: float, ceil_value: float) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    if floor_value is not None and not pd.isna(floor_value):
        value = max(value, float(floor_value))
    if ceil_value is not None and not pd.isna(ceil_value):
        value = min(value, float(ceil_value))
    return value


class MarketDataService(Protocol):
    """Interface for loading market data."""

    def load_history(self, tickers: Sequence[str]) -> pd.DataFrame:  # pragma: no cover - protocol definition
        ...

    def load_intraday(self, tickers: Sequence[str]) -> pd.DataFrame:  # pragma: no cover - protocol definition
        ...


class VndirectMarketDataService:
    """Production data source backed by VNDIRECT APIs."""

    def __init__(self, config: EngineConfig) -> None:
        self._config = config

    def load_history(self, tickers: Sequence[str]) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame(columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume", "t"])
        return ensure_and_load_history_df(
            list(tickers),
            outdir=str(self._config.market_cache_dir),
            min_days=self._config.history_min_days,
            resolution="D",
        )

    def load_intraday(self, tickers: Sequence[str]) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame(columns=["Ticker", "Ts", "Price", "RSI14", "TimeVN"])
        return ensure_intraday_latest_df(list(tickers), window_minutes=self._config.intraday_window_minutes)


class TechnicalSnapshotBuilder:
    def __init__(self, config: EngineConfig) -> None:
        self._config = config

    def build(self, history_df: pd.DataFrame, intraday_df: pd.DataFrame, industry_df: pd.DataFrame) -> pd.DataFrame:
        if history_df.empty:
            raise RuntimeError("No historical prices available to build technical snapshot")
        if "Ticker" not in history_df.columns:
            raise ValueError("History dataframe missing 'Ticker' column")
        if "Close" not in history_df.columns:
            raise ValueError("History dataframe missing 'Close' column")
        intraday_lookup = {
            str(row.Ticker).upper(): row for row in intraday_df.itertuples(index=False)
        }
        industry_lookup = {}
        if "Ticker" in industry_df.columns:
            sector_col = "Sector" if "Sector" in industry_df.columns else None
            for row in industry_df.itertuples(index=False):
                ticker = str(getattr(row, "Ticker")).upper()
                sector = getattr(row, sector_col) if sector_col else None
                industry_lookup[ticker] = sector
        rows: List[Dict[str, object]] = []
        for ticker, ticker_df in history_df.groupby("Ticker"):
            ticker = str(ticker).upper()
            series = ticker_df.sort_values("t" if "t" in ticker_df.columns else "Date").reset_index(drop=True)
            if series.empty:
                continue
            last = series.iloc[-1]
            last_close = float(last.get("Close", float("nan")))
            last_volume = float(last.get("Volume", 0.0)) if not pd.isna(last.get("Volume")) else 0.0
            prev_close = float("nan")
            if len(series) >= 2:
                prev = series.iloc[-2]
                prev_close = float(prev.get("Close", float("nan")))
            close_series = pd.to_numeric(series["Close"], errors="coerce")
            high_series = pd.to_numeric(series.get("High"), errors="coerce") if "High" in series else close_series
            low_series = pd.to_numeric(series.get("Low"), errors="coerce") if "Low" in series else close_series
            vol_series = pd.to_numeric(series.get("Volume"), errors="coerce") if "Volume" in series else pd.Series([float("nan")]*len(close_series))

            data: Dict[str, object] = {
                "Ticker": ticker,
                "Sector": industry_lookup.get(ticker, ""),
                "LastClose": last_close,
                "PreviousClose": prev_close,
                "Volume": last_volume,
            }
            if ticker in intraday_lookup:
                snap = intraday_lookup[ticker]
                price = float(getattr(snap, "Price", last_close) or last_close)
                updated = getattr(snap, "TimeVN", "")
                source = "intraday"
            else:
                price = last_close
                updated = str(last.get("Date", ""))
                source = "close"
            data["LastPrice"] = price
            if not pd.isna(prev_close) and prev_close:
                data["ChangePct"] = (price / prev_close - 1.0) * 100.0
            else:
                data["ChangePct"] = float("nan")
            data["LastUpdated"] = updated
            data["PriceSource"] = source

            for window in self._config.moving_averages:
                if window <= 0:
                    continue
                col = f"SMA_{window}"
                if len(close_series) >= window:
                    data[col] = float(ma(close_series, window).iloc[-1])
                else:
                    data[col] = float("nan")
            # EMA
            for period in self._config.ema_periods:
                if period <= 0:
                    continue
                col = f"EMA_{period}"
                if len(close_series) >= period:
                    data[col] = float(ema(close_series, period).iloc[-1])
                else:
                    data[col] = float("nan")
            for period in self._config.rsi_periods:
                if period <= 0:
                    continue
                col = f"RSI_{period}"
                if len(close_series) > period:
                    data[col] = float(rsi_wilder(close_series, period).iloc[-1])
                else:
                    data[col] = float("nan")
            for period in self._config.atr_periods:
                if period <= 0:
                    continue
                col = f"ATR_{period}"
                if len(close_series) > period:
                    atr_val = float(atr_wilder(high_series, low_series, close_series, period).iloc[-1])
                    data[col] = atr_val
                    # ATR percentage vs last price
                    if last_close and not pd.isna(last_close) and last_close != 0:
                        data[f"ATRPct_{period}"] = atr_val / last_close * 100.0
                    else:
                        data[f"ATRPct_{period}"] = float("nan")
                else:
                    data[col] = float("nan")
                    data[f"ATRPct_{period}"] = float("nan")
            if len(close_series) >= max(self._config.macd_fast, self._config.macd_slow):
                close_numeric = pd.to_numeric(close_series, errors="coerce")
                macd_line = ema(close_numeric, self._config.macd_fast) - ema(
                    close_numeric, self._config.macd_slow
                )
                macd_signal = macd_line.ewm(span=self._config.macd_signal, adjust=False).mean()
                macd_value = float(macd_line.iloc[-1])
                macd_signal_value = float(macd_signal.iloc[-1])
                data["MACD"] = macd_value
                data["MACDSignal"] = macd_signal_value
                data["MACD_Hist"] = macd_value - macd_signal_value
            else:
                data["MACD"] = float("nan")
                data["MACDSignal"] = float("nan")
                data["MACD_Hist"] = float("nan")
            # Bollinger and z-score
            for w in self._config.bollinger_windows:
                if w <= 1:
                    continue
                if len(close_series) >= w:
                    sma_w = ma(close_series, w).iloc[-1]
                    std_w = pd.to_numeric(close_series, errors="coerce").rolling(w).std().iloc[-1]
                    if pd.isna(sma_w) or pd.isna(std_w) or std_w == 0:
                        data[f"Z_{w}"] = float("nan")
                    else:
                        data[f"Z_{w}"] = float((last_close - float(sma_w)) / float(std_w))
                    if self._config.bollinger_include_bands:
                        k = self._config.bollinger_k
                        data[f"BBU_{w}_{int(k)}"] = float(sma_w + k * std_w) if not pd.isna(sma_w) and not pd.isna(std_w) else float("nan")
                        data[f"BBL_{w}_{int(k)}"] = float(sma_w - k * std_w) if not pd.isna(sma_w) and not pd.isna(std_w) else float("nan")
                else:
                    data[f"Z_{w}"] = float("nan")
                    if self._config.bollinger_include_bands:
                        k = self._config.bollinger_k
                        data[f"BBU_{w}_{int(k)}"] = float("nan")
                        data[f"BBL_{w}_{int(k)}"] = float("nan")
            # Rolling returns
            for p in self._config.returns_periods:
                if p <= 0:
                    continue
                col = f"Return_{p}"
                if len(close_series) > p and close_series.iloc[-p-1] and not pd.isna(close_series.iloc[-p-1]):
                    base = float(close_series.iloc[-p-1])
                    data[col] = (last_close / base - 1.0) * 100.0 if base else float("nan")
                else:
                    data[col] = float("nan")
            # ADV periods
            for p in self._config.adv_periods:
                if p <= 0:
                    continue
                col = f"ADV_{p}"
                if len(vol_series) >= p:
                    data[col] = float(pd.to_numeric(vol_series, errors="coerce").rolling(p).mean().iloc[-1])
                else:
                    data[col] = float("nan")
            # 52-week range context (close-based hi/lo)
            L = self._config.range_lookback_days
            if L > 1 and len(high_series) >= 1:
                # Use High/Low if available; fallback to Close
                window = min(L, len(high_series))
                max_hi = float(pd.concat([high_series.tail(window)], axis=1).max().iloc[-1]) if "High" in series else float(close_series.tail(window).max())
                min_lo = float(pd.concat([low_series.tail(window)], axis=1).min().iloc[-1]) if "Low" in series else float(close_series.tail(window).min())
                data["Hi_252"] = max_hi if L == 252 else max_hi
                data["Lo_252"] = min_lo if L == 252 else min_lo
                if max_hi and not pd.isna(max_hi):
                    data["PctFromHi_252"] = (last_close / max_hi - 1.0) * 100.0
                else:
                    data["PctFromHi_252"] = float("nan")
                if min_lo and not pd.isna(min_lo):
                    data["PctToLo_252"] = (last_close / min_lo - 1.0) * 100.0
                else:
                    data["PctToLo_252"] = float("nan")
            rows.append(data)
        snapshot = pd.DataFrame(rows)
        snapshot = snapshot.sort_values("Ticker").reset_index(drop=True)
        return snapshot


@dataclass
class PortfolioReport:
    positions: pd.DataFrame
    sector: pd.DataFrame


@dataclass
class PortfolioRefreshResult:
    reports: Dict[str, PortfolioReport]
    aggregate_positions: pd.DataFrame
    aggregate_sector: pd.DataFrame
    holdings: Dict[str, float]


class PortfolioReporter:
    def __init__(self, config: EngineConfig, industry_df: pd.DataFrame) -> None:
        self._config = config
        self._industry = industry_df

    def refresh(self, snapshot: pd.DataFrame) -> PortfolioRefreshResult:
        portfolios_dir = self._config.portfolio_dir
        if not portfolios_dir.exists():
            LOGGER.info("Portfolio directory %s does not exist; skipping reports", portfolios_dir)
            empty_positions = pd.DataFrame(
                columns=[
                    "Ticker",
                    "Quantity",
                    "AvgPrice",
                    "Last",
                    "MarketValue_kVND",
                    "CostBasis_kVND",
                    "Unrealized_kVND",
                    "PNLPct",
                ]
            )
            empty_sector = pd.DataFrame(columns=["Sector", "MarketValue_kVND", "WeightPct", "PNLPct"])
            return PortfolioRefreshResult({}, empty_positions, empty_sector, {})
        sector_lookup = {}
        if "Ticker" in self._industry.columns:
            for row in self._industry.itertuples(index=False):
                ticker = str(getattr(row, "Ticker")).upper()
                sector = getattr(row, "Sector", "") if hasattr(row, "Sector") else ""
                sector_lookup[ticker] = sector
        reports: Dict[str, PortfolioReport] = {}
        all_positions: List[pd.DataFrame] = []
        # New layout: each subfolder is a profile with portfolio.csv
        processed_profiles = set()
        for profile_dir in sorted(portfolios_dir.glob("*")):
            if not profile_dir.is_dir():
                continue
            file = profile_dir / "portfolio.csv"
            profile = profile_dir.name
            try:
                portfolio_df = pd.read_csv(file)
            except Exception as exc:  # pragma: no cover - defensive path
                LOGGER.warning("Failed to read portfolio %s: %s", file, exc)
                continue
            if portfolio_df.empty:
                continue
            if "Ticker" not in portfolio_df.columns or "Quantity" not in portfolio_df.columns or "AvgPrice" not in portfolio_df.columns:
                LOGGER.warning("Portfolio %s missing required columns", file)
                continue
            report = self._build_report(portfolio_df, snapshot, sector_lookup)
            if report is None:
                continue
            reports[profile] = report
            all_positions.append(report.positions.copy())
            processed_profiles.add(profile)

        # Legacy layout: also support direct CSVs under portfolio_dir
        for file in sorted(portfolios_dir.glob("*.csv")):
            profile = file.stem
            if profile in processed_profiles:
                continue
            try:
                portfolio_df = pd.read_csv(file)
            except Exception as exc:  # pragma: no cover - defensive path
                LOGGER.warning("Failed to read portfolio %s: %s", file, exc)
                continue
            if portfolio_df.empty:
                continue
            if "Ticker" not in portfolio_df.columns or "Quantity" not in portfolio_df.columns or "AvgPrice" not in portfolio_df.columns:
                LOGGER.warning("Portfolio %s missing required columns", file)
                continue
            report = self._build_report(portfolio_df, snapshot, sector_lookup)
            if report is None:
                continue
            reports[profile] = report
            all_positions.append(report.positions.copy())

        if not all_positions:
            empty_positions = pd.DataFrame(
                columns=[
                    "Ticker",
                    "Quantity",
                    "AvgPrice",
                    "Last",
                    "MarketValue_kVND",
                    "CostBasis_kVND",
                    "Unrealized_kVND",
                    "PNLPct",
                ]
            )
            empty_sector = pd.DataFrame(columns=["Sector", "MarketValue_kVND", "WeightPct", "PNLPct"])
            return PortfolioRefreshResult(reports, empty_positions, empty_sector, {})

        combined = pd.concat(all_positions, ignore_index=True)
        combined["Sector"] = combined["Sector"].fillna("Không rõ")
        agg_rows: List[Dict[str, object]] = []
        holdings: Dict[str, float] = {}
        for ticker, group in combined.groupby("Ticker", dropna=False):
            qty = float(group["Quantity"].sum())
            holdings[str(ticker)] = qty
            if qty == 0:
                avg_price = float("nan")
            else:
                avg_price = float((group["AvgPrice"] * group["Quantity"]).sum() / qty)
            last_vals = group["Last"].dropna()
            last_price = float(last_vals.iloc[0]) if not last_vals.empty else float("nan")
            market_value = float(group["MarketValue_kVND"].sum())
            cost_basis = float(group["CostBasis_kVND"].sum())
            unrealized = float(group["Unrealized_kVND"].sum())
            pnl_pct = float(unrealized / cost_basis) if cost_basis else float("nan")
            agg_rows.append(
                {
                    "Ticker": str(ticker),
                    "Quantity": qty,
                    "AvgPrice": avg_price,
                    "Last": last_price,
                    "MarketValue_kVND": market_value,
                    "CostBasis_kVND": cost_basis,
                    "Unrealized_kVND": unrealized,
                    "PNLPct": pnl_pct,
                }
            )
        aggregate_positions = pd.DataFrame(agg_rows).sort_values("Ticker").reset_index(drop=True)

        sector_summary = combined.groupby("Sector", dropna=False).agg(
            MarketValue_kVND=("MarketValue_kVND", "sum"),
            CostBasis_kVND=("CostBasis_kVND", "sum"),
        )
        total_market = float(sector_summary["MarketValue_kVND"].sum()) if not sector_summary.empty else 0.0
        if total_market:
            sector_summary["WeightPct"] = sector_summary["MarketValue_kVND"] / total_market
        else:
            sector_summary["WeightPct"] = 0.0
        sector_summary["PNLPct"] = sector_summary.apply(
            lambda row: ((row["MarketValue_kVND"] - row["CostBasis_kVND"]) / row["CostBasis_kVND"]) if row["CostBasis_kVND"] else float("nan"),
            axis=1,
        )
        sector_summary = (
            sector_summary.drop(columns=["CostBasis_kVND"]).reset_index().rename(columns={"index": "Sector"})
        )
        sector_summary = sector_summary.sort_values("Sector").reset_index(drop=True)

        return PortfolioRefreshResult(reports, aggregate_positions, sector_summary, holdings)

    def _build_report(
        self,
        portfolio_df: pd.DataFrame,
        snapshot: pd.DataFrame,
        sector_lookup: Dict[str, str],
    ) -> Optional[PortfolioReport]:
        df = portfolio_df.copy()
        df["Ticker"] = df["Ticker"].astype(str).str.upper()
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0)
        df["AvgPrice"] = pd.to_numeric(df["AvgPrice"], errors="coerce").fillna(0.0)
        df = df[df["Quantity"] != 0]
        if df.empty:
            return None
        merged = df.merge(snapshot, on="Ticker", how="left")
        if "Sector" not in merged.columns:
            merged["Sector"] = pd.Series([float("nan")] * len(merged), index=merged.index)
        merged["Sector"] = merged["Sector"].fillna(merged["Ticker"].map(sector_lookup))
        merged["LastPrice"] = pd.to_numeric(merged.get("LastPrice"), errors="coerce")
        merged["LastClose"] = pd.to_numeric(merged.get("LastClose"), errors="coerce")
        merged["Last"] = merged["LastPrice"].fillna(merged["LastClose"])
        merged["Last"] = pd.to_numeric(merged["Last"], errors="coerce")
        merged["MarketValue_kVND"] = merged["Quantity"] * merged["Last"].fillna(0.0)
        merged["CostBasis_kVND"] = merged["Quantity"] * merged["AvgPrice"].fillna(0.0)
        merged["Unrealized_kVND"] = merged["MarketValue_kVND"] - merged["CostBasis_kVND"]
        merged["PNLPct"] = merged.apply(
            lambda row: (row["Unrealized_kVND"] / row["CostBasis_kVND"]) if row["CostBasis_kVND"] else float("nan"),
            axis=1,
        )
        sectorized = merged[[
            "Ticker",
            "Sector",
            "Quantity",
            "AvgPrice",
            "Last",
            "MarketValue_kVND",
            "CostBasis_kVND",
            "Unrealized_kVND",
            "PNLPct",
        ]].copy()
        sectorized["Sector"] = sectorized["Sector"].fillna("Không rõ")

        positions_output = sectorized.sort_values("Ticker").reset_index(drop=True)

        total_market = float(sectorized["MarketValue_kVND"].sum())
        sector_summary = sectorized.groupby("Sector", dropna=False).agg(
            MarketValue_kVND=("MarketValue_kVND", "sum"),
            CostBasis_kVND=("CostBasis_kVND", "sum"),
        )
        if total_market:
            sector_summary["WeightPct"] = sector_summary["MarketValue_kVND"] / total_market
        else:
            sector_summary["WeightPct"] = 0.0
        sector_summary["PNLPct"] = sector_summary.apply(
            lambda row: ((row["MarketValue_kVND"] - row["CostBasis_kVND"]) / row["CostBasis_kVND"]) if row["CostBasis_kVND"] else float("nan"),
            axis=1,
        )
        sector_summary = (
            sector_summary.drop(columns=["CostBasis_kVND"]).reset_index().rename(columns={"index": "Sector"})
        )
        sector_summary = sector_summary.sort_values("Sector").reset_index(drop=True)

        return PortfolioReport(positions=positions_output, sector=sector_summary)


def _build_technical_output(snapshot: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Ticker",
        "Last",
        "Ref",
        "ChangePct",
        "SMA20",
        "SMA50",
        "SMA200",
        "EMA20",
        "RSI14",
        "ATR14",
        "MACD",
        "MACDSignal",
        "Z20",
        "Ret5d",
        "Ret20d",
        "ADV20",
        "High52w",
        "Low52w",
    ]
    if snapshot.empty:
        return pd.DataFrame(columns=columns)

    tickers = snapshot.get("Ticker", pd.Series([], dtype=object)).astype(str).str.upper()

    def numeric(name: str) -> pd.Series:
        if name in snapshot.columns:
            return pd.to_numeric(snapshot[name], errors="coerce")
        return pd.Series([float("nan")] * len(snapshot), index=snapshot.index)

    last = numeric("LastPrice")
    ref = numeric("LastClose")
    last = last.fillna(ref)
    ref = ref.fillna(last)
    change = pd.Series([float("nan")] * len(snapshot), index=snapshot.index)
    valid_mask = (~last.isna()) & (~ref.isna()) & (ref != 0)
    change.loc[valid_mask] = (last.loc[valid_mask] / ref.loc[valid_mask]) - 1.0

    sma20 = numeric("SMA_20")
    sma50 = numeric("SMA_50")
    sma200 = numeric("SMA_200")
    ema20 = numeric("EMA_20")
    rsi14 = numeric("RSI_14")
    atr14 = numeric("ATR_14")
    macd = numeric("MACD")
    macd_signal = numeric("MACDSignal")

    ret5 = numeric("Return_5") / 100.0
    ret20 = numeric("Return_20") / 100.0
    adv20 = numeric("ADV_20")
    hi52 = numeric("Hi_252")
    lo52 = numeric("Lo_252")

    z20 = pd.Series([float("nan")] * len(snapshot), index=snapshot.index)
    with_atr = (~atr14.isna()) & (atr14 != 0)
    z20.loc[with_atr] = ((last - sma20) / atr14).loc[with_atr]
    z20.loc[atr14 == 0] = 0.0

    technical = pd.DataFrame(
        {
            "Ticker": tickers,
            "Last": last,
            "Ref": ref,
            "ChangePct": change,
            "SMA20": sma20,
            "SMA50": sma50,
            "SMA200": sma200,
            "EMA20": ema20,
            "RSI14": rsi14,
            "ATR14": atr14,
            "MACD": macd,
            "MACDSignal": macd_signal,
            "Z20": z20,
            "Ret5d": ret5,
            "Ret20d": ret20,
            "ADV20": adv20,
            "High52w": hi52,
            "Low52w": lo52,
        }
    )
    technical = technical.sort_values("Ticker").reset_index(drop=True)
    return technical


def _build_bands(technical: pd.DataFrame) -> pd.DataFrame:
    columns = ["Ticker", "Ref", "Ceil", "Floor", "TickSize"]
    if technical.empty:
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, object]] = []
    for row in technical.itertuples(index=False):
        ref = float(getattr(row, "Ref", float("nan")))
        if math.isnan(ref):
            ceil_val = float("nan")
            floor_val = float("nan")
            tick_size = float("nan")
        else:
            tick_size = _tick_size(ref)
            ceil_val = floor_to_tick(ref * 1.07)
            floor_val = ceil_to_tick(ref * 0.93)
            if not math.isnan(ceil_val) and not math.isnan(floor_val) and ceil_val < floor_val:
                ceil_val, floor_val = floor_val, ceil_val
        rows.append(
            {
                "Ticker": getattr(row, "Ticker"),
                "Ref": ref,
                "Ceil": ceil_val,
                "Floor": floor_val,
                "TickSize": tick_size,
            }
        )
    df = pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True)
    return df


def _round_and_clamp(value: float, floor_value: float, ceil_value: float) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    rounded = round_to_tick(value)
    if pd.isna(rounded):
        return float("nan")
    return clamp_price(rounded, floor_value, ceil_value)


def _build_levels(technical: pd.DataFrame, bands: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Ticker",
        "Preset",
        "SideDeclared",
        "NearTouchBuy",
        "NearTouchSell",
        "Opp1Buy",
        "Opp1Sell",
        "Opp2Buy",
        "Opp2Sell",
    ]
    if technical.empty:
        return pd.DataFrame(columns=columns)
    merged = technical.merge(bands, on="Ticker", how="left", suffixes=("", "_band"))
    rows: List[Dict[str, object]] = []
    presets = [
        ("momentum", "BUY"),
        ("mean_reversion", "BOTH"),
        ("balanced", "BOTH"),
        ("risk_off", "SELL"),
    ]
    for row in merged.itertuples(index=False):
        ticker = getattr(row, "Ticker")
        last = float(getattr(row, "Last", float("nan")))
        sma20 = float(getattr(row, "SMA20", float("nan")))
        atr14 = float(getattr(row, "ATR14", float("nan")))
        floor_value = float(getattr(row, "Floor", float("nan")))
        ceil_value = float(getattr(row, "Ceil", float("nan")))
        base_tick = _tick_size(last if not math.isnan(last) else getattr(row, "Ref", float("nan")))

        def near_values(preset: str) -> Tuple[float, float]:
            if preset == "momentum":
                if math.isnan(base_tick):
                    return float("nan"), float("nan")
                return last + base_tick, last + 2 * base_tick
            if preset == "mean_reversion":
                if not math.isnan(sma20) and not math.isnan(atr14):
                    return sma20 - 0.5 * atr14, sma20 + 0.5 * atr14
                if math.isnan(base_tick):
                    return float("nan"), float("nan")
                return last - base_tick, last + base_tick
            if preset == "balanced":
                anchor = last
                if not math.isnan(sma20):
                    if math.isnan(anchor):
                        anchor = sma20
                    else:
                        anchor = (anchor + sma20) / 2.0
                if math.isnan(base_tick):
                    return anchor, anchor
                return anchor - base_tick, anchor + base_tick
            if preset == "risk_off":
                if math.isnan(base_tick):
                    return float("nan"), float("nan")
                return float("nan"), last - base_tick
            return float("nan"), float("nan")

        for preset, side in presets:
            near_buy_raw, near_sell_raw = near_values(preset)
            near_buy = _round_and_clamp(near_buy_raw, floor_value, ceil_value)
            near_sell = _round_and_clamp(near_sell_raw, floor_value, ceil_value)
            opp1_buy = float("nan")
            opp2_buy = float("nan")
            opp1_sell = float("nan")
            opp2_sell = float("nan")
            if not math.isnan(near_buy) and not math.isnan(sma20) and not math.isnan(atr14):
                opp1_buy = _round_and_clamp(min(near_buy, sma20 - 0.5 * atr14), floor_value, ceil_value)
                opp2_buy = _round_and_clamp(min(near_buy, sma20 - 1.0 * atr14), floor_value, ceil_value)
            if not math.isnan(near_sell) and not math.isnan(sma20) and not math.isnan(atr14):
                opp1_sell = _round_and_clamp(max(near_sell, sma20 + 0.5 * atr14), floor_value, ceil_value)
                opp2_sell = _round_and_clamp(max(near_sell, sma20 + 1.0 * atr14), floor_value, ceil_value)
            rows.append(
                {
                    "Ticker": ticker,
                    "Preset": preset,
                    "SideDeclared": side,
                    "NearTouchBuy": near_buy,
                    "NearTouchSell": near_sell,
                    "Opp1Buy": opp1_buy,
                    "Opp1Sell": opp1_sell,
                    "Opp2Buy": opp2_buy,
                    "Opp2Sell": opp2_sell,
                }
            )
    levels = pd.DataFrame(rows)
    levels = levels.sort_values(["Ticker", "Preset"]).reset_index(drop=True)
    return levels


def _round_to_lot(value: float, lot: int, mode: str = "round") -> int:
    if value is None or pd.isna(value) or lot <= 0:
        return 0
    units = float(value) / lot
    if mode == "floor":
        qty = math.floor(units)
    elif mode == "ceil":
        qty = math.ceil(units)
    else:
        qty = round(units)
    return int(qty * lot)


def _build_sizing(
    technical: pd.DataFrame,
    holdings: Dict[str, float],
    config: EngineConfig,
) -> pd.DataFrame:
    columns = [
        "Ticker",
        "TargetQty",
        "CurrentQty",
        "DeltaQty",
        "MaxOrderQty",
        "SliceCount",
        "SliceQty",
        "LiquidityScore_kVND",
        "VolatilityScore",
        "TodayFilledQty",
        "TodayWAP",
    ]
    if technical.empty:
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, object]] = []
    for row in technical.itertuples(index=False):
        ticker = getattr(row, "Ticker")
        last = float(getattr(row, "Last", float("nan")))
        atr = float(getattr(row, "ATR14", float("nan")))
        adv20 = float(getattr(row, "ADV20", float("nan")))
        current_qty = float(holdings.get(ticker, 0.0))
        target_qty = current_qty
        delta_qty = target_qty - current_qty
        max_by_adv = config.max_order_pct_adv * adv20 if not math.isnan(adv20) else 0.0
        raw_max_order = min(max_by_adv, float(config.max_qty_per_order)) if adv20 and adv20 > 0 else 0.0
        max_order_qty = _round_to_lot(raw_max_order, config.min_lot, mode="floor")
        abs_delta = abs(delta_qty)
        if abs_delta == 0:
            slice_count = 0
        else:
            adv_slice = config.slice_adv_ratio * adv20 if not math.isnan(adv20) else 0.0
            if adv_slice <= 0:
                slice_count = 1
            else:
                slice_count = max(1, math.ceil(abs_delta / adv_slice))
        if slice_count > 0:
            slice_qty = _round_to_lot(abs_delta / slice_count, config.min_lot)
        else:
            slice_qty = 0
        liquidity = float(adv20 * last) if not math.isnan(adv20) and not math.isnan(last) else float("nan")
        if not math.isnan(last) and last > 0 and not math.isnan(atr):
            volatility = atr / last
        else:
            volatility = 0.0
        rows.append(
            {
                "Ticker": ticker,
                "TargetQty": int(round(target_qty)),
                "CurrentQty": int(round(current_qty)),
                "DeltaQty": int(round(delta_qty)),
                "MaxOrderQty": int(max_order_qty),
                "SliceCount": int(slice_count),
                "SliceQty": int(slice_qty),
                "LiquidityScore_kVND": liquidity,
                "VolatilityScore": volatility,
                "TodayFilledQty": 0,
                "TodayWAP": float("nan"),
            }
        )
    sizing = pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True)
    return sizing


def _build_signals(technical: pd.DataFrame, bands: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Ticker",
        "PresetFitMomentum",
        "PresetFitMeanRev",
        "PresetFitBalanced",
        "BandDistance",
        "SectorBias",
        "RiskGuards",
    ]
    if technical.empty:
        return pd.DataFrame(columns=columns)
    snapshot_meta = snapshot[["Ticker", "PriceSource"]] if "PriceSource" in snapshot.columns else pd.DataFrame(columns=["Ticker", "PriceSource"])
    merged = technical.merge(bands, on="Ticker", how="left").merge(snapshot_meta, on="Ticker", how="left")
    rows: List[Dict[str, object]] = []
    for row in merged.itertuples(index=False):
        ticker = getattr(row, "Ticker")
        rsi = float(getattr(row, "RSI14", float("nan")))
        ret20 = float(getattr(row, "Ret20d", float("nan")))
        macd = float(getattr(row, "MACD", float("nan")))
        macd_signal = float(getattr(row, "MACDSignal", float("nan")))
        z20 = float(getattr(row, "Z20", float("nan")))
        atr = float(getattr(row, "ATR14", float("nan")))
        last = float(getattr(row, "Last", float("nan")))
        floor_value = float(getattr(row, "Floor", float("nan")))
        ceil_value = float(getattr(row, "Ceil", float("nan")))
        adv20 = float(getattr(row, "ADV20", float("nan")))
        price_source = getattr(row, "PriceSource", "")

        momentum_score = 0.0
        if not math.isnan(rsi) and rsi >= 55:
            momentum_score += 0.4
        if not math.isnan(ret20) and ret20 >= 0:
            momentum_score += 0.3
        if not math.isnan(macd) and not math.isnan(macd_signal) and macd > macd_signal:
            momentum_score += 0.3
        momentum_score = min(momentum_score, 1.0)

        mean_rev_score = 0.0
        if not math.isnan(z20) and z20 <= -0.5:
            mean_rev_score += 0.5
        if not math.isnan(atr) and atr > 0 and not math.isnan(last) and not math.isnan(floor_value):
            if (last - floor_value) <= atr:
                mean_rev_score += 0.5
        mean_rev_score = min(mean_rev_score, 1.0)

        balanced = (momentum_score + mean_rev_score) / 2.0

        if math.isnan(atr):
            band_distance = float("nan")
        elif atr == 0:
            band_distance = 0.0
        else:
            candidates = []
            if not math.isnan(ceil_value) and not math.isnan(last):
                candidates.append((ceil_value - last) / atr)
            if not math.isnan(last) and not math.isnan(floor_value):
                candidates.append((last - floor_value) / atr)
            band_distance = min(candidates) if candidates else float("nan")

        risk_flags: List[str] = []
        if math.isnan(atr) or atr == 0:
            risk_flags.append("ZERO_ATR")
        if math.isnan(adv20) or adv20 < 10_000:
            risk_flags.append("LOW_LIQ")
        tick = _tick_size(last)
        if not math.isnan(tick) and not math.isnan(last):
            if (not math.isnan(ceil_value) and (ceil_value - last) <= tick) or (not math.isnan(floor_value) and (last - floor_value) <= tick):
                risk_flags.append("NEAR_LIMIT")
        if str(price_source).lower() != "intraday":
            risk_flags.append("STALE_SNAPSHOT")

        rows.append(
            {
                "Ticker": ticker,
                "PresetFitMomentum": momentum_score,
                "PresetFitMeanRev": mean_rev_score,
                "PresetFitBalanced": balanced,
                "BandDistance": band_distance,
                "SectorBias": 0,
                "RiskGuards": "|".join(sorted(set(risk_flags))),
            }
        )
    signals = pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True)
    return signals


def _build_limits(config: EngineConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Aggressiveness": config.aggressiveness,
                "MaxOrderPctADV": config.max_order_pct_adv,
                "SliceADVRatio": config.slice_adv_ratio,
                "MinLot": config.min_lot,
                "MaxQtyPerOrder": config.max_qty_per_order,
            }
        ]
    )
class DataEngine:
    """Coordinates data collection, indicator computation, and report generation."""

    def __init__(self, config: EngineConfig, data_service: MarketDataService) -> None:
        self._config = config
        self._data_service = data_service

    def run(self) -> Dict[str, object]:
        self._wipe_output_dir()
        self._prepare_directories()
        universe_df = self._load_universe()
        tickers = self._resolve_tickers(universe_df)
        LOGGER.info("Processing %d tickers", len(tickers))
        history_df = self._data_service.load_history(tickers)
        intraday_df = self._data_service.load_intraday(tickers)
        snapshot_builder = TechnicalSnapshotBuilder(self._config)
        snapshot = snapshot_builder.build(history_df, intraday_df, universe_df)
        technical_df = _build_technical_output(snapshot)
        bands_df = _build_bands(technical_df)
        levels_df = _build_levels(technical_df, bands_df)
        portfolio_result = PortfolioReporter(self._config, universe_df).refresh(snapshot)
        sizing_df = _build_sizing(technical_df, portfolio_result.holdings, self._config)
        signals_df = _build_signals(technical_df, bands_df, snapshot)
        limits_df = _build_limits(self._config)
        positions_df = portfolio_result.aggregate_positions
        sector_df = portfolio_result.aggregate_sector

        out_dir = self._config.output_base_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        technical_path = out_dir / "technical.csv"
        bands_path = out_dir / "bands.csv"
        levels_path = out_dir / "levels.csv"
        sizing_path = out_dir / "sizing.csv"
        signals_path = out_dir / "signals.csv"
        limits_path = out_dir / "limits.csv"
        positions_path = out_dir / "positions.csv"
        sector_path = out_dir / "sector.csv"

        outputs = [
            (technical_path, technical_df),
            (bands_path, bands_df),
            (levels_path, levels_df),
            (sizing_path, sizing_df),
            (signals_path, signals_df),
            (limits_path, limits_df),
            (positions_path, positions_df),
            (sector_path, sector_df),
        ]
        for path, df in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)

        self._zip_prompt_files(
            portfolio_result,
            [technical_path, bands_path, levels_path, sizing_path, signals_path, limits_path],
        )
        return {
            "tickers": len(tickers),
            "snapshot_rows": len(snapshot),
            "output": str(technical_path),
        }

    def _prepare_directories(self) -> None:
        self._config.output_base_dir.mkdir(parents=True, exist_ok=True)
        self._config.presets_dir.mkdir(parents=True, exist_ok=True)
        self._config.portfolios_dir.mkdir(parents=True, exist_ok=True)
        self._config.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        self._config.market_cache_dir.mkdir(parents=True, exist_ok=True)
        self._config.portfolio_dir.mkdir(parents=True, exist_ok=True)
        self._config.order_history_dir.mkdir(parents=True, exist_ok=True)

    def _load_universe(self) -> pd.DataFrame:
        if not self._config.universe_csv.exists():
            raise RuntimeError(f"Universe CSV not found: {self._config.universe_csv}")
        df = pd.read_csv(self._config.universe_csv)
        if "Ticker" not in df.columns:
            raise RuntimeError("Universe CSV must contain 'Ticker' column")
        df["Ticker"] = df["Ticker"].astype(str).str.upper()
        if not self._config.include_indices:
            df = df[~df["Ticker"].str.contains("VNINDEX|VN30|VN100", na=False)]
        return df

    def _resolve_tickers(self, universe_df: pd.DataFrame) -> List[str]:
        tickers = set(universe_df["Ticker"].tolist())
        if self._config.portfolio_dir.exists():
            # New layout: data/portfolios/<profile>/portfolio.csv
            for profile_dir in sorted(self._config.portfolio_dir.glob("*")):
                p_csv = profile_dir / "portfolio.csv"
                if p_csv.is_file():
                    try:
                        df = pd.read_csv(p_csv, usecols=["Ticker"])  # validate required column
                    except Exception:
                        continue
                    for t in df["Ticker"].dropna().astype(str):
                        tickers.add(t.upper())
            # Legacy layout: data/portfolios/<profile>.csv
            for file in self._config.portfolio_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(file, usecols=["Ticker"])  # validate required column
                except Exception:
                    continue
                for t in df["Ticker"].dropna().astype(str):
                    tickers.add(t.upper())
        clean = sorted({t.strip().upper() for t in tickers if t and isinstance(t, str)})
        return clean

    def _wipe_output_dir(self) -> None:
        out_dir = self._config.output_base_dir
        # Defensive: only remove when the leaf is named 'out'
        if out_dir.exists():
            if out_dir.name != "out":
                raise RuntimeError(f"Refusing to delete non-standard output directory: {out_dir}")
            LOGGER.info("Wiping output directory: %s", out_dir)
            shutil.rmtree(out_dir)

    def _discover_profiles(self) -> List[str]:
        profiles: List[str] = []
        pf_dir = self._config.portfolio_dir
        if pf_dir.exists():
            for d in sorted(pf_dir.glob("*")):
                if d.is_dir() and (d / "portfolio.csv").is_file():
                    profiles.append(d.name)
            for f in sorted(pf_dir.glob("*.csv")):
                profiles.append(f.stem)
        # De-duplicate while keeping order
        seen = set()
        out: List[str] = []
        for p in profiles:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def _zip_prompt_files(self, result: PortfolioRefreshResult, common_files: List[Path]) -> None:
        profiles = sorted(set(result.reports.keys()) | set(self._discover_profiles()))
        if not profiles:
            LOGGER.info("No profiles discovered; skip zipping prompt bundles")
            return
        bundle_dir = self._config.output_base_dir.resolve()
        bundle_dir.mkdir(parents=True, exist_ok=True)
        empty_positions = pd.DataFrame(
            columns=[
                "Ticker",
                "Quantity",
                "AvgPrice",
                "Last",
                "MarketValue_kVND",
                "CostBasis_kVND",
                "Unrealized_kVND",
                "PNLPct",
            ]
        )
        empty_sector = pd.DataFrame(columns=["Sector", "MarketValue_kVND", "WeightPct", "PNLPct"])
        for profile in profiles:
            zip_path = bundle_dir / f"bundle_{profile}.zip"
            with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                added = 0
                for src in common_files:
                    if src and src.exists() and src.is_file():
                        zf.write(src, arcname=src.name)
                        added += 1
                report = result.reports.get(profile)
                positions_df = report.positions if report else empty_positions
                sector_df = report.sector if report else empty_sector
                pos_out = positions_df[[
                    "Ticker",
                    "Quantity",
                    "AvgPrice",
                    "Last",
                    "MarketValue_kVND",
                    "CostBasis_kVND",
                    "Unrealized_kVND",
                    "PNLPct",
                ]]
                zf.writestr("positions.csv", pos_out.to_csv(index=False))
                zf.writestr("sector.csv", sector_df.to_csv(index=False))
                added += 2
                LOGGER.info("Wrote %s with %d files", zip_path, added)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Collect market data and compute technical indicators")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/data_engine.yaml"),
        help="Path to engine YAML configuration",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit summary as JSON to stdout",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    config = EngineConfig.from_yaml(Path(args.config))
    service = VndirectMarketDataService(config)
    engine = DataEngine(config, service)
    summary = engine.run()
    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
