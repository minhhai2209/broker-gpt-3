"""Standalone data engine that collects market data and pre-computes metrics."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence

import pandas as pd
import yaml

from scripts.data_fetching.collect_intraday import ensure_intraday_latest_df
from scripts.data_fetching.fetch_ticker_data import ensure_and_load_history_df
from scripts.indicators import atr_wilder, macd_hist, ma, rsi_wilder, ema

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

        raw_presets = data.get("presets", {})
        if not isinstance(raw_presets, dict) or not raw_presets:
            raise ConfigurationError("At least one preset must be defined")
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
        market_snapshot_rel = output_cfg.get("market_snapshot", "market/technical_snapshot.csv")
        presets_rel = output_cfg.get("presets_dir", "presets")
        portfolios_rel = output_cfg.get("portfolios_dir", "portfolios")
        diagnostics_rel = output_cfg.get("diagnostics_dir", "diagnostics")
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
                data["MACD_Hist"] = float(
                    macd_hist(
                        close_series,
                        fast=self._config.macd_fast,
                        slow=self._config.macd_slow,
                        signal=self._config.macd_signal,
                    ).iloc[-1]
                )
            else:
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


class PresetWriter:
    def __init__(self, presets: Dict[str, PresetConfig], output_dir: Path) -> None:
        self._presets = presets
        self._output_dir = output_dir

    def write(self, snapshot: pd.DataFrame) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        base_cols = ["Ticker", "Sector", "LastPrice", "LastClose", "PriceSource"]
        for name, preset in self._presets.items():
            if snapshot.empty:
                df = pd.DataFrame(columns=base_cols)
            else:
                df = snapshot[base_cols].copy()
                for idx, adj in enumerate(preset.buy_tiers, start=1):
                    col = f"Buy_{idx}"
                    df[col] = (df["LastPrice"] * (1.0 + float(adj))).round(4)
                for idx, adj in enumerate(preset.sell_tiers, start=1):
                    col = f"Sell_{idx}"
                    df[col] = (df["LastPrice"] * (1.0 + float(adj))).round(4)
            out_path = self._output_dir / f"{name}.csv"
            df.to_csv(out_path, index=False)


class PortfolioReporter:
    def __init__(self, config: EngineConfig, industry_df: pd.DataFrame) -> None:
        self._config = config
        self._industry = industry_df

    def refresh(self, snapshot: pd.DataFrame) -> None:
        portfolios_dir = self._config.portfolio_dir
        if not portfolios_dir.exists():
            LOGGER.info("Portfolio directory %s does not exist; skipping reports", portfolios_dir)
            return
        snapshot_lookup = snapshot.set_index("Ticker") if not snapshot.empty else pd.DataFrame().set_index([])
        sector_lookup = {}
        if "Ticker" in self._industry.columns:
            for row in self._industry.itertuples(index=False):
                ticker = str(getattr(row, "Ticker")).upper()
                sector = getattr(row, "Sector", "") if hasattr(row, "Sector") else ""
                sector_lookup[ticker] = sector
        self._config.portfolios_dir.mkdir(parents=True, exist_ok=True)
        for file in sorted(portfolios_dir.glob("*.csv")):
            profile = file.stem
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
            portfolio_df["Ticker"] = portfolio_df["Ticker"].astype(str).str.upper()
            portfolio_df["Quantity"] = pd.to_numeric(portfolio_df["Quantity"], errors="coerce").fillna(0.0)
            portfolio_df["AvgPrice"] = pd.to_numeric(portfolio_df["AvgPrice"], errors="coerce").fillna(0.0)
            portfolio_df = portfolio_df[portfolio_df["Quantity"] != 0]
            if portfolio_df.empty:
                continue
            merged = portfolio_df.merge(snapshot, on="Ticker", how="left")
            merged["Sector"] = merged["Sector"].fillna(merged["Ticker"].map(sector_lookup))
            merged["LastPrice"] = pd.to_numeric(merged["LastPrice"], errors="coerce")
            merged["LastClose"] = pd.to_numeric(merged["LastClose"], errors="coerce")
            merged["LastPrice"] = merged["LastPrice"].fillna(merged["LastClose"])
            merged["MarketValue"] = merged["Quantity"] * merged["LastPrice"].fillna(0.0)
            merged["CostBasis"] = merged["Quantity"] * merged["AvgPrice"].fillna(0.0)
            merged["UnrealizedPnL"] = merged["MarketValue"] - merged["CostBasis"]
            merged["UnrealizedPct"] = merged.apply(
                lambda row: ((row["LastPrice"] / row["AvgPrice"] - 1.0) * 100.0) if row["AvgPrice"] else float("nan"),
                axis=1,
            )
            positions_path = self._config.portfolios_dir / f"{profile}_positions.csv"
            merged.to_csv(positions_path, index=False)
            sector_summary = merged.groupby(merged["Sector"].fillna("Không rõ"), dropna=False).agg(
                Quantity=("Quantity", "sum"),
                MarketValue=("MarketValue", "sum"),
                CostBasis=("CostBasis", "sum"),
                UnrealizedPnL=("UnrealizedPnL", "sum"),
            )
            sector_summary["UnrealizedPct"] = sector_summary.apply(
                lambda row: ((row["MarketValue"] / row["CostBasis"] - 1.0) * 100.0) if row["CostBasis"] else float("nan"),
                axis=1,
            )
            sector_summary = sector_summary.reset_index().rename(columns={"index": "Sector"})
            sector_path = self._config.portfolios_dir / f"{profile}_sector.csv"
            sector_summary.to_csv(sector_path, index=False)


class DataEngine:
    """Coordinates data collection, indicator computation, and report generation."""

    def __init__(self, config: EngineConfig, data_service: MarketDataService) -> None:
        self._config = config
        self._data_service = data_service

    def run(self) -> Dict[str, object]:
        self._prepare_directories()
        universe_df = self._load_universe()
        tickers = self._resolve_tickers(universe_df)
        LOGGER.info("Processing %d tickers", len(tickers))
        history_df = self._data_service.load_history(tickers)
        intraday_df = self._data_service.load_intraday(tickers)
        snapshot_builder = TechnicalSnapshotBuilder(self._config)
        snapshot = snapshot_builder.build(history_df, intraday_df, universe_df)
        self._write_market_snapshot(snapshot)
        PresetWriter(self._config.presets, self._config.presets_dir).write(snapshot)
        PortfolioReporter(self._config, universe_df).refresh(snapshot)
        return {
            "tickers": len(tickers),
            "snapshot_rows": len(snapshot),
            "output": str(self._config.market_snapshot_path),
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
            for file in self._config.portfolio_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(file, usecols=["Ticker"])
                except Exception:
                    continue
                for t in df["Ticker"].dropna().astype(str):
                    tickers.add(t.upper())
        clean = sorted({t.strip().upper() for t in tickers if t and isinstance(t, str)})
        return clean

    def _write_market_snapshot(self, snapshot: pd.DataFrame) -> None:
        self._config.market_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.to_csv(self._config.market_snapshot_path, index=False)


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
