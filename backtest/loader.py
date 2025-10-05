"""Data loading and context management for backtests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import pandas as pd

from .utils import (
    ConfigError,
    as_date,
    business_days_between,
    ensure_sorted_calendar,
    load_config,
)


@dataclass
class LoaderConfig:
    prices_path: Path
    calendar_path: Path | None = None
    corporate_actions_path: Path | None = None
    initial_portfolio_path: Path | None = None
    watchlist_path: Path | None = None
    indicators_path: Path | None = None
    intraday_dir: Path | None = None
    exchange: str = "HOSE"
    order_builder: str | None = None


class BacktestLoader:
    """Loader that provides per-day contexts for the backtest runner."""

    def __init__(self, config: LoaderConfig, *, raw_config: Mapping[str, Any] | None = None):
        self.config = config
        self.raw_config = dict(raw_config or {})
        self._prices: pd.DataFrame | None = None
        self._calendar: List[date] | None = None
        self._initial_portfolio: pd.DataFrame | None = None
        self._watchlist: pd.DataFrame | None = None
        self._indicators: pd.DataFrame | None = None
        self._corporate_actions: pd.DataFrame | None = None
        self._order_builder: Callable[..., Any] | None = None
        if self.config.order_builder:
            from .utils import load_callable

            self._order_builder = load_callable(self.config.order_builder)
        self._engine_config = dict(self.raw_config.get("config") or {})

    @classmethod
    def from_config(cls, path: Path | str) -> "BacktestLoader":
        cfg = load_config(path)
        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> "BacktestLoader":
        data_cfg = cfg.get("data")
        if not isinstance(data_cfg, Mapping):
            raise ConfigError("config.data must be a mapping")
        prices_path = Path(data_cfg.get("prices")) if data_cfg.get("prices") else None
        if prices_path is None:
            raise ConfigError("config.data.prices is required")
        calendar_path = (
            Path(data_cfg.get("calendar")) if data_cfg.get("calendar") else None
        )
        corporate_actions_path = (
            Path(data_cfg.get("corporate_actions"))
            if data_cfg.get("corporate_actions")
            else None
        )
        portfolio_path = (
            Path(data_cfg.get("initial_portfolio"))
            if data_cfg.get("initial_portfolio")
            else None
        )
        watchlist_path = (
            Path(data_cfg.get("watchlist")) if data_cfg.get("watchlist") else None
        )
        indicators_path = (
            Path(data_cfg.get("indicators")) if data_cfg.get("indicators") else None
        )
        intraday_dir = Path(data_cfg.get("intraday_dir")) if data_cfg.get("intraday_dir") else None
        exchange = str(cfg.get("exchange", "HOSE"))
        engine_cfg = cfg.get("engine") if isinstance(cfg.get("engine"), Mapping) else {}
        builder_path = None
        if isinstance(engine_cfg, Mapping):
            builder_path = engine_cfg.get("builder")
        loader_cfg = LoaderConfig(
            prices_path=prices_path,
            calendar_path=calendar_path,
            corporate_actions_path=corporate_actions_path,
            initial_portfolio_path=portfolio_path,
            watchlist_path=watchlist_path,
            indicators_path=indicators_path,
            intraday_dir=intraday_dir,
            exchange=exchange,
            order_builder=builder_path,
        )
        return cls(loader_cfg, raw_config=cfg)

    # ------------------------------------------------------------------
    # Lazy loading helpers
    # ------------------------------------------------------------------

    @property
    def prices(self) -> pd.DataFrame:
        if self._prices is None:
            if not self.config.prices_path.exists():
                raise ConfigError(f"Missing prices file: {self.config.prices_path}")
            df = pd.read_csv(self.config.prices_path)
            required_cols = {"date", "symbol", "open", "high", "low", "close"}
            missing = required_cols - set(map(str.lower, df.columns))
            if missing:
                raise ConfigError(
                    "Prices file must contain columns: date, symbol, open, high, low, close"
                )
            # Normalize column names
            df.columns = [c.lower() for c in df.columns]
            df["date"] = pd.to_datetime(df["date"]).dt.date
            self._prices = df
        return self._prices

    @property
    def calendar(self) -> List[date]:
        if self._calendar is None:
            if self.config.calendar_path and self.config.calendar_path.exists():
                cal_df = pd.read_csv(self.config.calendar_path)
                if "date" not in cal_df.columns:
                    raise ConfigError("Trading calendar must have a 'date' column")
                dates = [as_date(v) for v in cal_df["date"].tolist()]
            else:
                dates = sorted(self.prices["date"].unique())
            self._calendar = ensure_sorted_calendar(dates)
        return self._calendar

    @property
    def initial_portfolio(self) -> pd.DataFrame:
        if self._initial_portfolio is None:
            if self.config.initial_portfolio_path is None:
                columns = ["symbol", "qty", "avg_cost"]
                self._initial_portfolio = pd.DataFrame(columns=columns)
            else:
                self._initial_portfolio = pd.read_csv(self.config.initial_portfolio_path)
        return self._initial_portfolio

    @property
    def watchlist(self) -> pd.DataFrame:
        if self._watchlist is None:
            if self.config.watchlist_path and self.config.watchlist_path.exists():
                self._watchlist = pd.read_csv(self.config.watchlist_path)
            else:
                self._watchlist = pd.DataFrame(columns=["symbol", "priority"])
        return self._watchlist

    @property
    def indicators(self) -> pd.DataFrame:
        if self._indicators is None:
            if self.config.indicators_path and self.config.indicators_path.exists():
                self._indicators = pd.read_csv(self.config.indicators_path)
            else:
                self._indicators = pd.DataFrame()
        return self._indicators

    @property
    def corporate_actions(self) -> pd.DataFrame:
        if self._corporate_actions is None:
            if self.config.corporate_actions_path and self.config.corporate_actions_path.exists():
                self._corporate_actions = pd.read_csv(self.config.corporate_actions_path)
            else:
                self._corporate_actions = pd.DataFrame()
        return self._corporate_actions

    # ------------------------------------------------------------------
    # Context API
    # ------------------------------------------------------------------

    def get_trading_days(self, start: date, end: date) -> List[date]:
        return business_days_between(self.calendar, start, end)

    def get_day_prices(self, day: date) -> pd.DataFrame:
        df = self.prices
        frame = df[df["date"] == day]
        return frame.copy()

    def get_history_until(self, day: date) -> pd.DataFrame:
        df = self.prices
        frame = df[df["date"] <= day]
        return frame.copy()

    def get_day_context(self, day: date) -> Dict[str, Any]:
        day_prices = self.get_day_prices(day)
        history = self.get_history_until(day)
        indicators = self.indicators
        if not indicators.empty and "date" in indicators.columns:
            indicators = indicators[indicators["date"] <= pd.Timestamp(day)]
        watchlist = self.watchlist
        if not watchlist.empty and "as_of" in watchlist.columns:
            watchlist = watchlist[watchlist["as_of"] <= str(day)]
        context = {
            "date": day,
            "prices": day_prices,
            "history": history,
            "watchlist": watchlist,
            "indicators": indicators,
            "corporate_actions": self.corporate_actions,
            "exchange": self.config.exchange,
            "engine_config": dict(self._engine_config),
            "trading_days": list(self.calendar),
        }
        if self._order_builder is not None:
            context["order_builder"] = self._order_builder
        return context
