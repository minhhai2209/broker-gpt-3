"""Data engine that materialises broker contract outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP, ROUND_UP, getcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)

getcontext().prec = 28


class EngineError(RuntimeError):
    """Raised when the engine cannot complete a required step."""


@dataclass
class AttachmentBundleResult:
    path: Path
    files: List[Path]
    missing: List[Path]


@dataclass
class OutputSpec:
    name: str
    path: Path
    dataframe: pd.DataFrame
    expected_columns: Sequence[str]
    require_rows: bool = False


@dataclass
class EngineConfig:
    """Resolved paths for the contract engine."""

    repo_root: Path
    out_dir: Path
    data_dir: Path
    config_dir: Path
    bundle_dir: Path
    profile: str = "alpha"

    def __post_init__(self) -> None:
        profile = str(self.profile).strip()
        if not profile:
            raise EngineError("Profile name must not be empty")
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
        if any(ch not in allowed for ch in profile):
            raise EngineError(f"Profile name contains invalid characters: {profile}")
        self.profile = profile

    @classmethod
    def from_yaml(cls, path: Path, profile: str = "alpha") -> "EngineConfig":
        if not path.exists():
            raise EngineError(f"Config file not found: {path}")
        config_dir = path.parent.resolve()
        repo_root = _find_repo_root(config_dir)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise EngineError("Engine config must be a mapping")
        paths_cfg = raw.get("paths", {}) or {}
        if not isinstance(paths_cfg, dict):
            raise EngineError("paths section must be a mapping")

        out_dir = _resolve_path(paths_cfg.get("out", "out"), config_dir, repo_root)
        data_dir = _resolve_path(paths_cfg.get("data", "data"), config_dir, repo_root)
        cfg_dir = _resolve_path(paths_cfg.get("config", "config"), config_dir, repo_root)
        bundle_dir = _resolve_path(paths_cfg.get("bundle", ".artifacts/engine"), config_dir, repo_root)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            repo_root=repo_root,
            out_dir=out_dir,
            data_dir=data_dir,
            config_dir=cfg_dir,
            bundle_dir=bundle_dir,
            profile=profile,
        )

    # Input paths -----------------------------------------------------
    @property
    def technical_snapshot_path(self) -> Path:
        return self.out_dir / "market" / "technical_snapshot.csv"

    @property
    def presets_dir(self) -> Path:
        return self.out_dir / "presets"

    @property
    def portfolio_holdings_path(self) -> Path:
        return self.data_dir / "portfolios" / f"{self.profile}.csv"

    @property
    def portfolio_positions_path(self) -> Path:
        return self.out_dir / "portfolios" / f"{self.profile}_positions.csv"

    @property
    def portfolio_sector_path(self) -> Path:
        return self.out_dir / "portfolios" / f"{self.profile}_sector.csv"

    @property
    def fills_path(self) -> Path:
        return self.data_dir / "order_history" / f"{self.profile}_fills.csv"

    @property
    def params_path(self) -> Path:
        return self.config_dir / "params.yaml"

    @property
    def blocklist_path(self) -> Path:
        return self.config_dir / "blocklist.csv"

    @property
    def universe_path(self) -> Path:
        return self.data_dir / "universe" / "vn100.csv"

    @property
    def news_score_path(self) -> Path:
        return self.out_dir / "news" / "news_score.csv"

    # Output paths ----------------------------------------------------
    @property
    def trading_bands_path(self) -> Path:
        return self.out_dir / "market" / "trading_bands.csv"

    @property
    def levels_path(self) -> Path:
        return self.out_dir / "signals" / "levels.csv"

    @property
    def sizing_path(self) -> Path:
        return self.out_dir / "signals" / "sizing.csv"

    @property
    def signals_path(self) -> Path:
        return self.out_dir / "signals" / "signals.csv"

    @property
    def orders_dir(self) -> Path:
        return self.out_dir / "orders"

    @property
    def orders_latest_path(self) -> Path:
        return self.orders_dir / f"{self.profile}_LO_latest.csv"

    @property
    def run_manifest_path(self) -> Path:
        return self.out_dir / "run" / f"{self.profile}_manifest.json"

    @property
    def attachment_bundle_path(self) -> Path:
        return self.bundle_dir / f"{self.profile}_attachments_latest.zip"

    def ensure_output_dirs(self) -> None:
        for path in [
            self.trading_bands_path.parent,
            self.levels_path.parent,
            self.sizing_path.parent,
            self.signals_path.parent,
            self.orders_dir,
            self.run_manifest_path.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)


class DataEngine:
    """Contract-compliant data engine."""

    def __init__(self, config: EngineConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    def run(self) -> Dict[str, object]:
        cfg = self._config
        cfg.ensure_output_dirs()

        universe = self._load_universe(cfg.universe_path)
        snapshot = self._load_snapshot(cfg.technical_snapshot_path)
        presets = self._load_presets(cfg.presets_dir)
        blocklisted = self._load_blocklist(cfg.blocklist_path)
        params = self._load_params(cfg.params_path)
        holdings = self._load_holdings(cfg.portfolio_holdings_path)
        sectors = self._load_optional_csv(cfg.portfolio_sector_path)
        fills = self._load_optional_csv(cfg.fills_path)
        news = self._load_optional_csv(cfg.news_score_path)

        candidate_indices = self._filter_candidate_tickers(snapshot, presets, universe)
        if not candidate_indices:
            LOGGER.warning("No eligible tickers after universe/preset filtering")

        trading_bands, band_flags = self._build_trading_bands(snapshot.iloc[candidate_indices])
        trading_bands.to_csv(cfg.trading_bands_path, index=False)

        levels, rule_flags = self._build_levels(snapshot, trading_bands, presets)
        levels.to_csv(cfg.levels_path, index=False)

        sizing = self._build_sizing(snapshot, holdings, fills, params)
        sizing.to_csv(cfg.sizing_path, index=False)

        base_flags = self._initial_risk_flags(snapshot, blocklisted, band_flags, rule_flags)
        signals = self._build_signals(snapshot, trading_bands, sectors, news, base_flags)

        orders, additional_flags = self._build_orders(
            snapshot,
            trading_bands,
            levels,
            sizing,
            signals,
            params,
            base_flags,
        )
        signals = self._update_signals_with_flags(signals, additional_flags)
        signals.to_csv(cfg.signals_path, index=False)

        orders.to_csv(cfg.orders_latest_path, index=False)
        if not orders.empty:
            dated_path = cfg.orders_dir / f"{cfg.profile}_LO_{datetime.now().strftime('%Y%m%d')}.csv"
            orders.to_csv(dated_path, index=False)

        self._write_manifest(cfg, snapshot, presets, params)

        self._validate_outputs(
            [
                OutputSpec(
                    name="trading_bands",
                    path=cfg.trading_bands_path,
                    dataframe=trading_bands,
                    expected_columns=["Ticker", "Ref", "Ceil", "Floor", "TickSize"],
                    require_rows=True,
                ),
                OutputSpec(
                    name="levels",
                    path=cfg.levels_path,
                    dataframe=levels,
                    expected_columns=[
                        "Ticker",
                        "Preset",
                        "NearTouchBuy",
                        "NearTouchSell",
                        "Opp1Buy",
                        "Opp1Sell",
                        "Opp2Buy",
                        "Opp2Sell",
                        "Limit_kVND",
                    ],
                    require_rows=True,
                ),
                OutputSpec(
                    name="sizing",
                    path=cfg.sizing_path,
                    dataframe=sizing,
                    expected_columns=[
                        "Ticker",
                        "TargetQty",
                        "CurrentQty",
                        "DeltaQty",
                        "MaxOrderQty",
                        "SliceCount",
                        "SliceQty",
                        "LiquidityScore",
                        "VolatilityScore",
                        "TodayFilledQty",
                        "TodayWAP",
                    ],
                    require_rows=True,
                ),
                OutputSpec(
                    name="signals",
                    path=cfg.signals_path,
                    dataframe=signals,
                    expected_columns=[
                        "Ticker",
                        "PresetFitMomentum",
                        "PresetFitMeanRev",
                        "PresetFitBalanced",
                        "BandDistance",
                        "NewsScore",
                        "EventFlags",
                        "SectorBias",
                        "RiskGuards",
                    ],
                    require_rows=True,
                ),
                OutputSpec(
                    name="orders",
                    path=cfg.orders_latest_path,
                    dataframe=orders,
                    expected_columns=["Ticker", "Side", "Quantity", "LimitPrice"],
                    require_rows=False,
                ),
            ]
        )
        self._validate_manifest(cfg.run_manifest_path)

        bundle = self._bundle_outputs(
            cfg,
            [
                cfg.trading_bands_path,
                cfg.levels_path,
                cfg.sizing_path,
                cfg.signals_path,
                cfg.orders_latest_path,
                cfg.run_manifest_path,
            ],
        )

        return {
            "tickers": int(len(snapshot)),
            "orders": int(len(orders)),
            "profile": cfg.profile,
            "attachment_bundle": str(bundle.path),
            "attachment_files": [str(p) for p in bundle.files],
            "missing_attachments": [str(p) for p in bundle.missing],
        }

    # ------------------------------------------------------------------
    # Loading helpers
    def _load_universe(self, path: Path) -> List[str]:
        if not path.exists():
            raise EngineError("Universe file missing (vn100.csv)")
        df = pd.read_csv(path)
        if "Ticker" not in df.columns:
            raise EngineError("Universe file missing 'Ticker' column")
        return sorted({str(t).upper() for t in df["Ticker"].dropna()})

    def _load_snapshot(self, path: Path) -> pd.DataFrame:
        required = {
            "Ticker",
            "Last",
            "Ref",
            "SMA20",
            "ATR14",
            "RSI14",
            "MACD",
            "MACDSignal",
            "Z20",
            "Ret5d",
            "Ret20d",
            "ADV20",
            "High52w",
            "Low52w",
        }
        df = self._load_csv_with_required(path, required)
        df["Ticker"] = df["Ticker"].astype(str).str.upper()
        return df

    def _load_presets(self, presets_dir: Path) -> Dict[str, pd.DataFrame]:
        if not presets_dir.exists():
            raise EngineError(f"Presets directory missing: {presets_dir}")
        preset_files = sorted(presets_dir.glob("*.csv"))
        if not preset_files:
            raise EngineError("No preset CSV files found")
        presets: Dict[str, pd.DataFrame] = {}
        for file in preset_files:
            df = pd.read_csv(file)
            if "Ticker" not in df.columns or "RuleType" not in df.columns:
                LOGGER.warning("Skipping preset %s missing required columns", file)
                continue
            df["Ticker"] = df["Ticker"].astype(str).str.upper()
            df["PresetName"] = file.stem
            presets[file.stem] = df
        if not presets:
            raise EngineError("No valid preset definitions found")
        return presets

    def _load_blocklist(self, path: Path) -> Dict[str, str]:
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        if "Ticker" not in df.columns:
            raise EngineError("Blocklist missing 'Ticker' column")
        result: Dict[str, str] = {}
        for row in df.itertuples(index=False):
            ticker = str(getattr(row, "Ticker", "")).upper()
            if not ticker:
                continue
            reason = str(getattr(row, "Reason", "")).strip()
            result[ticker] = reason
        return result

    def _load_params(self, path: Path) -> Dict[str, object]:
        if not path.exists():
            raise EngineError("params.yaml missing")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise EngineError("params.yaml must be a mapping")
        return data

    def _load_holdings(self, path: Path) -> pd.DataFrame:
        df = self._load_csv_with_required(path, {"Ticker", "Quantity", "AvgPrice"})
        df["Ticker"] = df["Ticker"].astype(str).str.upper()
        aggregated = df.groupby("Ticker", as_index=False).agg({"Quantity": "sum", "AvgPrice": "mean"})
        return aggregated

    def _load_optional_csv(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def _load_csv_with_required(self, path: Path, required: Iterable[str]) -> pd.DataFrame:
        if not path.exists():
            raise EngineError(f"Required file missing: {path}")
        df = pd.read_csv(path)
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise EngineError(f"{path} missing columns: {', '.join(missing)}")
        return df

    def _filter_candidate_tickers(
        self, snapshot: pd.DataFrame, presets: Dict[str, pd.DataFrame], universe: Sequence[str]
    ) -> List[int]:
        preset_tickers = set()
        for df in presets.values():
            preset_tickers.update(df["Ticker"].astype(str).str.upper())
        allowed = set(universe)
        indices = [
            idx
            for idx, row in snapshot.iterrows()
            if str(row.Ticker).upper() in allowed and str(row.Ticker).upper() in preset_tickers
        ]
        return indices

    # ------------------------------------------------------------------
    # Computation helpers
    def _build_trading_bands(self, snapshot: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, set[str]]]:
        rows: List[Dict[str, object]] = []
        flags: Dict[str, set[str]] = {}
        for row in snapshot.itertuples(index=False):
            ticker = str(row.Ticker).upper()
            ref = _decimal(getattr(row, "Ref", float("nan")))
            if ref is None:
                continue
            tick = _tick_size(ref)
            ceil = _floor_to_tick(ref * Decimal("1.07"), tick)
            floor = _ceil_to_tick(ref * Decimal("0.93"), tick)
            if floor > ceil:
                floor, ceil = ceil, floor
                flags.setdefault(ticker, set()).add("BAND_ERROR")
            rows.append(
                {
                    "Ticker": ticker,
                    "Ref": float(ref),
                    "Ceil": float(ceil),
                    "Floor": float(floor),
                    "TickSize": float(tick),
                }
            )
        df = pd.DataFrame(rows, columns=["Ticker", "Ref", "Ceil", "Floor", "TickSize"])
        return df, flags

    def _build_levels(
        self,
        snapshot: pd.DataFrame,
        trading_bands: pd.DataFrame,
        presets: Dict[str, pd.DataFrame],
    ) -> Tuple[pd.DataFrame, Dict[str, set[str]]]:
        snap_lookup = {row.Ticker: row._asdict() for row in snapshot.itertuples(index=False)}
        band_lookup = {row.Ticker: row._asdict() for row in trading_bands.itertuples(index=False)}
        rows: List[Dict[str, object]] = []
        flags: Dict[str, set[str]] = {}
        for preset_name, df in presets.items():
            for row in df.itertuples(index=False):
                ticker = str(getattr(row, "Ticker")).upper()
                if ticker not in snap_lookup or ticker not in band_lookup:
                    continue
                snap = snap_lookup[ticker]
                bands = band_lookup[ticker]
                rule = str(getattr(row, "RuleType", "")).strip()
                side = str(getattr(row, "Side", "BOTH")).strip().upper() or "BOTH"
                param1 = getattr(row, "Param1", float("nan"))
                param2 = getattr(row, "Param2", float("nan"))
                result = self._evaluate_rule(rule, snap, bands, param1, param2)
                if result is None:
                    flags.setdefault(ticker, set()).add("UNKNOWN_RULE")
                    continue
                buy_level, sell_level, opp_buy, opp_sell = result
                limit_kvnd: Optional[int | str] = ""
                if side in {"BUY", "BOTH"} and buy_level is not None:
                    limit_kvnd = int(_kvnd(buy_level)) if side == "BUY" else ""
                if side == "SELL" and sell_level is not None:
                    limit_kvnd = int(_kvnd(sell_level))
                rows.append(
                    {
                        "Ticker": ticker,
                        "Preset": preset_name,
                        "NearTouchBuy": _float_or_blank(buy_level if side != "SELL" else None),
                        "NearTouchSell": _float_or_blank(sell_level if side != "BUY" else None),
                        "Opp1Buy": _float_or_blank(opp_buy if side != "SELL" else None),
                        "Opp1Sell": _float_or_blank(opp_sell if side != "BUY" else None),
                        "Opp2Buy": "",
                        "Opp2Sell": "",
                        "Limit_kVND": limit_kvnd,
                    }
                )
        df = pd.DataFrame(
            rows,
            columns=[
                "Ticker",
                "Preset",
                "NearTouchBuy",
                "NearTouchSell",
                "Opp1Buy",
                "Opp1Sell",
                "Opp2Buy",
                "Opp2Sell",
                "Limit_kVND",
            ],
        )
        return df, flags

    def _evaluate_rule(
        self,
        rule: str,
        snap: Dict[str, object],
        bands: Dict[str, object],
        param1: object,
        param2: object,
    ) -> Optional[Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal], Optional[Decimal]]]:
        last = _decimal(snap.get("Last"))
        sma20 = _decimal(snap.get("SMA20"))
        atr14 = _decimal(snap.get("ATR14"))
        z20 = _decimal(snap.get("Z20"))
        tick = Decimal(str(bands.get("TickSize", "0.01")))
        floor = _decimal(bands.get("Floor"))
        ceil = _decimal(bands.get("Ceil"))
        if last is None or tick is None or floor is None or ceil is None:
            return None

        def clamp(value: Decimal | None) -> Optional[Decimal]:
            if value is None:
                return None
            rounded = _round_to_tick(value, tick)
            return _clamp_to_band(rounded, floor, ceil)

        if rule == "offset_ticks_from_last":
            buy_offset = Decimal(str(param1)) if not _is_nan(param1) else Decimal("0")
            sell_offset = Decimal(str(param2)) if not _is_nan(param2) else Decimal("0")
            buy = clamp(last + buy_offset * tick)
            sell = clamp(last + sell_offset * tick)
            opp_buy = clamp(last + (buy_offset - Decimal("1")) * tick)
            opp_sell = clamp(last + (sell_offset + Decimal("1")) * tick)
            return buy, sell, opp_buy, opp_sell
        if rule == "SMA20±kATR":
            if atr14 is None:
                return None
            k_buy = Decimal(str(param1)) if not _is_nan(param1) else Decimal("0")
            k_sell = Decimal(str(param2)) if not _is_nan(param2) else Decimal("0")
            buy = clamp(sma20 - k_buy * atr14) if sma20 is not None else None
            sell = clamp(sma20 + k_sell * atr14) if sma20 is not None else None
            return buy, sell, None, None
        if rule == "zscore_thresholds":
            if z20 is None or atr14 is None or sma20 is None:
                return None
            z_buy_max = Decimal(str(param1)) if not _is_nan(param1) else Decimal("0")
            z_sell_min = Decimal(str(param2)) if not _is_nan(param2) else Decimal("0")
            buy = clamp(sma20 - abs(z_buy_max) * atr14) if z20 <= z_buy_max else None
            sell = clamp(sma20 + abs(z_sell_min) * atr14) if z20 >= z_sell_min else None
            return buy, sell, None, None
        if rule == "last_vs_floorceil":
            buy = clamp(max(last, floor))
            sell = clamp(min(last, ceil))
            return buy, sell, None, None
        if rule == "weighted_momentum_meanrev":
            if atr14 is None or sma20 is None:
                return None
            w_mom = Decimal(str(param1)) if not _is_nan(param1) else Decimal("0.5")
            w_mr = Decimal(str(param2)) if not _is_nan(param2) else (Decimal("1") - w_mom)
            mom_buy = clamp(last - tick)
            mom_sell = clamp(last + tick)
            mr_buy = clamp(sma20 - Decimal("0.25") * atr14)
            mr_sell = clamp(sma20 + Decimal("0.25") * atr14)
            buy = clamp(_weighted_sum([(w_mom, mom_buy), (w_mr, mr_buy)]))
            sell = clamp(_weighted_sum([(w_mom, mom_sell), (w_mr, mr_sell)]))
            return buy, sell, None, None
        if rule == "risk_off_trim":
            sell_offset = Decimal(str(param1)) if not _is_nan(param1) else Decimal("0")
            sell = clamp(last + sell_offset * tick)
            return None, sell, None, None
        return None

    def _build_sizing(
        self,
        snapshot: pd.DataFrame,
        holdings: pd.DataFrame,
        fills: pd.DataFrame,
        params: Dict[str, object],
    ) -> pd.DataFrame:
        fills_summary = self._summarise_fills(fills)
        holdings_lookup = holdings.set_index("Ticker")["Quantity"].to_dict()
        adv_lookup = snapshot.set_index("Ticker")["ADV20"].to_dict()
        last_lookup = snapshot.set_index("Ticker")["Last"].to_dict()
        atr_lookup = snapshot.set_index("Ticker")["ATR14"].to_dict()

        max_order_pct_adv = float(params.get("max_order_pct_adv", 0.05))
        slice_adv_ratio = float(params.get("slice_adv_ratio", 0.02))
        buy_budget = float(params.get("buy_budget_vnd", 0.0))
        sell_budget = float(params.get("sell_budget_vnd", 0.0))
        max_qty_per_order = int(params.get("max_qty_per_order", 500000))

        rows: List[Dict[str, object]] = []
        for row in snapshot.itertuples(index=False):
            ticker = str(row.Ticker)
            current_qty = float(holdings_lookup.get(ticker, 0.0))
            target_qty = current_qty
            delta_qty = target_qty - current_qty
            adv20 = float(adv_lookup.get(ticker, float("nan")))
            last_price = float(last_lookup.get(ticker, float("nan")))
            atr14 = float(atr_lookup.get(ticker, float("nan")))

            liquidity_score = adv20 * last_price if not math.isnan(adv20) and not math.isnan(last_price) else float("nan")
            volatility_score = (atr14 / last_price) if last_price not in (0, float("nan")) and not math.isnan(atr14) and not math.isnan(last_price) and last_price != 0 else 0.0

            budget_side = buy_budget if delta_qty > 0 else sell_budget
            if last_price and not math.isnan(last_price) and last_price > 0:
                budget_qty = budget_side / last_price if budget_side > 0 else float("inf")
            else:
                budget_qty = 0.0
            adv_cap = max_order_pct_adv * adv20 if not math.isnan(adv20) else float("inf")
            max_order_qty = min(max_qty_per_order, adv_cap, budget_qty)
            if math.isnan(max_order_qty):
                max_order_qty = 0.0
            max_order_qty = max(0.0, float(max_order_qty))

            slice_count = 0
            slice_qty = 0
            abs_delta = abs(delta_qty)
            if abs_delta > 0 and not math.isnan(adv20) and adv20 > 0:
                slice_count = max(1, int(math.ceil(abs_delta / (slice_adv_ratio * adv20))))
                slice_qty = _round_to_lot(abs_delta / slice_count)
            fills_info = fills_summary.get(ticker, {"qty": 0.0, "wap": float("nan")})
            rows.append(
                {
                    "Ticker": ticker,
                    "TargetQty": float(target_qty),
                    "CurrentQty": float(current_qty),
                    "DeltaQty": float(delta_qty),
                    "MaxOrderQty": float(max_order_qty),
                    "SliceCount": int(slice_count),
                    "SliceQty": int(slice_qty),
                    "LiquidityScore": float(liquidity_score) if not math.isnan(liquidity_score) else float("nan"),
                    "VolatilityScore": float(volatility_score),
                    "TodayFilledQty": float(fills_info["qty"]),
                    "TodayWAP": float(fills_info["wap"]),
                }
            )
        df = pd.DataFrame(
            rows,
            columns=[
                "Ticker",
                "TargetQty",
                "CurrentQty",
                "DeltaQty",
                "MaxOrderQty",
                "SliceCount",
                "SliceQty",
                "LiquidityScore",
                "VolatilityScore",
                "TodayFilledQty",
                "TodayWAP",
            ],
        )
        return df

    def _summarise_fills(self, fills: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        if fills.empty:
            return {}
        if "Ticker" not in fills.columns or "Qty" not in fills.columns or "Price" not in fills.columns:
            LOGGER.warning("Fills missing columns, ignoring")
            return {}
        summaries: Dict[str, Dict[str, float]] = {}
        grouped = fills.groupby("Ticker")
        for ticker, grp in grouped:
            qty = float(grp["Qty"].sum())
            wap = float((grp["Qty"] * grp["Price"]).sum() / qty) if qty else float("nan")
            summaries[str(ticker).upper()] = {"qty": qty, "wap": wap}
        return summaries

    def _initial_risk_flags(
        self,
        snapshot: pd.DataFrame,
        blocklisted: Dict[str, str],
        band_flags: Dict[str, set[str]],
        rule_flags: Dict[str, set[str]],
    ) -> Dict[str, set[str]]:
        result: Dict[str, set[str]] = {}
        for row in snapshot.itertuples(index=False):
            ticker = str(row.Ticker).upper()
            guards: set[str] = set()
            if ticker in blocklisted:
                guards.add("BLOCKLIST")
            atr = getattr(row, "ATR14", float("nan"))
            last = getattr(row, "Last", float("nan"))
            adv = getattr(row, "ADV20", float("nan"))
            if math.isnan(atr) or atr == 0:
                guards.add("ZERO_ATR")
            if math.isnan(last) or last == 0:
                guards.add("ZERO_LAST")
            if math.isnan(adv) or adv == 0:
                guards.add("LOW_LIQ")
            if ticker in band_flags:
                guards.update(band_flags[ticker])
            if ticker in rule_flags:
                guards.update(rule_flags[ticker])
            result[ticker] = guards
        return result

    def _build_signals(
        self,
        snapshot: pd.DataFrame,
        trading_bands: pd.DataFrame,
        sectors: pd.DataFrame,
        news: pd.DataFrame,
        risk_flags: Dict[str, set[str]],
    ) -> pd.DataFrame:
        band_lookup = trading_bands.set_index("Ticker").to_dict(orient="index")
        sector_lookup = {}
        if not sectors.empty and "Sector" in sectors.columns and "PNLPct" in sectors.columns:
            for row in sectors.itertuples(index=False):
                sector_lookup[str(getattr(row, "Sector", ""))] = float(getattr(row, "PNLPct", 0.0))
        news_lookup = {}
        if not news.empty and "Ticker" in news.columns:
            latest = news.sort_values("AsOf" if "AsOf" in news.columns else "Ticker")
            for row in latest.itertuples(index=False):
                ticker = str(getattr(row, "Ticker", "")).upper()
                score = float(getattr(row, "Score", 0.0)) if "Score" in news.columns else 0.0
                flags = str(getattr(row, "Flags", "")) if "Flags" in news.columns else ""
                news_lookup[ticker] = (score, flags)

        rows: List[Dict[str, object]] = []
        for row in snapshot.itertuples(index=False):
            ticker = str(row.Ticker).upper()
            bands = band_lookup.get(ticker, {})
            ceil = bands.get("Ceil")
            floor = bands.get("Floor")
            last = getattr(row, "Last", float("nan"))
            atr = getattr(row, "ATR14", float("nan"))
            band_distance = 0.0
            if not math.isnan(atr) and atr > 0 and ceil is not None and floor is not None and not math.isnan(last):
                band_distance = min((ceil - last) / atr, (last - floor) / atr)
                band_distance = max(0.0, band_distance)
            momentum_score = self._momentum_fit(row)
            meanrev_score = self._meanrev_fit(row, floor)
            balanced_score = (momentum_score + meanrev_score) / 2.0
            news_score, news_flags = news_lookup.get(ticker, (0.0, ""))
            sector_bias = 0
            sector_name = getattr(row, "Sector", "") if hasattr(row, "Sector") else ""
            if sector_name in sector_lookup:
                pnl_pct = sector_lookup[sector_name]
                if pnl_pct >= 5:
                    sector_bias = 2
                elif pnl_pct >= 2:
                    sector_bias = 1
                elif pnl_pct <= -5:
                    sector_bias = -2
                elif pnl_pct <= -2:
                    sector_bias = -1
            guards = sorted(risk_flags.get(ticker, set()))
            rows.append(
                {
                    "Ticker": ticker,
                    "PresetFitMomentum": round(momentum_score, 4),
                    "PresetFitMeanRev": round(meanrev_score, 4),
                    "PresetFitBalanced": round(balanced_score, 4),
                    "BandDistance": round(band_distance, 4),
                    "NewsScore": round(news_score, 4),
                    "EventFlags": news_flags,
                    "SectorBias": sector_bias,
                    "RiskGuards": "|".join(guards) if guards else "",
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "Ticker",
                "PresetFitMomentum",
                "PresetFitMeanRev",
                "PresetFitBalanced",
                "BandDistance",
                "NewsScore",
                "EventFlags",
                "SectorBias",
                "RiskGuards",
            ],
        )

    def _momentum_fit(self, row: pd.Series) -> float:
        score = 0.0
        count = 0
        rsi = getattr(row, "RSI14", float("nan"))
        ret20 = getattr(row, "Ret20d", float("nan"))
        macd = getattr(row, "MACD", float("nan"))
        macd_signal = getattr(row, "MACDSignal", float("nan"))
        if not math.isnan(rsi):
            count += 1
            if rsi > 55:
                score += 1
        if not math.isnan(ret20):
            count += 1
            if ret20 > 0:
                score += 1
        if not math.isnan(macd) and not math.isnan(macd_signal):
            count += 1
            if macd > macd_signal:
                score += 1
        return score / count if count else 0.0

    def _meanrev_fit(self, row: pd.Series, floor: Optional[float]) -> float:
        z20 = getattr(row, "Z20", float("nan"))
        last = getattr(row, "Last", float("nan"))
        floor_val = floor if floor is not None else float("nan")
        scores: List[float] = []
        if not math.isnan(z20):
            scores.append(1.0 if z20 <= -0.5 else 0.0)
        if not math.isnan(last) and not math.isnan(floor_val):
            diff = last - floor_val
            scores.append(1.0 if diff <= 1 else 0.0)
        return sum(scores) / len(scores) if scores else 0.0

    def _build_orders(
        self,
        snapshot: pd.DataFrame,
        trading_bands: pd.DataFrame,
        levels: pd.DataFrame,
        sizing: pd.DataFrame,
        signals: pd.DataFrame,
        params: Dict[str, object],
        risk_flags: Dict[str, set[str]],
    ) -> Tuple[pd.DataFrame, Dict[str, set[str]]]:
        level_lookup = levels.set_index(["Ticker", "Preset"]).to_dict(orient="index")
        bands_lookup = trading_bands.set_index("Ticker").to_dict(orient="index")
        sizing_lookup = sizing.set_index("Ticker").to_dict(orient="index")
        signal_lookup = signals.set_index("Ticker").to_dict(orient="index")

        aggressiveness = str(params.get("aggressiveness", "low")).lower()
        buy_budget = float(params.get("buy_budget_vnd", 0.0))
        buy_spent = 0.0

        rows: List[Dict[str, object]] = []
        additional_flags: Dict[str, set[str]] = {ticker: set(flags) for ticker, flags in risk_flags.items()}

        for row in sizing.itertuples(index=False):
            ticker = str(row.Ticker)
            delta = float(getattr(row, "DeltaQty", 0.0))
            if delta == 0:
                continue
            side = "BUY" if delta > 0 else "SELL"
            guards = additional_flags.setdefault(ticker, set())
            if "BLOCKLIST" in guards:
                continue
            if "LOW_LIQ" in guards and abs(delta) > 100:
                continue
            band = bands_lookup.get(ticker)
            if not band:
                continue
            floor = Decimal(str(band.get("Floor")))
            ceil = Decimal(str(band.get("Ceil")))
            tick = Decimal(str(band.get("TickSize", "0.01")))

            signal_view = dict(signal_lookup.get(ticker, {}))
            if ticker in sizing_lookup:
                signal_view.setdefault("VolatilityScore", sizing_lookup[ticker].get("VolatilityScore", 0.0))
            preset_choice = self._select_preset_for_side(side, signal_view)
            level = level_lookup.get((ticker, preset_choice))
            if not level:
                continue
            if side == "BUY":
                base_price = level.get("NearTouchBuy")
            else:
                base_price = level.get("NearTouchSell")
            if _is_nan(base_price):
                continue
            price = Decimal(str(base_price))

            price = self._apply_aggressiveness(
                price,
                tick,
                floor,
                ceil,
                side,
                aggressiveness,
                signal_view,
            )

            clamped_price = _clamp_to_band(price, floor, ceil)
            if clamped_price != price:
                guards.add("CLAMPED")
                price = clamped_price

            if price == floor or price == ceil:
                guards.add("NEAR_LIMIT")

            slice_qty = int(getattr(row, "SliceQty", 0))
            if slice_qty <= 0:
                slice_qty = _round_to_lot(abs(delta)) or 0
            total_qty = min(abs(delta), float(getattr(row, "MaxOrderQty", abs(delta))))
            total_qty = min(total_qty, 500000)
            total_qty = _round_to_lot(total_qty)
            if total_qty == 0:
                continue
            order_qty = min(total_qty, slice_qty if slice_qty else total_qty)
            order_qty = _round_to_lot(order_qty)
            if order_qty == 0:
                continue

            vnd_value = float(price) * order_qty
            if side == "BUY":
                if buy_spent + vnd_value > buy_budget:
                    remaining = buy_budget - buy_spent
                    if remaining <= 0:
                        continue
                    order_qty = _round_to_lot(remaining / float(price))
                    if order_qty == 0:
                        continue
                    vnd_value = float(price) * order_qty
                buy_spent += vnd_value
            else:
                sell_volume += vnd_value

            rows.append(
                {
                    "Ticker": ticker,
                    "Side": side,
                    "Quantity": int(order_qty),
                    "LimitPrice": int(_kvnd(price)),
                }
            )

        df = pd.DataFrame(rows, columns=["Ticker", "Side", "Quantity", "LimitPrice"])
        return df, additional_flags

    def _select_preset_for_side(self, side: str, signal: Dict[str, object]) -> str:
        candidates = {
            "BUY": [
                (signal.get("PresetFitMomentum", 0.0), "momentum"),
                (signal.get("PresetFitMeanRev", 0.0), "mean_reversion"),
                (signal.get("PresetFitBalanced", 0.0), "balanced"),
            ],
            "SELL": [
                (signal.get("PresetFitMomentum", 0.0), "momentum"),
                (signal.get("PresetFitBalanced", 0.0), "balanced"),
                (signal.get("PresetFitMeanRev", 0.0), "risk_off"),
            ],
        }
        ranked = sorted(candidates.get(side, []), key=lambda item: float(item[0]), reverse=True)
        for _, preset in ranked:
            return preset
        return "balanced"

    def _apply_aggressiveness(
        self,
        price: Decimal,
        tick: Decimal,
        floor: Decimal,
        ceil: Decimal,
        side: str,
        aggressiveness: str,
        signal: Dict[str, object],
    ) -> Decimal:
        adjustment = Decimal("0")
        news_score = Decimal(str(signal.get("NewsScore", 0.0)))
        band_distance = Decimal(str(signal.get("BandDistance", 0.0)))
        volatility = Decimal(str(signal.get("VolatilityScore", 0.0))) if "VolatilityScore" in signal else Decimal("0")

        if aggressiveness == "med":
            if volatility > Decimal("0.05") or news_score < Decimal("0"):
                adjustment = -tick if side == "BUY" else tick
        elif aggressiveness == "high":
            if news_score > Decimal("0") and band_distance > Decimal("1"):
                adjustment = tick if side == "BUY" else -tick
        return _clamp_to_band(price + adjustment, floor, ceil)

    def _update_signals_with_flags(
        self, signals: pd.DataFrame, new_flags: Dict[str, set[str]]
    ) -> pd.DataFrame:
        if signals.empty:
            return signals
        signals = signals.copy()
        guard_lookup = {ticker: "|".join(sorted(flags)) if flags else "" for ticker, flags in new_flags.items()}
        signals["RiskGuards"] = signals["Ticker"].map(lambda t: guard_lookup.get(t, ""))
        return signals

    # ------------------------------------------------------------------
    # Manifest & bundling
    def _validate_outputs(self, specs: Sequence[OutputSpec]) -> None:
        for spec in specs:
            if not spec.path.exists():
                raise EngineError(f"Missing output: {spec.name} at {spec.path}")
            actual_columns = list(spec.dataframe.columns)
            expected_columns = list(spec.expected_columns)
            if actual_columns != expected_columns:
                raise EngineError(
                    f"Output {spec.name} columns mismatch. Expected {expected_columns}, got {actual_columns}"
                )
            if spec.require_rows and spec.dataframe.empty:
                raise EngineError(f"Output {spec.name} must contain at least one row")
            if spec.path.stat().st_size == 0:
                raise EngineError(f"Output {spec.name} produced an empty file at {spec.path}")

    def _write_manifest(
        self,
        cfg: EngineConfig,
        snapshot: pd.DataFrame,
        presets: Dict[str, pd.DataFrame],
        params: Dict[str, object],
    ) -> Dict[str, object]:
        source_files = [
            cfg.technical_snapshot_path,
            cfg.portfolio_holdings_path,
            cfg.portfolio_positions_path,
            cfg.portfolio_sector_path,
            cfg.fills_path,
            cfg.universe_path,
            cfg.blocklist_path,
            cfg.params_path,
        ]
        for preset in cfg.presets_dir.glob("*.csv"):
            source_files.append(preset)
        if cfg.news_score_path.exists():
            source_files.append(cfg.news_score_path)
        params_hash = hashlib.sha256(cfg.params_path.read_bytes()).hexdigest() if cfg.params_path.exists() else ""
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_files": sorted({_as_repo_relative(p, default=str(p), repo_root=cfg.repo_root) for p in source_files if p.exists()}),
            "params_hash": params_hash,
        }
        cfg.run_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def _validate_manifest(self, path: Path) -> None:
        if not path.exists():
            raise EngineError(f"Manifest not found at {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        required_keys = {"generated_at", "source_files", "params_hash"}
        missing = required_keys - set(data.keys())
        if missing:
            raise EngineError(f"Manifest missing keys: {sorted(missing)}")
        if not isinstance(data.get("source_files"), list):
            raise EngineError("Manifest source_files must be a list")

    def _bundle_outputs(self, cfg: EngineConfig, files: Sequence[Path]) -> AttachmentBundleResult:
        bundle_path = cfg.attachment_bundle_path
        found: List[Path] = []
        missing: List[Path] = []
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                if not path.exists():
                    missing.append(path)
                    continue
                arcname = _as_repo_relative(path, default=path.name, repo_root=cfg.repo_root)
                archive.write(path, arcname)
                found.append(Path(arcname))
        return AttachmentBundleResult(path=bundle_path, files=found, missing=missing)


# ----------------------------------------------------------------------
# Utility helpers
def _decimal(value: object) -> Optional[Decimal]:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return Decimal(str(value))
    except Exception:  # pragma: no cover - defensive
        return None


def _tick_size(price: Decimal) -> Decimal:
    if price < Decimal("10"):
        return Decimal("0.01")
    if price < Decimal("50"):
        return Decimal("0.05")
    return Decimal("0.10")


def _floor_to_tick(value: Decimal, tick: Decimal) -> Decimal:
    return (value / tick).to_integral_value(rounding=ROUND_DOWN) * tick


def _ceil_to_tick(value: Decimal, tick: Decimal) -> Decimal:
    return (value / tick).to_integral_value(rounding=ROUND_UP) * tick


def _round_to_tick(value: Decimal, tick: Decimal) -> Decimal:
    return (value / tick).to_integral_value(rounding=ROUND_HALF_UP) * tick


def _clamp_to_band(value: Decimal, floor: Decimal, ceil: Decimal) -> Decimal:
    if value < floor:
        return floor
    if value > ceil:
        return ceil
    return value


def _kvnd(price: Decimal | float | int) -> int:
    value = Decimal(str(price)) / Decimal("1000")
    return int(value.to_integral_value(rounding=ROUND_HALF_UP))


def _float_or_blank(value: Optional[Decimal]) -> object:
    return float(value) if value is not None else ""


def _is_nan(value: object) -> bool:
    try:
        return math.isnan(float(value))
    except Exception:
        return False


def _weighted_sum(pairs: Sequence[Tuple[Decimal, Optional[Decimal]]]) -> Optional[Decimal]:
    total_weight = Decimal("0")
    total = Decimal("0")
    for weight, value in pairs:
        if value is None:
            continue
        total_weight += weight
        total += weight * value
    if total_weight == 0:
        return None
    return total / total_weight


def _round_to_lot(quantity: float) -> int:
    if quantity <= 0:
        return 0
    lots = round(quantity / 100.0)
    return int(lots * 100)


def _find_repo_root(start: Path) -> Path:
    current = start
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    return start


def _resolve_path(candidate: object, config_dir: Path, repo_root: Path) -> Path:
    if isinstance(candidate, Path):
        path = candidate
    elif isinstance(candidate, str):
        path = Path(candidate)
    else:
        raise EngineError(f"Expected string path, got {type(candidate).__name__}")
    if path.is_absolute():
        return path.resolve()
    config_resolved = (config_dir / path).resolve()
    try:
        config_resolved.relative_to(repo_root)
    except ValueError:
        raise EngineError(
            f"Path '{candidate}' escapes repository root {repo_root}. Use absolute paths for external locations."
        )
    if config_resolved.exists():
        return config_resolved

    root_resolved = (repo_root / path).resolve()
    try:
        root_resolved.relative_to(repo_root)
    except ValueError:
        raise EngineError(
            f"Path '{candidate}' escapes repository root {repo_root}. Use absolute paths for external locations."
        )
    return root_resolved


def _as_repo_relative(path: Path, default: str, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root))
    except Exception:
        return default


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the broker data engine")
    parser.add_argument("--config", type=Path, default=Path("config/data_engine.yaml"))
    parser.add_argument("--profile", default="alpha", help="Portfolio profile name (default: alpha)")
    args = parser.parse_args(argv)
    config = EngineConfig.from_yaml(args.config, profile=args.profile)
    engine = DataEngine(config)
    summary = engine.run()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

