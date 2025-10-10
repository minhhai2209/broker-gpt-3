"""Shared helpers for the ML calibrator train/score scripts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .ml_config import MLConfig, load_config


BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "out"
OUT_ORDERS_DIR = OUT_DIR / "orders"
OUT_ML_DIR = OUT_DIR / "ml"
OUT_DEBUG_DIR = OUT_DIR / "debug"
MODELS_DIR = BASE_DIR / "models"

VN_TZ = timezone(timedelta(hours=7))


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ORDERS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ML_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _to_datetime(series: pd.Series) -> pd.Series:
    vals = pd.to_datetime(series, errors="coerce")
    vals = vals.dt.tz_localize(VN_TZ) if vals.dt.tz is None else vals.dt.tz_convert(VN_TZ)
    return vals


def append_ml_log(event: Dict[str, Any]) -> None:
    _ensure_dirs()
    path = OUT_DEBUG_DIR / "ml_calibrator.ndjson"
    event = dict(event)
    event.setdefault("ts", datetime.now(timezone.utc).isoformat())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def load_input_frames(cfg: MLConfig) -> Dict[str, pd.DataFrame]:
    paths = cfg.io_paths
    frames: Dict[str, pd.DataFrame] = {}
    required = {
        "snapshot",
        "metrics",
        "sector_strength",
        "session",
        "prices",
        "portfolio",
    }
    missing = [key for key in required if key not in paths]
    if missing:
        raise FileNotFoundError(f"ML config missing IO path(s): {', '.join(sorted(missing))}")
    for name, rel_path in paths.items():
        path = BASE_DIR / rel_path
        if not path.exists():
            raise FileNotFoundError(f"Missing required input for ML calibrator: {path}")
        frames[name] = pd.read_csv(path)
    return frames


def _pivot_prices(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    if "Date" not in prices.columns or "Ticker" not in prices.columns:
        raise KeyError("prices_history.csv requires 'Date' and 'Ticker' columns")
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices = prices.dropna(subset=["Date"])
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper()
    if "Close" not in prices.columns:
        raise KeyError("prices_history.csv requires 'Close' column")
    prices["Close"] = pd.to_numeric(prices["Close"], errors="coerce")
    prices = prices.dropna(subset=["Close"])
    pivot = prices.pivot_table(index="Date", columns="Ticker", values="Close")
    pivot = pivot.sort_index()
    pivot = pivot.replace([np.inf, -np.inf], np.nan)
    pivot = pivot.dropna(axis=0, how="all")
    return pivot


def _rolling_features(pivot: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    returns = pivot.pct_change()
    log_ret = np.log(pivot).diff()
    features: Dict[str, pd.DataFrame] = {}
    features["ret_1d"] = returns
    features["ret_3d"] = pivot.pct_change(3)
    features["ret_5d"] = pivot.pct_change(5)
    features["ret_20d"] = pivot.pct_change(20)
    features["vol_10d"] = log_ret.rolling(window=10, min_periods=5).std()
    features["vol_20d"] = log_ret.rolling(window=20, min_periods=10).std()
    rolling_max = pivot.rolling(window=20, min_periods=10).max()
    features["drawdown_20d"] = pivot / rolling_max - 1.0
    ma20 = pivot.rolling(window=20, min_periods=5).mean()
    ma50 = pivot.rolling(window=50, min_periods=20).mean()
    features["trend_ma20"] = pivot / ma20 - 1.0
    features["trend_ma50"] = pivot / ma50 - 1.0
    delta = pivot.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window=14, min_periods=7).mean()
    roll_down = down.rolling(window=14, min_periods=7).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    features["rsi_14"] = rsi
    return features


def build_feature_panel(prices_history: pd.DataFrame, horizon: int, index_symbol: str = "VNINDEX") -> pd.DataFrame:
    pivot = _pivot_prices(prices_history)
    if index_symbol not in pivot.columns:
        raise KeyError(f"Price history missing index symbol '{index_symbol}' for excess return labels")
    features = _rolling_features(pivot)
    # Convert to long format
    stacked = []
    for name, frame in features.items():
        stacked.append(frame.stack(future_stack=True).rename(name))
    feat_df = pd.concat(stacked, axis=1)
    feat_df = feat_df.reset_index()
    feat_df = feat_df.rename(columns={"level_0": "date", "level_1": "ticker", "Date": "date", "Ticker": "ticker"})
    feat_df["date"] = pd.to_datetime(feat_df["date"], errors="coerce")
    feat_df = feat_df.dropna(subset=["date", "ticker"])
    feat_df["ticker"] = feat_df["ticker"].astype(str).str.upper()

    # Compute future excess returns for labelling
    future = pivot.shift(-horizon) / pivot - 1.0
    excess = future.subtract(future[index_symbol], axis=0)
    excess_long = excess.stack(future_stack=True).reset_index()
    excess_long = excess_long.rename(columns={"level_0": "date", "level_1": "ticker", 0: "excess_future", "Date": "date", "Ticker": "ticker"})
    excess_long["date"] = pd.to_datetime(excess_long["date"], errors="coerce")
    excess_long["ticker"] = excess_long["ticker"].astype(str).str.upper()
    feat_df = feat_df.merge(excess_long, on=["date", "ticker"], how="left")
    return feat_df


def attach_labels(feature_panel: pd.DataFrame, cfg: MLConfig, horizon: int) -> pd.DataFrame:
    df = feature_panel.copy()
    fee = float(cfg.raw.get("train", {}).get("fee_bps", 0.0) or 0.0) / 10000.0
    slip = float(cfg.raw.get("train", {}).get("slip_bps", 0.0) or 0.0) / 10000.0
    theta = float(cfg.raw.get("train", {}).get("atr_theta", 0.0) or 0.0)
    threshold = fee + slip + theta * df.get("vol_20d", 0.0).fillna(0.0)
    df["label"] = (df["excess_future"] > threshold).astype(float)
    df.loc[df["excess_future"].isna(), "label"] = np.nan
    # Drop rows without enough history (vol_20d etc) or label
    df = df.dropna(subset=["date", "ticker"])
    cutoff = df["date"].max() - pd.Timedelta(days=horizon)
    df = df[df["date"] <= cutoff]
    df = df.dropna(subset=["label"])
    df = df[df["ticker"] != "VNINDEX"]
    return df


def latest_feature_slice(feature_panel: pd.DataFrame) -> Tuple[pd.Timestamp, pd.DataFrame]:
    if feature_panel.empty:
        raise ValueError("Feature panel is empty; cannot score")
    feature_panel = feature_panel.dropna(subset=["date"])  # ensure no NaT
    latest_date = feature_panel["date"].max()
    latest = feature_panel[feature_panel["date"] == latest_date].copy()
    latest = latest[latest["ticker"] != "VNINDEX"]
    return latest_date, latest


def feature_columns() -> Sequence[str]:
    return [
        "ret_1d",
        "ret_3d",
        "ret_5d",
        "ret_20d",
        "vol_10d",
        "vol_20d",
        "drawdown_20d",
        "trend_ma20",
        "trend_ma50",
        "rsi_14",
    ]


@dataclass
class ModelBundle:
    model: Any
    metadata: Dict[str, Any]


def _model_dir_name(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def save_model_bundle(model: Any, metadata: Dict[str, Any]) -> Path:
    _ensure_dirs()
    now = datetime.now(VN_TZ)
    dir_path = MODELS_DIR / _model_dir_name(now)
    idx = 1
    while dir_path.exists():
        idx += 1
        dir_path = MODELS_DIR / f"{_model_dir_name(now)}_{idx:02d}"
    dir_path.mkdir(parents=True, exist_ok=False)
    model_path = dir_path / "model.txt"
    try:
        model.save_model(str(model_path))  # type: ignore[attr-defined]
    except AttributeError as exc:  # pragma: no cover - unexpected type
        raise RuntimeError("Model object does not support save_model") from exc
    meta_path = dir_path / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, default=str)
    return dir_path


def find_latest_model() -> Optional[Path]:
    if not MODELS_DIR.exists():
        return None
    candidates = [p for p in MODELS_DIR.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1]


def load_model_bundle(path: Optional[Path] = None) -> ModelBundle:
    model_dir = path or find_latest_model()
    if model_dir is None:
        raise FileNotFoundError("No trained ML model found under models/")
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {model_dir}")
    with meta_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    model_path = model_dir / "model.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact in {model_dir}")
    try:
        import lightgbm as lgb  # type: ignore

        model = lgb.Booster(model_file=str(model_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to load LightGBM model from {model_path}: {exc}") from exc
    return ModelBundle(model=model, metadata=metadata)


def predict_proba(bundle: ModelBundle, frame: pd.DataFrame) -> np.ndarray:
    feats = feature_columns()
    missing = [col for col in feats if col not in frame.columns]
    if missing:
        raise KeyError(f"Missing required feature columns for scoring: {', '.join(missing)}")
    values = frame[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    raw = bundle.model.predict(values)  # type: ignore[attr-defined]
    preds = np.asarray(raw, dtype=float)
    preds = np.clip(preds, 0.0, 1.0)
    return preds


def compute_regime_adjustments(
    session: pd.DataFrame,
    sector_strength: pd.DataFrame,
    cfg: MLConfig,
    risk_on_prob: float,
) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
    clamp_budget = cfg.inference.get("clamp", {}).get("buy_budget_frac", [0.0, 0.3])
    clamp_bias = cfg.inference.get("clamp", {}).get("bias", [-0.2, 0.2])
    lo_b, hi_b = float(clamp_budget[0]), float(clamp_budget[1])
    lo_bias, hi_bias = float(clamp_bias[0]), float(clamp_bias[1])
    if risk_on_prob >= 0.65:
        budget = 0.08
        limits = {"new_max": 3, "add_max": 4}
    elif risk_on_prob >= 0.45:
        budget = 0.05
        limits = {"new_max": 2, "add_max": 3}
    else:
        budget = 0.02
        limits = {"new_max": 1, "add_max": 2}
    budget = max(lo_b, min(hi_b, budget))

    bias: Dict[str, Any] = {}
    if not sector_strength.empty and {"Sector", "sector_strength_percentile"}.issubset(sector_strength.columns):
        for _, row in sector_strength.iterrows():
            sector = str(row.get("Sector", "")).strip()
            if not sector:
                continue
            try:
                pct = float(row.get("sector_strength_percentile"))
            except Exception:
                continue
            if not np.isfinite(pct):
                continue
            if pct >= 0.65:
                bias[f"sector_bias.{sector}"] = min(hi_bias, max(lo_bias, 0.05 + (pct - 0.65) * 0.2))
            elif pct <= 0.35:
                bias[f"sector_bias.{sector}"] = max(lo_bias, min(hi_bias, -0.05 - (0.35 - pct) * 0.2))
    return budget, limits, bias


def extract_risk_on_probability(session: pd.DataFrame) -> float:
    if session.empty:
        return 0.5
    for column in ("risk_on_prob", "RiskOnProb", "RiskOnProbability"):
        if column in session.columns:
            try:
                val = float(session.iloc[0][column])
                if np.isfinite(val):
                    return max(0.0, min(1.0, val))
            except Exception:
                continue
    if "IndexChangePct" in session.columns:
        try:
            change = float(session.iloc[0]["IndexChangePct"])
            return max(0.0, min(1.0, 0.5 + 0.1 * change))
        except Exception:
            return 0.5
    return 0.5


def write_patch(
    preds: pd.DataFrame,
    cfg: MLConfig,
    budget: float,
    limits: Dict[str, Any],
    bias: Dict[str, Any],
    risk_on_prob: float,
    model_meta: Dict[str, Any],
) -> Dict[str, Any]:
    _ensure_dirs()
    p_gate = float(cfg.inference.get("p_gate", 0.65) or 0.65)
    execution = {
        "fill_prob_target": float(cfg.execution.get("fill_prob_target", 0.65) or 0.65),
        "slip_bps_cap": float(cfg.execution.get("slip_bps_cap", 40) or 40),
    }
    now = datetime.now(VN_TZ)
    ttl = now + timedelta(hours=12)
    patch = {
        "meta": {
            "source": "cal_ml",
            "time": now.isoformat(),
            "ttl": ttl.isoformat(),
            "model": model_meta,
            "p_gate": p_gate,
            "risk_on_prob": risk_on_prob,
        },
        "set": {"buy_budget_frac": budget},
        "limits": limits,
        "bias": bias,
        "gate": {row["ticker"]: {"p_succ": float(row["p_succ"]) } for _, row in preds.iterrows()},
        "exec": execution,
    }
    patch_path = OUT_ORDERS_DIR / "patch_ml.json"
    with patch_path.open("w", encoding="utf-8") as handle:
        json.dump(patch, handle, ensure_ascii=False, indent=2)
    return patch


def write_predictions(preds: pd.DataFrame) -> None:
    _ensure_dirs()
    preds.to_csv(OUT_ML_DIR / "preds_daily.csv", index=False)


def write_metrics(metrics: Dict[str, Any]) -> None:
    _ensure_dirs()
    path = OUT_ML_DIR / "metrics_oos.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)


def write_empty_patch(cfg: MLConfig, note: str) -> Dict[str, Any]:
    _ensure_dirs()
    now = datetime.now(VN_TZ)
    ttl = now + timedelta(hours=4)
    meta = {
        "source": "cal_ml",
        "time": now.isoformat(),
        "ttl": ttl.isoformat(),
        "p_gate": float(cfg.inference.get("p_gate", 0.65) or 0.65),
        "note": note,
    }
    patch = {
        "meta": meta,
        "set": {},
        "limits": {},
        "bias": {},
        "gate": {},
        "exec": {},
    }
    with (OUT_ORDERS_DIR / "patch_ml.json").open("w", encoding="utf-8") as handle:
        json.dump(patch, handle, ensure_ascii=False, indent=2)
    return patch
