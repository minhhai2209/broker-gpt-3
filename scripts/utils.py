from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
import csv as _csv
import importlib.util
import pandas as pd

_ZONEINFO_AVAILABLE = importlib.util.find_spec("zoneinfo") is not None
if _ZONEINFO_AVAILABLE:
    from zoneinfo import ZoneInfo, available_timezones  # type: ignore
else:  # pragma: no cover - fallback for limited environments
    ZoneInfo = None  # type: ignore
    def available_timezones():  # type: ignore
        return set()


def detect_session_phase_now_vn() -> str:
    if _ZONEINFO_AVAILABLE and "Asia/Ho_Chi_Minh" in available_timezones():
        tz = ZoneInfo("Asia/Ho_Chi_Minh")  # type: ignore[arg-type]
        now = datetime.now(tz).time()
    else:
        now = datetime.now().time()
    if now < time(9, 0):
        return "pre"
    if time(9, 0) <= now < time(11, 30):
        return "morning"
    if time(11, 30) <= now < time(13, 0):
        return "lunch"
    if time(13, 0) <= now < time(14, 30):
        return "afternoon"
    if time(14, 30) <= now < time(14, 45):
        return "ATC"
    return "post"


def hose_tick_size(price_thousand: float) -> float:
    if price_thousand < 10:
        return 0.01
    if price_thousand < 49.95:
        return 0.05
    return 0.10


def clip_to_band(price: float, band_floor: float | None, band_ceiling: float | None) -> float:
    if band_floor is not None:
        price = max(price, band_floor)
    if band_ceiling is not None:
        price = min(price, band_ceiling)
    return price


def round_to_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return round(round(price / tick) * tick, 2)


def to_float(x) -> float | None:
    if pd.isna(x):
        return None
    numeric = pd.to_numeric(pd.Series([x]), errors='coerce').iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def load_universe_from_files(industry_map_path: str = 'data/industry_map.csv') -> list[str]:
    src = Path(industry_map_path)
    if not src.exists():
        raise FileNotFoundError(f"Missing industry map CSV: {src}")
    tickers: list[str] = []
    with src.open(newline='', encoding='utf-8') as f:
        r = _csv.DictReader(f)
        for row in r:
            t = (row.get('Ticker') or '').strip().upper()
            if t:
                tickers.append(t)
    seen: set[str] = set(); uniq: list[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t); uniq.append(t)
    return uniq
