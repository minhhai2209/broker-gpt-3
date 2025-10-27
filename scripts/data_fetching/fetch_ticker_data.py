"""Self-contained OHLCV cache using VNDIRECT dchart API."""
from __future__ import annotations
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

VN_TZ = timezone(timedelta(hours=7))
BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DATA_DIR = BASE_DIR / 'out' / 'data'


def fetch_history(symbol: str, resolution: str, frm: int, to: int) -> Optional[Dict]:
    """Fetch OHLC history. Retries transient non-JSON/empty responses a few times, then fails.

    Returns parsed dict on success, or None when API reports no data for the window.
    """
    url = f"https://dchart-api.vndirect.com.vn/dchart/history?symbol={symbol}&resolution={resolution}&from={frm}&to={to}"
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Referer': 'https://dchart.vndirect.com.vn/'
    }
    last_exc = None
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=10, headers=headers)
            r.raise_for_status()
            # Some infra occasionally returns empty body with 200; treat as retryable
            if not r.text or not r.text.strip():
                raise ValueError('empty body')
            data = r.json()
            if data.get('s') != 'ok' or not data.get('t'):
                return None
            return data
        except Exception as exc:
            last_exc = exc
            # Brief backoff on JSON/empty-body/network hiccups
            time.sleep(0.6 * (attempt + 1))
            continue
    # Final failure: surface rich context
    try:
        ct = r.headers.get('Content-Type', '') if 'r' in locals() else ''
        preview = (r.text or '')[:200] if 'r' in locals() else ''
    except Exception:
        ct = ''
        preview = ''
    # Graceful degrade: warn and record symbol, then signal no new data for this pass.
    print(
        f"[warn] VNDIRECT dchart API error after retries for {symbol} {resolution} from={frm} to={to}: "
        f"{type(last_exc).__name__}: {last_exc}. CT='{ct}' Preview={preview!r}"
    )
    try:
        OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with (OUT_DATA_DIR / 'fetch_errors.txt').open('a', encoding='utf-8') as f:
            f.write(f"{symbol}\n")
    except Exception:
        pass
    return None


def save_csv(path: Path, data: Dict):
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    if not data or 't' not in data or not data['t']:
        return
    df = pd.DataFrame({
        't': data['t'], 'open': data['o'], 'high': data['h'], 'low': data['l'], 'close': data['c'], 'volume': data.get('v', [None]*len(data['t']))
    })
    # Add human-readable VN date column for convenience
    ts = pd.to_datetime(df['t'], unit='s', utc=True).dt.tz_convert(VN_TZ)
    df['date_vn'] = ts.dt.strftime('%Y-%m-%d')
    df.to_csv(path, index=False)


def merge_incremental(path: Path, incoming: Dict) -> bool:
    """Merge incoming OHLC dict into existing CSV by timestamp 't'. Returns True if wrote file."""
    import pandas as pd
    if not incoming or 't' not in incoming or not incoming['t']:
        return False
    new = pd.DataFrame({
        't': incoming['t'], 'open': incoming['o'], 'high': incoming['h'], 'low': incoming['l'], 'close': incoming['c'], 'volume': incoming.get('v', [None]*len(incoming['t']))
    })
    if path.exists():
        cur = pd.read_csv(path)
        all_df = pd.concat([cur, new], ignore_index=True)
        # Drop duplicates by timestamp, keep last occurrence
        all_df = all_df.drop_duplicates(subset=['t'], keep='last').sort_values('t').reset_index(drop=True)
        # Recompute human-readable date in VN
        ts = pd.to_datetime(all_df['t'], unit='s', utc=True).dt.tz_convert(VN_TZ)
        all_df['date_vn'] = ts.dt.strftime('%Y-%m-%d')
        path.parent.mkdir(parents=True, exist_ok=True)
        all_df.to_csv(path, index=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        new = new.sort_values('t').reset_index(drop=True)
        ts = pd.to_datetime(new['t'], unit='s', utc=True).dt.tz_convert(VN_TZ)
        new['date_vn'] = ts.dt.strftime('%Y-%m-%d')
        new.to_csv(path, index=False)
    return True


def last_timestamp(path: Path) -> Optional[int]:
    import pandas as pd
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=['t'])
    if df.empty:
        return None
    values = pd.to_numeric(df['t'], errors='coerce').dropna()
    if values.empty:
        return None
    return int(values.max())


def first_timestamp_and_count(path: Path) -> Tuple[Optional[int], int]:
    """Return (earliest_ts, row_count) from an existing CSV, or (None, 0)."""
    import pandas as pd
    if not path.exists():
        return None, 0
    df = pd.read_csv(path, usecols=['t'])
    if df.empty:
        return None, 0
    ts = pd.to_numeric(df['t'], errors='coerce').dropna()
    if ts.empty:
        return None, 0
    return int(ts.min()), int(len(ts))


def ensure_ohlc_cache(ticker: str,
                      outdir: str = 'out/data',
                      min_days: int = 400,
                      resolution: str = 'D') -> None:
    """Ensure per-ticker OHLC CSV exists and is up-to-date.

    - Creates file if missing with at least `min_days` history.
    - Backfills older data if existing file has fewer than `min_days` rows.
    - Appends missing recent bars to current time.
    """
    outp = Path(outdir) / f"{ticker}_daily.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    now_ts = int(datetime.now(VN_TZ).timestamp())

    # 1) If file missing, fetch full window [now - min_days, now]
    if not outp.exists() or outp.stat().st_size == 0:
        frm = int((datetime.now(VN_TZ) - timedelta(days=min_days)).timestamp())
        data = fetch_history(ticker, resolution, frm, now_ts)
        if data and data.get('s') == 'ok' and data.get('t'):
            save_csv(outp, data)
            print(f"Seeded {ticker}: {len(data['t'])} bars")
        else:
            print(f"No data to seed {ticker}")
        return

    # 2) Backfill older part if fewer than min_days rows
    earliest_ts, row_count = first_timestamp_and_count(outp)
    if row_count < min_days:
        backfill_to = int((datetime.now(VN_TZ) - timedelta(days=min_days)).timestamp())
        # fetch [backfill_to, earliest_ts-60]
        if earliest_ts is None:
            back_from = backfill_to
            back_to = now_ts
        else:
            back_from = backfill_to
            back_to = max(backfill_to, earliest_ts - 60)
        if back_from < back_to:
            data_old = fetch_history(ticker, resolution, back_from, back_to)
            if data_old and data_old.get('s') == 'ok' and data_old.get('t'):
                merge_incremental(outp, data_old)
                print(f"Backfilled {ticker}: +{len(data_old['t'])} bars")

    # 3) Append the most recent tail if any
    lt = last_timestamp(outp)
    frm_new = max(0, (lt + 60) if lt is not None else int((datetime.now(VN_TZ) - timedelta(days=min_days)).timestamp()))
    if frm_new < now_ts:
        data_new = fetch_history(ticker, resolution, frm_new, now_ts)
        if data_new and data_new.get('s') == 'ok' and data_new.get('t'):
            merge_incremental(outp, data_new)
            print(f"Updated {ticker}: +{len(data_new['t'])} bars")
        else:
            # Nothing new – keep silent or log lightly
            pass


def ensure_and_load_history_df(tickers: List[str], outdir: str = 'out/data', min_days: int = 400, resolution: str = 'D'):
    """Ensure caches for tickers, then load and merge into a single DataFrame.
    Returns columns: Date,Ticker,Open,High,Low,Close,Volume,t
    """
    import pandas as pd
    tickers = [t.strip().upper() for t in tickers if str(t).strip()]
    # Reset fetch error log for this run
    try:
        errf = Path(outdir) / 'fetch_errors.txt'
        errf.unlink(missing_ok=True)
    except Exception:
        pass
    for t in tickers:
        ensure_ohlc_cache(t, outdir=outdir, min_days=min_days, resolution=resolution)
    rows = []
    for t in tickers:
        path = Path(outdir) / f"{t}_daily.csv"
        if not path.exists() or path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        # Harmonize
        for c in ['t','open','high','low','close','volume']:
            if c not in df.columns: df[c] = pd.NA
        df_out = pd.DataFrame({
            'Date': (pd.to_datetime(df['t'], unit='s', utc=True).dt.tz_convert(VN_TZ).dt.strftime('%Y-%m-%d') if 't' in df.columns else pd.NA),
            'Ticker': t,
            'Open': df['open'],
            'High': df['high'],
            'Low': df['low'],
            'Close': df['close'],
            'Volume': df['volume'],
            't': df['t'],
        })
        rows.append(df_out)
    if not rows:
        return pd.DataFrame(columns=['Date','Ticker','Open','High','Low','Close','Volume','t'])
    out = pd.concat(rows, ignore_index=True)
    if 't' in out.columns:
        out = out.sort_values(['Ticker','t']).reset_index(drop=True)
    return out
