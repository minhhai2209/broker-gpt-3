"""
Auto‑ingest one or more portfolio CSVs from a fixed folder and emit out/portfolio_clean.csv.

Expected input schema per file: Ticker,Quantity,AvgCost[,CostValue]
- Input `AvgCost` may be in VND per share (common broker export).
- Output is STANDARDIZED to THOUSAND VND per share for `AvgCost`.
- If `CostValue` column is present, it is recomputed consistently in THOUSAND units as Quantity × AvgCost(thousand).

If multiple CSV files are present (e.g., user has positions across two broker apps), they are MERGED by ticker:
- Total Quantity = sum of quantities across files (per ticker)
- AvgCost(thousand) = weighted average = sum(Quantity × AvgCost_thousand) / sum(Quantity)
- CostValue(thousand) = Total Quantity × AvgCost(thousand)

Only CSV is supported.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _format_samples(df: pd.DataFrame, column: str, max_items: int = 3) -> str:
    sample_rows = df.head(max_items)
    formatted = []
    for row in sample_rows.itertuples(index=False):
        ticker = getattr(row, 'Ticker', '') or '<blank>'
        value = getattr(row, column)
        formatted.append(f"{ticker}:{value!r}")
    suffix = ' …' if len(df) > max_items else ''
    return ', '.join(formatted) + suffix


def _load_portfolio_csv(src: Path) -> pd.DataFrame:
    df = pd.read_csv(src)
    req = {'Ticker', 'Quantity', 'AvgCost'}
    if not req.issubset(set(df.columns)):
        raise SystemExit(f'{src.name}: CSV missing required columns: Ticker, Quantity, AvgCost')
    df = df[['Ticker', 'Quantity', 'AvgCost'] + ([c for c in ['CostValue'] if c in df.columns])].copy()
    df['Ticker'] = df['Ticker'].astype(str).str.upper().str.strip().str.replace(r'[^A-Z0-9]+', '', regex=True)

    qty_raw = df['Quantity'].copy()
    qty_numeric = pd.to_numeric(qty_raw, errors='coerce')
    invalid_qty = qty_numeric.isna() | ~np.isfinite(qty_numeric)
    if invalid_qty.any():
        samples = _format_samples(df.loc[invalid_qty, ['Ticker', 'Quantity']], 'Quantity')
        raise SystemExit(f'{src.name}: invalid Quantity detected (non-numeric/NaN): {samples}')
    fractional_qty = (qty_numeric % 1 != 0)
    if fractional_qty.any():
        samples = _format_samples(df.loc[fractional_qty, ['Ticker', 'Quantity']], 'Quantity')
        raise SystemExit(f'{src.name}: Quantity must be whole shares; offending rows: {samples}')
    df['Quantity'] = qty_numeric.astype(int)

    avg_raw = df['AvgCost'].copy()
    avg_numeric = pd.to_numeric(avg_raw, errors='coerce')
    invalid_avg = avg_numeric.isna() | ~np.isfinite(avg_numeric)
    if invalid_avg.any():
        samples = _format_samples(df.loc[invalid_avg, ['Ticker', 'AvgCost']], 'AvgCost')
        raise SystemExit(f'{src.name}: invalid AvgCost detected (non-numeric/NaN): {samples}')
    df['AvgCost'] = avg_numeric

    df['CostRow'] = df['Quantity'] * df['AvgCost']
    return df


def ingest_portfolio(indir: str = 'in/portfolio', outdir: str = 'out') -> Path:
    indir = Path(indir)
    indir.mkdir(parents=True, exist_ok=True)
    files = [p for p in indir.iterdir() if p.is_file() and not p.name.startswith('.') and p.suffix.lower() == '.csv']
    if not files:
        raise SystemExit(f'No portfolio CSV found in {indir}. Place one or more CSV files with columns Ticker,Quantity,AvgCost.')

    frames: list[pd.DataFrame] = []
    for src in files:
        frames.append(_load_portfolio_csv(src))

    merged = pd.concat(frames, ignore_index=True)
    # Drop rows with empty ticker
    merged = merged[merged['Ticker'].astype(str).str.len() > 0]
    if merged.empty:
        raise SystemExit('Portfolio ingest produced no valid rows after cleaning; check input CSVs for blank tickers.')
    # Group and compute weighted average cost (thousand)
    agg = merged.groupby('Ticker', as_index=False).agg(
        Quantity=('Quantity','sum'),
        CostSum=('CostRow','sum'),
    )
    # Avoid division by zero
    agg['AvgCost'] = (agg['CostSum'] / agg['Quantity'].where(agg['Quantity'] != 0, other=1)).where(agg['Quantity'] != 0, other=0.0)
    agg['CostValue'] = agg['Quantity'] * agg['AvgCost']
    # Order columns
    df_out = agg[['Ticker','Quantity','AvgCost','CostValue']].copy()

    od = Path(outdir)
    od.mkdir(parents=True, exist_ok=True)
    outp = od / 'portfolio_clean.csv'
    df_out.to_csv(outp, index=False)
    src_names = ', '.join(p.name for p in files[:5]) + (' ...' if len(files) > 5 else '')
    print(f'Wrote {outp} from {len(files)} file(s): {src_names}')
    return outp


def ingest_portfolio_df(indir: str = 'in/portfolio') -> pd.DataFrame:
    """Return cleaned/merged portfolio DataFrame with columns: Ticker,Quantity,AvgCost,CostValue.
    Does not write to disk.
    """
    indir_p = Path(indir)
    indir_p.mkdir(parents=True, exist_ok=True)
    files = [p for p in indir_p.iterdir() if p.is_file() and not p.name.startswith('.') and p.suffix.lower() == '.csv']
    if not files:
        raise SystemExit(f'No portfolio CSV found in {indir_p}. Place one or more CSV files with columns Ticker,Quantity,AvgCost.')
    frames: list[pd.DataFrame] = []
    for src in files:
        frames.append(_load_portfolio_csv(src))

    merged = pd.concat(frames, ignore_index=True)
    merged = merged[merged['Ticker'].astype(str).str.len() > 0]
    if merged.empty:
        raise SystemExit('Portfolio ingest produced no valid rows after cleaning; check input CSVs for blank tickers.')
    agg = merged.groupby('Ticker', as_index=False).agg(
        Quantity=('Quantity','sum'),
        CostSum=('CostRow','sum'),
    )
    agg['AvgCost'] = (agg['CostSum'] / agg['Quantity'].where(agg['Quantity'] != 0, other=1)).where(agg['Quantity'] != 0, other=0.0)
    agg['CostValue'] = agg['Quantity'] * agg['AvgCost']
    return agg[['Ticker','Quantity','AvgCost','CostValue']].copy()
