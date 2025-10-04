from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple
import re

NUMERIC_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _safe_float(x):
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(',', '')
    if not s or s.lower() in {'nan', 'none'}:
        return 0.0
    return float(s) if NUMERIC_PATTERN.match(s) else 0.0


def build_portfolio_pnl(
    portfolio_csv: str = 'out/portfolio_clean.csv',
    snapshot_csv: str = 'out/snapshot.csv',
    out_summary_csv: str = 'out/portfolio_pnl_summary.csv',
    out_by_sector_csv: str = 'out/portfolio_pnl_by_sector.csv',
) -> Tuple[Path, Path]:
    qty: Dict[str, int] = {}
    avgcost: Dict[str, float] = {}
    with open(portfolio_csv, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            t = (r.get('Ticker') or '').strip().upper()
            if not t:
                continue
            qty[t] = int(float(r.get('Quantity') or 0))
            avgcost[t] = _safe_float(r.get('AvgCost') or 0)

    # Build sector map from snapshot (if present) or fallback to industry_map.csv
    sector_map: Dict[str, str] = {}
    index_labels = {"VNINDEX", "VN30", "VN100"}
    try:
        with open(snapshot_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if 'Sector' in reader.fieldnames if reader.fieldnames else False:
                for r in reader:
                    t = (r.get('Ticker') or '').strip().upper()
                    if not t:
                        continue
                    s = (r.get('Sector') or '').strip()
                    if s:
                        sector_map[t] = s
    except FileNotFoundError:
        pass
    # Fallback to static industry map if snapshot lacks Sector
    if not sector_map:
        try:
            with open('data/industry_map.csv', newline='', encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    t = (r.get('Ticker') or '').strip().upper()
                    s = (r.get('Sector') or '').strip()
                    if t and s:
                        sector_map[t] = s
        except FileNotFoundError:
            pass

    rows = []
    with open(snapshot_csv, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            t = (r.get('Ticker') or '').strip().upper()
            if not t:
                continue
            q = qty.get(t, 0)
            if q <= 0:
                continue
            px = _safe_float(r.get('Price') or r.get('P') or 0)
            ac = avgcost.get(t, _safe_float(r.get('AvgCost') or 0))
            raw_sec = (r.get('Sector') or '').strip()
            sector = raw_sec or sector_map.get(t) or ('Index' if t in index_labels else 'N/A')
            cost = ac * q
            mkt = px * q
            pnl = mkt - cost
            rows.append((t, sector, q, ac, px, cost, mkt, pnl))

    total_cost = sum(r[5] for r in rows)
    total_mkt = sum(r[6] for r in rows)
    total_pnl = sum(r[7] for r in rows)
    ret_pct = (total_pnl / total_cost * 100.0) if total_cost else 0.0

    p_summary = Path(out_summary_csv)
    p_summary.parent.mkdir(parents=True, exist_ok=True)
    with p_summary.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['TotalCost', 'TotalMarket', 'TotalPnL', 'ReturnPct'])
        w.writerow([f'{total_cost:.1f}', f'{total_mkt:.1f}', f'{total_pnl:.1f}', f'{ret_pct:.2f}'])

    by_sec: Dict[str, Tuple[float, float, float]] = {}
    for _, sec, _q, _ac, _px, cost, mkt, pnl in rows:
        c, m, p = by_sec.get(sec, (0.0, 0.0, 0.0))
        by_sec[sec] = (c + cost, m + mkt, p + pnl)

    p_by_sector = Path(out_by_sector_csv)
    with p_by_sector.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Sector', 'Cost', 'Market', 'PnL', 'ReturnPct'])
        for sec, (c, m, p) in sorted(by_sec.items(), key=lambda x: x[0]):
            pct = (p / c * 100.0) if c else 0.0
            w.writerow([sec, f'{c:.1f}', f'{m:.1f}', f'{p:.1f}', f'{pct:.2f}'])

    return p_summary, p_by_sector


def build_portfolio_pnl_dfs(portfolio_df: 'pd.DataFrame', snapshot_df: 'pd.DataFrame') -> tuple['pd.DataFrame','pd.DataFrame']:
    import pandas as pd
    from pathlib import Path
    qty = {str(t).upper(): int(q) for t, q in zip(portfolio_df.get('Ticker', []), portfolio_df.get('Quantity', []))} if not portfolio_df.empty else {}
    avgcost = {str(t).upper(): float(ac) for t, ac in zip(portfolio_df.get('Ticker', []), portfolio_df.get('AvgCost', []))} if not portfolio_df.empty else {}
    # Build sector map: prefer snapshot column if present; otherwise fallback to data/industry_map.csv
    sector_map: dict[str, str] = {}
    index_labels = {"VNINDEX", "VN30", "VN100"}
    if not snapshot_df.empty and 'Sector' in snapshot_df.columns:
        tmp = snapshot_df[['Ticker', 'Sector']].copy()
        tmp['Ticker'] = tmp['Ticker'].astype(str).str.upper()
        tmp['Sector'] = tmp['Sector'].astype(str).str.strip()
        sector_map = {t: s for t, s in zip(tmp['Ticker'], tmp['Sector']) if s}
    if not sector_map:
        p = Path('data/industry_map.csv')
        if p.exists():
            imap = pd.read_csv(p)
            if {'Ticker','Sector'}.issubset(imap.columns):
                tickers = imap['Ticker'].astype(str).str.upper()
                sectors = imap['Sector'].astype(str).str.strip()
                sector_map = {t: s for t, s in zip(tickers, sectors) if s}
    rows = []
    for _, r in snapshot_df.iterrows():
        t = str(r.get('Ticker') or '').upper()
        if not t:
            continue
        q = qty.get(t, 0)
        if q <= 0:
            continue
        px = float(r.get('Price') or r.get('P') or 0)
        ac = avgcost.get(t, float(r.get('AvgCost') or 0))
        raw_sec = str(r.get('Sector') or '').strip()
        sector = raw_sec or sector_map.get(t) or ('Index' if t in index_labels else 'N/A')
        cost = ac * q
        mkt = px * q
        pnl = mkt - cost
        rows.append((t, sector, q, ac, px, cost, mkt, pnl))
    df = pd.DataFrame(rows, columns=['Ticker','Sector','Qty','AvgCost','Price','Cost','Mkt','PnL'])
    total_cost = float(df['Cost'].sum()) if not df.empty else 0.0
    total_mkt = float(df['Mkt'].sum()) if not df.empty else 0.0
    total_pnl = float(df['PnL'].sum()) if not df.empty else 0.0
    ret_pct = (total_pnl / total_cost * 100.0) if total_cost else 0.0
    summary = pd.DataFrame([[f'{total_cost:.1f}', f'{total_mkt:.1f}', f'{total_pnl:.1f}', f'{ret_pct:.2f}']], columns=['TotalCost','TotalMarket','TotalPnL','ReturnPct'])
    by = df.groupby('Sector', as_index=False).agg(Cost=('Cost','sum'), Market=('Mkt','sum'), PnL=('PnL','sum')) if not df.empty else pd.DataFrame(columns=['Sector','Cost','Market','PnL'])
    if not by.empty:
        by['ReturnPct'] = by.apply(lambda r: (r['PnL']/r['Cost']*100.0) if r['Cost'] else 0.0, axis=1)
        by['Cost'] = by['Cost'].map(lambda x: f'{x:.1f}')
        by['Market'] = by['Market'].map(lambda x: f'{x:.1f}')
        by['PnL'] = by['PnL'].map(lambda x: f'{x:.1f}')
        by['ReturnPct'] = by['ReturnPct'].map(lambda x: f'{x:.2f}')
    return summary, by
