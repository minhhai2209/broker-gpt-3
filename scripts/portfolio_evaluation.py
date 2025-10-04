from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import math
import pandas as pd
import numpy as np


@dataclass
class EvalConfig:
    liq_frac: float = 0.20  # 20% of ADTV per day (sensible default)


def _to_float(x):
    try:
        v = pd.to_numeric(pd.Series([x]), errors='coerce').iloc[0]
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def _weights_from_portfolio(portfolio_df: pd.DataFrame, snapshot_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    px_map = {str(t).upper(): _to_float(p) for t, p in zip(snapshot_df['Ticker'], pd.to_numeric(snapshot_df.get('Price', np.nan), errors='coerce'))}
    rows = []
    total_k = 0.0
    for _, r in portfolio_df.iterrows():
        t = str(r['Ticker']).upper()
        q = float(_to_float(r.get('Quantity')) or 0.0)
        p = float(px_map.get(t) or 0.0)
        v_k = q * p
        rows.append({'Ticker': t, 'Quantity': q, 'Price': p, 'Value_k': v_k})
        total_k += v_k
    df = pd.DataFrame(rows)
    if total_k > 0:
        df['Weight'] = df['Value_k'] / total_k
    else:
        df['Weight'] = 0.0
    return df, total_k


def build_portfolio_evaluation(
    portfolio_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    out_dir: Path,
    *,
    regime: object | None = None,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = EvalConfig()
    base, total_k = _weights_from_portfolio(portfolio_df, snapshot_df)
    # Optional: read days_to_liq fraction from policy (regime)
    try:
        if regime is not None:
            eva = getattr(regime, 'evaluation', {}) or {}
            frac = eva.get('days_to_liq_frac') if isinstance(eva, dict) else None
            if frac is not None:
                cfg.liq_frac = float(frac)
    except Exception:
        pass
    met = metrics_df.set_index('Ticker') if (isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty and 'Ticker' in metrics_df.columns) else pd.DataFrame()

    # Join metrics (Sector, ADTV20, ATR%, Beta)
    def _get(idx: str, key: str):
        try:
            if idx in met.index:
                return _to_float(met.loc[idx].get(key))
        except Exception:
            return None
        return None

    sectors = []
    adtvk = []
    atrp = []
    beta = []
    for t in base['Ticker']:
        s = None
        try:
            if t in met.index:
                raw = met.loc[t].get('Sector')
                s = '' if (raw is None or (isinstance(raw, float) and math.isnan(raw))) else str(raw)
        except Exception:
            s = ''
        sectors.append(s)
        adtvk.append(_get(t, 'AvgTurnover20D_k') or 0.0)
        atrp.append(_get(t, 'ATR14_Pct') or 0.0)
        beta.append(_get(t, 'Beta60D') or 1.0)
    base['Sector'] = sectors
    base['ADTV20_k'] = adtvk
    base['ATR14_Pct'] = atrp
    base['Beta60D'] = beta

    # Liquidity days to liquidate at cfg.liq_frac
    def _days_to_liq(vk: float, adtv_k: float, frac: float) -> float:
        cap = max(1e-9, adtv_k * max(1e-9, frac))
        return float(vk) / cap if cap > 0 else float('inf')

    base['DaysToLiq'] = [
        _days_to_liq(vk, adtv, cfg.liq_frac) for vk, adtv in zip(base['Value_k'], base['ADTV20_k'])
    ]

    # Portfolio stats
    w = base['Weight'].fillna(0.0).to_numpy()
    hhi = float((w ** 2).sum()) if w.size else 0.0
    top1 = float(w.max()) if w.size else 0.0
    top3 = float(np.sort(w)[-3:].sum()) if w.size >= 3 else float(w.sum())
    wavg_atr = float((base['ATR14_Pct'].fillna(0.0) * base['Weight']).sum()) if total_k > 0 else 0.0
    wavg_beta = float((base['Beta60D'].fillna(1.0) * base['Weight']).sum()) if total_k > 0 else 1.0

    # Sector exposures
    sec = base.groupby('Sector', as_index=False)['Value_k'].sum().sort_values('Value_k', ascending=False)
    sec['Weight'] = sec['Value_k'] / total_k if total_k > 0 else 0.0

    # Caps compliance (from regime sizing if available)
    max_pos = None; max_sec = None
    try:
        sz = getattr(regime, 'sizing', {}) or {}
        max_pos = float(sz.get('max_pos_frac')) if sz and sz.get('max_pos_frac') is not None else None
        max_sec = float(sz.get('max_sector_frac')) if sz and sz.get('max_sector_frac') is not None else None
    except Exception:
        max_pos = None; max_sec = None
    base['PosCapExceeded'] = False
    if max_pos is not None and max_pos > 0:
        base['PosCapExceeded'] = base['Weight'] > max_pos
    if max_sec is not None and max_sec > 0 and not sec.empty:
        sec['SecCapExceeded'] = sec['Weight'] > max_sec

    # Write CSV with per-ticker eval
    csv_path = out_dir / 'portfolio_evaluation.csv'
    base_out = base[['Ticker','Quantity','Price','Value_k','Weight','Sector','ADTV20_k','DaysToLiq','ATR14_Pct','Beta60D','PosCapExceeded']].copy()
    base_out.to_csv(csv_path, index=False)

    # Write TXT summary
    txt_path = out_dir / 'portfolio_evaluation.txt'
    lines: list[str] = []
    lines.append('BÁO CÁO ĐÁNH GIÁ DANH MỤC (tự động)')
    lines.append(f"- Tổng giá trị (nghìn): {total_k:,.0f}")
    lines.append(f"- HHI (concentration): {hhi:.4f} | Top-1: {top1*100:.1f}% | Top-3: {top3*100:.1f}%")
    lines.append(f"- ATR% bình quân gia quyền: {wavg_atr:.2f}% | Beta bình quân: {wavg_beta:.2f}")
    if max_pos is not None:
        lines.append(f"- Trần vị thế theo policy (max_pos_frac): {max_pos*100:.1f}%")
    if max_sec is not None:
        lines.append(f"- Trần ngành theo policy (max_sector_frac): {max_sec*100:.1f}%")
    # Top positions by weight
    topN = base.sort_values('Weight', ascending=False).head(5)
    lines.append('- Top vị thế:')
    for _, r in topN.iterrows():
        lines.append(f"  • {r['Ticker']}: {r['Weight']*100:.1f}% | ADTV20 {r['ADTV20_k']:,.0f}k | DaysToLiq@{cfg.liq_frac*100:.0f}%≈{r['DaysToLiq']:.1f}")
    # Sector exposures
    if not sec.empty:
        lines.append('- Phân bổ theo ngành:')
        for _, r in sec.iterrows():
            flag = ''
            if max_sec is not None and max_sec > 0 and r['Weight'] > max_sec:
                flag = ' (vượt trần)'
            lines.append(f"  • {r['Sector'] or 'N/A'}: {r['Weight']*100:.1f}%{flag}")
    # Position cap exceed list
    if base_out['PosCapExceeded'].any():
        lines.append('- Cảnh báo: các mã vượt trần vị thế: ' + ', '.join(base_out[base_out['PosCapExceeded']]['Ticker'].tolist()))
    txt_path.write_text('\n'.join(lines), encoding='utf-8')
    return txt_path, csv_path
