from __future__ import annotations

"""
Calibrate regime_model intercept and threshold using VNINDEX history.

Approach
- Keep component means/std and weights from the current policy_overrides.
- Build a daily time series of component values we can compute robustly from index data:
  trend_strength, volatility (EW std annualized), index_return (6-day smoothed daily %),
  drawdown, ATR14 percentile. For unknown components present in policy, use their
  configured mean (z=0), so they don't bias calibration.
- Optimize intercept to minimize log-loss with labels = 1{fwd_return_{h}>0}.
- Pick threshold by maximizing Youden's J statistic (TPR-FPR) on the calibration set.

Usage
  python -m scripts.tuning.calibrators.calibrate_regime --horizon 21 --write
  # Without --write: dry-run, prints suggested intercept/threshold.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
OUT_DIR = BASE_DIR / 'out'
ORDERS_DIR = OUT_DIR / 'orders'
CONFIG_PATH = BASE_DIR / 'config' / 'policy_overrides.json'


def _load_history(min_days: int = 500) -> pd.DataFrame:
    path = OUT_DIR / 'prices_history.csv'
    if path.exists():
        ph = pd.read_csv(path)
    else:
        from scripts.data_fetching.fetch_ticker_data import ensure_and_load_history_df
        ph = ensure_and_load_history_df(['VNINDEX'], outdir=str(OUT_DIR / 'data'), min_days=min_days, resolution='D')
    ph['Ticker'] = ph['Ticker'].astype(str).str.upper()
    vn = ph[ph['Ticker'] == 'VNINDEX'].copy()
    if vn.empty:
        raise SystemExit('prices_history has no VNINDEX; cannot calibrate')
    vn['Date'] = pd.to_datetime(vn['Date'], errors='coerce')
    vn = vn.dropna(subset=['Date']).sort_values('Date')
    # If history exists but is too short, try a one-shot refill of VNINDEX cache
    if len(vn) < min_days:
        try:
            from scripts.data_fetching.fetch_ticker_data import ensure_and_load_history_df as _ensure_hist
            refill = _ensure_hist(['VNINDEX'], outdir=str(OUT_DIR / 'data'), min_days=max(min_days, 700), resolution='D')
            if not refill.empty:
                refill['Ticker'] = refill['Ticker'].astype(str).str.upper()
                vn2 = refill[refill['Ticker'] == 'VNINDEX'].copy()
                vn2['Date'] = pd.to_datetime(vn2['Date'], errors='coerce')
                vn2 = vn2.dropna(subset=['Date']).sort_values('Date')
                if len(vn2) >= min_days:
                    return vn2
        except Exception as exc:
            raise SystemExit(
                'Failed to refill VNINDEX history via ensure_and_load_history_df; '
                'fix data pipeline or rerun fetch before calibration'
            ) from exc
    return vn


def _ema(series: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(series, errors='coerce').ewm(span=span, adjust=False).mean()


def _atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    h = pd.to_numeric(high, errors='coerce')
    l = pd.to_numeric(low, errors='coerce')
    c = pd.to_numeric(close, errors='coerce')
    prev_close = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


@dataclass
class PolicyRegime:
    intercept: float
    threshold: float
    components: Dict[str, Dict[str, float]]  # name -> {mean,std,weight}


def _load_policy() -> PolicyRegime:
    # Read from out/orders if present (runtime copy), otherwise from config
    src = ORDERS_DIR / 'policy_overrides.json'
    if not src.exists():
        src = CONFIG_PATH
    text = src.read_text(encoding='utf-8')
    import re
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(^|\s)//.*$", "", text, flags=re.M)
    text = re.sub(r"(^|\s)#.*$", "", text, flags=re.M)
    import json
    js = json.loads(text)
    rm = js.get('regime_model') or {}
    if not rm or not rm.get('components'):
        raise SystemExit('policy_overrides has no regime_model.components')
    comps = {}
    for name, conf in rm['components'].items():
        comps[name] = {
            'mean': float(conf.get('mean', 0.0)),
            'std': float(conf.get('std', 1.0)) or 1.0,
            'weight': float(conf.get('weight', 0.0)),
        }
    return PolicyRegime(
        intercept=float(rm.get('intercept', 0.0)),
        threshold=float(rm.get('threshold', 0.5)),
        components=comps,
    )


def _compute_components_df(vn: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(vn['Close'], errors='coerce')
    high = pd.to_numeric(vn.get('High'), errors='coerce') if 'High' in vn.columns else pd.Series(index=vn.index, dtype=float)
    low = pd.to_numeric(vn.get('Low'), errors='coerce') if 'Low' in vn.columns else pd.Series(index=vn.index, dtype=float)
    ret = close.pct_change()
    # Trend strength vs MA200
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    trend_strength = (close - ma200) / ma200
    ma200_slope = (ma200 - ma200.shift(20)) / ma200.shift(20)
    uptrend = ((close > ma200) & (ma200_slope > 0)).astype(float)
    # Momentum 63d
    mom63 = close.pct_change(63)
    mom_ratio = np.maximum(0.0, mom63) / 0.12
    mom_ratio = np.clip(mom_ratio, 0.0, 1.0)
    # Approximate percentile ranks over rolling window
    def _pct_rank(s: pd.Series, win: int = 252) -> pd.Series:
        out = s.copy() * 0.0
        for i in range(len(s)):
            lo = max(0, i - win + 1)
            window = s.iloc[lo:i+1]
            x = s.iloc[i]
            if np.isfinite(x) and window.notna().any():
                out.iloc[i] = (window <= x).mean()
            else:
                out.iloc[i] = np.nan
        return out
    mom_pct = _pct_rank(mom63)
    trend_up = np.clip(np.maximum(trend_strength, 0.0) / 0.10, 0.0, 1.0)
    momentum_norm = 0.5 * mom_pct + 0.5 * mom_ratio
    trend_momentum = 0.5 * trend_up + 0.5 * momentum_norm
    # Drawdown
    roll_max = close.cummax()
    dd = 1.0 - close / roll_max.replace(0, np.nan)
    dd = dd.replace([np.inf, -np.inf], np.nan)
    dd = dd.clip(lower=0.0)
    dd_comp = np.minimum(1.0 - (dd / 0.20), 1.0)
    dd_pct = _pct_rank(dd)
    dd_comp = np.minimum(dd_comp, 1.0 - dd_pct)
    # Volatility annualized (EW std 20)
    vol_ew = ret.ewm(span=20, adjust=False).std()
    vol_ann = vol_ew * np.sqrt(252.0)
    vol_comp = np.clip(1.0 - (vol_ann / 0.45), 0.0, 1.0)
    vol_pct = _pct_rank(vol_ann)
    vol_comp = np.minimum(vol_comp, 1.0 - vol_pct)
    # Index ATR14 percentile (if Hi/Lo available)
    atr_pct = pd.Series(index=vn.index, dtype=float)
    if 'High' in vn.columns and 'Low' in vn.columns:
        atr = _atr_wilder(high, low, close, 14)
        with np.errstate(divide='ignore', invalid='ignore'):
            atr_pct = (atr / close).replace([np.inf, -np.inf], np.nan)
        atr_pct = atr_pct.clip(lower=0.0)
    atr_pctile = _pct_rank(atr_pct.fillna(atr_pct.median()) if atr_pct.notna().any() else pd.Series([0.5]*len(vn)))
    # Smoothed index return over 6 sessions (percent)
    daily_pct = ret * 100.0
    idx_smoothed = daily_pct.rolling(6).mean()
    idx_norm = np.clip(idx_smoothed / 1.5, -1.0, 1.0)
    idx_comp = np.clip(np.maximum(idx_norm, 0.0), 0.0, 1.0)
    out = pd.DataFrame({
        'Date': vn['Date'].values,
        'trend': trend_momentum.values,
        'index_return': idx_comp.values,
        'volatility': vol_comp.values,
        'drawdown': dd_comp.fillna(0.5).values,
        'index_atr_percentile': atr_pctile.fillna(0.5).values,
        # Components we cannot compute reliably here will be proxied later
    })
    return out.dropna().reset_index(drop=True)


def _labels_by_date(vn: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return DataFrame with columns ['Date','y'] aligned to vn['Date'].
    y = 1 if forward return over `horizon` sessions > 0, else 0.
    Last `horizon` rows have no label and are dropped.
    """
    vnx = vn.copy()
    vnx['Date'] = pd.to_datetime(vnx['Date'], errors='coerce')
    close = pd.to_numeric(vnx['Close'], errors='coerce')
    fwd = close.shift(-horizon) / close - 1.0
    y = (fwd > 0.0).astype(int)
    lab = pd.DataFrame({'Date': vnx['Date'], 'y': y})
    lab = lab.dropna(subset=['Date'])
    lab = lab.iloc[:-horizon] if len(lab) > horizon else lab.iloc[0:0]
    return lab.reset_index(drop=True)


def _prepare_design(X: pd.DataFrame, policy: PolicyRegime) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    names: List[str] = []
    Z_cols: List[np.ndarray] = []
    for name, conf in policy.components.items():
        mean = float(conf['mean']); std = float(conf['std']) if float(conf['std']) != 0.0 else 1.0
        if name in X.columns:
            z = (pd.to_numeric(X[name], errors='coerce') - mean) / std
        else:
            # Unknown component: use mean so z=0
            z = pd.Series([0.0] * len(X))
        Z_cols.append(z.to_numpy(dtype=float))
        names.append(name)
    Z = np.column_stack(Z_cols) if Z_cols else np.zeros((len(X), 0))
    w = np.array([policy.components[n]['weight'] for n in names], dtype=float) if names else np.zeros(0)
    return Z, w, names


def _calibrate_intercept(z_lin: np.ndarray, y: np.ndarray) -> float:
    # Find b that minimizes log-loss of sigmoid(z_lin + b) vs y
    # f(b) = sum(log(1+exp(z_i+b)) - y_i*(z_i+b)), f'(b)=sum(sigmoid(z_i+b)-y_i)
    # Use Newton with backtracking
    b = 0.0
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-(z_lin + b)))
        g = (p - y).sum()
        h = (p * (1.0 - p)).sum()
        if h <= 1e-9:
            break
        step = -g / h
        # Backtracking to ensure improvement
        ll_prev = -(np.log1p(np.exp(z_lin + b)) - y * (z_lin + b)).sum()
        alpha = 1.0
        for _ in range(10):
            b_new = b + alpha * step
            ll_new = -(np.log1p(np.exp(z_lin + b_new)) - y * (z_lin + b_new)).sum()
            if ll_new >= ll_prev:
                b = b_new
                break
            alpha *= 0.5
        if abs(step) < 1e-6:
            break
    return float(b)


def _choose_threshold(prob: np.ndarray, y: np.ndarray) -> float:
    # Maximize Youden's J = TPR - FPR across candidate thresholds
    thr_grid = np.linspace(0.2, 0.8, 61)
    best = (0.5, -1.0)
    for t in thr_grid:
        pred = (prob >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        j = tpr - fpr
        if j > best[1]:
            best = (t, j)
    return float(best[0])


def _auc_score(prob: np.ndarray, y: np.ndarray) -> float:
    """Compute ROC-AUC for binary labels using a threshold sweep.

    No sklearn dependency; works with ties.
    """
    if prob.size == 0 or y.size == 0:
        return 0.0
    # Sort by probability descending
    order = np.argsort(-prob)
    p = prob[order]
    t = y[order]
    # Compute TPR/FPR at unique thresholds
    P = float((t == 1).sum())
    N = float((t == 0).sum())
    if P <= 0 or N <= 0:
        return 0.0
    tp = 0.0; fp = 0.0
    auc = 0.0
    prev_fpr = 0.0; prev_tpr = 0.0
    last = None
    for i in range(len(p)):
        if last is None or p[i] != last:
            # Trapezoid between previous and current point
            auc += (prev_fpr - (fp / N)) * (prev_tpr + (tp / P)) * 0.5
            prev_fpr = fp / N
            prev_tpr = tp / P
            last = p[i]
        if t[i] == 1:
            tp += 1.0
        else:
            fp += 1.0
    # Final segment
    auc += (prev_fpr - (fp / N)) * (prev_tpr + (tp / P)) * 0.5
    return float(abs(auc))


def _oos_report(
    Z: np.ndarray,
    w: np.ndarray,
    y: np.ndarray,
    split_frac: float = 0.8,
) -> dict:
    """Train/valid split diagnostics for intercept/threshold calibration.

    - Fit intercept and threshold on the first `split_frac` portion (chronological).
    - Evaluate on the remaining portion to report AUC/accuracy/Youden's J.
    Returns a dict with train/test metrics for audit. Does not write to policy.
    """
    n = int(len(y))
    if n < 50 or Z.shape[0] != n:
        return {}
    k = max(10, int(n * split_frac))
    k = min(max(k, 10), n - 10)
    if k <= 10 or (n - k) <= 10:
        return {}
    z_lin_tr = (Z[:k] @ w) if Z.size else np.zeros(k)
    z_lin_te = (Z[k:] @ w) if Z.size else np.zeros(n - k)
    y_tr = y[:k]
    y_te = y[k:]

    b_tr = _calibrate_intercept(z_lin_tr, y_tr)
    prob_tr = 1.0 / (1.0 + np.exp(-(z_lin_tr + b_tr)))
    thr_tr = _choose_threshold(prob_tr, y_tr)

    prob_te = 1.0 / (1.0 + np.exp(-(z_lin_te + b_tr)))
    pred_te = (prob_te >= thr_tr).astype(int)

    # Metrics
    def _acc(p: np.ndarray, yv: np.ndarray) -> float:
        return float((p == yv).mean()) if len(p) == len(yv) and len(yv) else 0.0

    def _youden(prob: np.ndarray, yv: np.ndarray, thr: float) -> float:
        pred = (prob >= thr).astype(int)
        tp = int(((pred == 1) & (yv == 1)).sum())
        fn = int(((pred == 0) & (yv == 1)).sum())
        fp = int(((pred == 1) & (yv == 0)).sum())
        tn = int(((pred == 0) & (yv == 0)).sum())
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        return float(tpr - fpr)

    out = {
        'n_train': int(k),
        'n_test': int(n - k),
        'train': {
            'auc': _auc_score(prob_tr, y_tr),
            'acc': _acc((prob_tr >= thr_tr).astype(int), y_tr),
            'youden_j': _youden(prob_tr, y_tr, thr_tr),
            'threshold': float(thr_tr),
            'intercept': float(b_tr),
        },
        'test': {
            'auc': _auc_score(prob_te, y_te),
            'acc': _acc(pred_te, y_te),
            'youden_j': _youden(prob_te, y_te, thr_tr),
        },
    }
    return out


def calibrate(horizon: int = 21, write: bool = False) -> tuple[float, float, int, float, list[str]]:
    """Programmatic entry point to calibrate intercept/threshold.

    Returns (intercept, threshold, sample_size, positive_rate, components_used)
    and optionally writes the values back to config if write=True.
    Additionally, computes an 80/20 OOS diagnostic report for audit and writes
    it to out/orders/regime_calibration_report.json when possible.
    """
    vn = _load_history(min_days=500)
    comps = _compute_components_df(vn)
    vn_pos = vn.reset_index(drop=True)
    lab = _labels_by_date(vn_pos, horizon)
    df = comps.merge(lab, on='Date', how='inner')
    if df.empty:
        raise SystemExit('calibrate: insufficient overlap between components and labels')
    y = df['y'].to_numpy(dtype=float)
    comps = df.drop(columns=['y'])

    policy = _load_policy()
    Z, w, names = _prepare_design(comps.drop(columns=['Date']), policy)
    z_lin = (Z @ w) if Z.size else np.zeros(len(comps))
    b = _calibrate_intercept(z_lin, y)
    prob = 1.0 / (1.0 + np.exp(-(z_lin + b)))
    thr = _choose_threshold(prob, y)

    # OOS diagnostics for audit (does not affect written values)
    try:
        report = _oos_report(Z, w, y, split_frac=0.8)
        if report:
            # Also include in-sample stats for context
            report.setdefault('insample', {})
            report['insample']['n'] = int(len(y))
            report['insample']['pos_rate'] = float(y.mean()) if len(y) else 0.0
            report['insample']['auc'] = _auc_score(prob, y)
            report['insample']['threshold'] = float(thr)
            report['insample']['intercept'] = float(b)
            # Write to orders dir if available
            ORDERS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = ORDERS_DIR / 'regime_calibration_report.json'
            import json as _json
            out_path.write_text(_json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception as exc:
        raise SystemExit(
            'Failed to generate or write regime calibration diagnostics; check logs for root cause'
        ) from exc

    pos_rate = float(y.mean()) if len(y) else 0.0
    if write:
        import re, json
        target = ORDERS_DIR / 'policy_overrides.json'
        # Prefer writing to runtime copy; if missing, fall back to config path
        if not target.exists():
            target = CONFIG_PATH
        text = target.read_text(encoding='utf-8')
        text_nc = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        text_nc = re.sub(r"(^|\s)//.*$", "", text_nc, flags=re.M)
        text_nc = re.sub(r"(^|\s)#.*$", "", text_nc, flags=re.M)
        obj = json.loads(text_nc)
        obj.setdefault('regime_model', {})['intercept'] = float(b)
        obj.setdefault('regime_model', {})['threshold'] = float(thr)
        target.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    return float(b), float(thr), int(len(y)), pos_rate, list(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--horizon', type=int, default=21, help='forward-return horizon (sessions) for labeling')
    args = ap.parse_args()

    b, thr, n, pos_rate, names = calibrate(horizon=args.horizon, write=True)
    print('[calibrate] Components used:', ', '.join(names) if names else '(none)')
    print(f"[calibrate] Sample size={n}, positive rate={pos_rate:.2f}")
    print(f"[calibrate] Suggested intercept={b:.4f}, threshold={thr:.3f}")
    print(f"[calibrate] Wrote intercept/threshold -> {CONFIG_PATH}")


if __name__ == '__main__':
    main()
