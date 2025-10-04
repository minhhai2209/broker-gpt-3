"""On-the-fly calibration for mean-variance sizing.

This module evaluates candidate parameter grids (risk_alpha, cov_reg, bl_alpha_scale)
using recent price history so that the order engine can adapt before generating
orders. The calibration is lightweight: walk-forward over the most recent window
with the same covariance/expected-return machinery used in the live allocator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Dict, Any

import numpy as np
import pandas as pd

from .portfolio_risk import (
    ExpectedReturnInputs,
    compute_cov_matrix,
    compute_expected_returns,
    solve_mean_variance_weights,
    normalize_scores,
    TRADING_DAYS_PER_YEAR,
)


@dataclass(frozen=True)
class CalibrationOutcome:
    params: Dict[str, float]
    diagnostics: Dict[str, Any]


def _ensure_list_float(values: Sequence[object], name: str) -> Tuple[float, ...]:
    if not isinstance(values, Sequence) or not values:
        raise SystemExit(f"mean_variance_calibration.{name} must be a non-empty list of numbers")
    out = []
    for v in values:
        try:
            out.append(float(v))
        except Exception as exc:  # pragma: no cover - config error
            raise SystemExit(f"Invalid value in mean_variance_calibration.{name}: {v}") from exc
    return tuple(sorted(set(out)))


def _max_drawdown(returns: np.ndarray) -> float:
    curve = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(curve)
    drawdown = 1.0 - np.divide(curve, peak, out=np.ones_like(curve), where=peak > 0)
    return float(np.max(drawdown))


def calibrate_mean_variance_params(
    prices_history: pd.DataFrame,
    scores: Mapping[str, float],
    tickers: Iterable[str],
    *,
    market_symbol: str,
    sector_by_ticker: Mapping[str, Optional[str]],
    sizing_conf: Mapping[str, object],
    calibration_conf: Mapping[str, object],
) -> CalibrationOutcome:
    enable = int(float(calibration_conf.get('enable', 0) or 0))
    if enable <= 0:
        raise SystemExit("mean_variance_calibration.enable must be 1 to run calibration")

    risk_alpha_grid = _ensure_list_float(calibration_conf.get('risk_alpha', []), 'risk_alpha')
    cov_reg_grid = _ensure_list_float(calibration_conf.get('cov_reg', []), 'cov_reg')
    bl_alpha_grid = _ensure_list_float(calibration_conf.get('bl_alpha_scale', []), 'bl_alpha_scale')

    lookback_days = int(calibration_conf.get('lookback_days', sizing_conf.get('cov_lookback_days', 250)))
    test_horizon_days = int(calibration_conf.get('test_horizon_days', 60))
    min_history_days = int(calibration_conf.get('min_history_days', lookback_days + test_horizon_days + 10))

    tickers = [str(t).upper() for t in tickers if str(t).strip()]
    if not tickers:
        raise SystemExit("mean_variance calibration requires at least one ticker candidate")

    required_cols = {"Date", "Ticker", "Close"}
    if prices_history is None or prices_history.empty:
        raise SystemExit("prices_history is required for mean-variance calibration")
    if not required_cols.issubset(prices_history.columns):
        missing = ", ".join(sorted(required_cols - set(prices_history.columns)))
        raise SystemExit(f"prices_history missing columns for calibration: {missing}")

    symbols = set(tickers)
    symbols.add(market_symbol)
    hist = prices_history[prices_history['Ticker'].isin(symbols)].copy()
    if hist.empty:
        raise SystemExit("No overlapping history for calibration universe")
    hist['Date'] = pd.to_datetime(hist['Date'], errors='coerce')
    hist = hist.dropna(subset=['Date']).sort_values('Date')
    pivot = hist.pivot(index='Date', columns='Ticker', values='Close')
    pivot = pivot.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='all')
    if pivot.shape[0] < min_history_days:
        raise SystemExit(
            f"Not enough price history for calibration (need >= {min_history_days} rows, have {pivot.shape[0]})"
        )
    if market_symbol not in pivot.columns:
        raise SystemExit(f"Market index '{market_symbol}' missing in price history for calibration")

    evaluation_end = pivot.index[-1]
    evaluation_start_idx = max(lookback_days, pivot.index.get_loc(pivot.index[-1]) - test_horizon_days)
    evaluation_indices = range(evaluation_start_idx, len(pivot.index) - 1)
    if not evaluation_indices:
        raise SystemExit("Insufficient evaluation horizon for calibration")

    score_series = pd.Series({str(k).upper(): float(v) for k, v in scores.items()}, dtype=float)
    normalized_scores = normalize_scores(score_series.to_dict()) if not score_series.empty else pd.Series()

    results: list[dict[str, object]] = []

    for risk_alpha in risk_alpha_grid:
        if risk_alpha <= 0:
            raise SystemExit("mean_variance_calibration.risk_alpha must contain positive values")
        for cov_reg in cov_reg_grid:
            if cov_reg < 0:
                raise SystemExit("mean_variance_calibration.cov_reg must be >= 0")
            for bl_alpha_scale in bl_alpha_grid:
                walk_returns: list[float] = []
                active_counts: list[int] = []
                turnovers: list[float] = []
                prev_weights: Optional[pd.Series] = None

                for idx in evaluation_indices:
                    window = pivot.iloc[idx - lookback_days: idx + 1]
                    if window.shape[0] < lookback_days:
                        continue
                    assets = [t for t in tickers if t in window.columns and t != market_symbol]
                    sub_prices = window[assets].dropna(axis=1, how='all').dropna(axis=0, how='any')
                    if sub_prices.shape[1] < 2:
                        continue
                    returns = np.log(sub_prices).diff().dropna(how='any')
                    if returns.shape[0] < max(40, lookback_days // 5):
                        continue
                    try:
                        cov = compute_cov_matrix(returns.tail(lookback_days), reg=cov_reg)
                    except ValueError:
                        continue
                    cov = cov.loc[:, cov.index]
                    market_series = window[market_symbol].reindex(sub_prices.index).dropna()
                    if market_series.shape[0] < returns.shape[0]:
                        continue
                    score_view = normalized_scores.reindex(cov.index).fillna(0.0)
                    mu_inputs = ExpectedReturnInputs(
                        prices=sub_prices[cov.index],
                        market_index=market_series,
                        rf_annual=float(sizing_conf.get('bl_rf_annual', 0.03)),
                        market_premium_annual=float(sizing_conf.get('bl_mkt_prem_annual', 0.07)),
                        score_view=score_view,
                        alpha_scale=bl_alpha_scale,
                    )
                    try:
                        mu_annual = compute_expected_returns(mu_inputs)
                        weights = solve_mean_variance_weights(
                            mu_annual=mu_annual,
                            cov=cov,
                            risk_alpha=risk_alpha,
                            max_pos_frac=float(sizing_conf.get('max_pos_frac', 1.0)),
                            max_sector_frac=float(sizing_conf.get('max_sector_frac', 1.0)),
                            min_names_target=int(float(sizing_conf.get('min_names_target', 0) or 0)),
                            sector_by_ticker=sector_by_ticker,
                        )
                    except Exception:
                        continue
                    next_prices = pivot.iloc[idx + 1]
                    base_prices = window.iloc[-1]
                    aligned = weights.index
                    next_ret = []
                    for ticker in aligned:
                        nxt = next_prices.get(ticker)
                        cur = base_prices.get(ticker)
                        if nxt is None or cur is None or cur <= 0:
                            next_ret.append(0.0)
                        else:
                            next_ret.append(float(nxt) / float(cur) - 1.0)
                    walk_returns.append(float(np.dot(weights.to_numpy(), np.array(next_ret))))
                    active_counts.append(int((weights > 1e-6).sum()))
                    if prev_weights is not None:
                        joined = weights.reindex(prev_weights.index).fillna(0.0)
                        prev_joined = prev_weights.reindex(weights.index).fillna(0.0)
                        turnover = 0.5 * float(np.abs(joined - prev_joined).sum())
                        turnovers.append(turnover)
                    prev_weights = weights

                if not walk_returns:
                    continue
                returns_arr = np.array(walk_returns, dtype=float)
                mean_daily = returns_arr.mean()
                vol_daily = returns_arr.std(ddof=0)
                sharpe = float(mean_daily / vol_daily) if vol_daily > 1e-9 else 0.0
                ann_return = float((1.0 + mean_daily) ** TRADING_DAYS_PER_YEAR - 1.0)
                ann_vol = float(vol_daily * np.sqrt(TRADING_DAYS_PER_YEAR))
                max_dd = _max_drawdown(returns_arr)
                avg_names = float(np.mean(active_counts)) if active_counts else 0.0
                avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0
                objective = sharpe - 0.5 * max_dd - 0.1 * avg_turnover
                results.append({
                    'risk_alpha': risk_alpha,
                    'cov_reg': cov_reg,
                    'bl_alpha_scale': bl_alpha_scale,
                    'objective': objective,
                    'ann_return': ann_return,
                    'ann_vol': ann_vol,
                    'sharpe': sharpe,
                    'max_drawdown': max_dd,
                    'avg_turnover': avg_turnover,
                    'avg_names': avg_names,
                    'observations': len(walk_returns),
                })

    if not results:
        raise SystemExit("Mean-variance calibration produced no valid evaluation results")

    results_sorted = sorted(results, key=lambda x: (x['objective'], -x['max_drawdown']), reverse=True)
    best = results_sorted[0]
    params = {
        'risk_alpha': float(best['risk_alpha']),
        'cov_reg': float(best['cov_reg']),
        'bl_alpha_scale': float(best['bl_alpha_scale']),
    }
    diagnostics = {
        'results': results_sorted[:5],
        'grid_size': len(results),
        'lookback_days': lookback_days,
        'test_horizon_days': test_horizon_days,
        'evaluation_end': str(evaluation_end.date()),
    }
    return CalibrationOutcome(params=params, diagnostics=diagnostics)
