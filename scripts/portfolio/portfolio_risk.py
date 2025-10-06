"""Portfolio risk and allocation utilities for the order engine.

All functions here enforce explicit validation and fail fast when inputs are
incomplete. The engine should surface actionable diagnostics instead of falling
back to implicit defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Dict

import numpy as np
import pandas as pd

try:  # Prefer scikit-learn's implementation when available
    from sklearn.covariance import ledoit_wolf as _ledoit_wolf
except Exception:  # pragma: no cover - environment without sklearn
    _ledoit_wolf = None

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class ExpectedReturnInputs:
    prices: pd.DataFrame  # Wide prices (Date index, tickers columns)
    market_index: pd.Series  # Market benchmark prices aligned with prices
    rf_annual: float
    market_premium_annual: float
    score_view: pd.Series  # Conviction scores indexed by ticker
    alpha_scale: float


def normalize_scores(scores: Mapping[str, float], clip_z: float = 3.0) -> pd.Series:
    """Return scores scaled to [-1, 1] via z-score → logistic transform.

    Raising when the mapping is empty keeps upstream logic honest: we prefer the
    caller to guard against missing scores rather than assume neutral views.
    """

    if not scores:
        raise ValueError("normalize_scores requires at least one score")
    series = pd.Series(scores, dtype=float)
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        raise ValueError("normalize_scores received only null/inf scores")
    # Z-score with population std (ddof=0) for stability on short windows
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    if std <= 1e-12:
        # All scores identical → neutral views
        return pd.Series(0.0, index=series.index)
    z = (series - mean) / std
    z = z.clip(lower=-abs(clip_z), upper=abs(clip_z))
    # Logistic squashing mapped to [-1, 1]
    logistic = 1.0 / (1.0 + np.exp(-z))
    scaled = 2.0 * logistic - 1.0
    return pd.Series(scaled, index=series.index)


def _sanitize_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        raise ValueError("DataFrame is None")
    if df.empty:
        raise ValueError("DataFrame is empty")
    numeric = df.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna(axis=0, how="any")
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.empty:
        raise ValueError("No valid numeric data after cleaning")
    return numeric


def _ledoit_wolf_fallback(data: np.ndarray) -> np.ndarray:
    n_samples, n_features = data.shape
    if n_samples <= 1:
        raise ValueError("Need at least two observations for Ledoit-Wolf shrinkage")
    centered = data - data.mean(axis=0, keepdims=True)
    emp_cov = (centered.T @ centered) / float(n_samples)
    mu = np.trace(emp_cov) / float(n_features)
    prior = mu * np.identity(n_features)
    centered_sq = centered ** 2
    phi_matrix = (centered_sq.T @ centered_sq) / float(n_samples) - emp_cov ** 2
    phi = float(np.sum(phi_matrix))
    gamma = float(np.sum((emp_cov - prior) ** 2))
    if gamma <= 1e-24:
        shrinkage = 0.0
    else:
        shrinkage = phi / gamma / float(n_samples)
        shrinkage = min(1.0, max(0.0, shrinkage))
    return shrinkage * prior + (1.0 - shrinkage) * emp_cov


def compute_cov_matrix(
    returns_df: pd.DataFrame,
    shrink: str = "ledoit_wolf",
    reg: float = 1e-4,
) -> pd.DataFrame:
    """Estimate a stable covariance matrix for daily (log) returns.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Wide returns matrix (Date index × tickers). Each column is expected to be
        numeric with no missing values. Upstream should pass an aligned window
        (e.g., last 250 trading days) to respect the configured lookback.
    shrink : str
        Shrinkage method. "ledoit_wolf" uses the analytic Ledoit–Wolf estimator
        (preferred). Any other value falls back to the sample covariance matrix.
    reg : float
        Diagonal ridge term to guarantee positive definiteness. Must be ≥ 0.
    """

    if float(reg) < 0:
        raise ValueError("reg (diagonal regularisation) must be >= 0")
    clean = _sanitize_numeric_frame(returns_df)
    if clean.shape[0] < 30:
        raise ValueError("Insufficient observations for covariance (need >= 30)")
    cols = list(clean.columns)
    data = clean.to_numpy(dtype=float)

    if shrink.lower() == "ledoit_wolf":
        if _ledoit_wolf is not None:
            cov_matrix, _ = _ledoit_wolf(data, assume_centered=True)
        else:
            cov_matrix = _ledoit_wolf_fallback(data)
        cov = pd.DataFrame(cov_matrix, index=cols, columns=cols)
    else:
        cov = clean.cov()
    cov = cov.astype(float)
    # Ensure symmetry before applying ridge to guard numerical noise
    cov = (cov + cov.T) * 0.5
    if reg > 0:
        idx = np.diag_indices_from(cov.values)
        cov.values[idx] += float(reg)
    # Validate positive definiteness via Cholesky (raises if singular)
    try:
        np.linalg.cholesky(cov.values)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - rare, but fail fast
        raise ValueError(f"Covariance matrix not positive definite: {exc}") from exc
    return cov


def compute_expected_returns(inputs: ExpectedReturnInputs) -> pd.Series:
    """Blend CAPM equilibrium returns with score-based views (Black–Litterman).

    All rates are handled in simple-return space and annualised. The caller is
    responsible for providing prices aligned to the desired lookback.
    """

    prices = inputs.prices.copy()
    market = inputs.market_index.copy()
    if prices is None or prices.empty:
        raise ValueError("prices DataFrame is empty")
    if market is None or market.empty:
        raise ValueError("market_index series is empty")
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    market.index = pd.to_datetime(market.index, errors="coerce")
    prices = prices.sort_index().dropna(axis=0, how="all")
    market = market.sort_index().dropna()
    if prices.empty or market.empty:
        raise ValueError("Insufficient price history for expected return estimation")

    # Align window
    common = prices.index.intersection(market.index)
    if len(common) < 60:
        raise ValueError("Need at least 60 observations for CAPM regression")
    prices = prices.loc[common]
    market = market.loc[common]
    prices = prices.dropna(axis=1, how="all")
    if prices.shape[1] == 0:
        raise ValueError("All tickers dropped after aligning prices with market index")

    # Compute simple daily returns; drop any day with missing data to avoid bias
    asset_returns = prices.pct_change().dropna(how="any")
    market_returns = market.pct_change().reindex(asset_returns.index).dropna()
    asset_returns = asset_returns.loc[market_returns.index]
    if len(asset_returns) < 40:
        raise ValueError("Need at least 40 aligned return observations for CAPM")

    rf_daily = np.power(1.0 + float(inputs.rf_annual), 1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    market_excess = market_returns - rf_daily
    if np.allclose(market_excess.var(ddof=0), 0.0):
        raise ValueError("Market excess returns variance is zero; cannot estimate beta")

    normalized_scores = normalize_scores(inputs.score_view.to_dict())
    normalized_scores = normalized_scores.reindex(asset_returns.columns).fillna(0.0)

    exp_returns = {}
    mkt_excess_values = market_excess.to_numpy()
    denom = float((mkt_excess_values ** 2).sum())
    if denom <= 1e-12:
        raise ValueError("Market excess returns have insufficient variability")

    for ticker in asset_returns.columns:
        ri = asset_returns[ticker].to_numpy()
        excess = ri - rf_daily
        beta = float(np.dot(excess, mkt_excess_values) / denom)
        mu_capm = float(inputs.rf_annual + beta * inputs.market_premium_annual)
        alpha = float(inputs.alpha_scale) * float(normalized_scores.get(ticker, 0.0))
        exp = mu_capm + alpha
        # Guard against pathological inputs (e.g., <-100%)
        exp = max(exp, -0.99)
        exp_returns[ticker] = exp

    return pd.Series(exp_returns, index=asset_returns.columns, dtype=float)


def solve_mean_variance_weights(
    mu_annual: pd.Series,
    cov: pd.DataFrame,
    *,
    risk_alpha: float,
    max_pos_frac: float,
    max_sector_frac: float,
    min_names_target: int,
    sector_by_ticker: Mapping[str, Optional[str]],
) -> pd.Series:
    """Solve mean-variance allocation with position & sector caps.

    Parameters
    ----------
    mu_annual : pd.Series
        Annualized expected returns indexed by ticker.
    cov : pd.DataFrame
        Covariance matrix of daily log returns.
    risk_alpha : float
        Risk-aversion parameter (λ). Higher → more risk-averse.
    max_pos_frac : float
        Maximum fraction per ticker (0..1). <=0 means no cap.
    max_sector_frac : float
        Maximum fraction per sector (0..1). <=0 means no cap.
    min_names_target : int
        Minimum number of names desired (used to tighten position cap).
    sector_by_ticker : Mapping[str, Optional[str]]
        Map from ticker to sector label; missing sectors skip sector cap for that ticker.
    """

    if cov is None or cov.empty:
        raise ValueError("Covariance matrix is empty in mean-variance solver")
    tickers = list(cov.index)
    if not tickers:
        raise ValueError("No tickers provided to mean-variance solver")
    mu = mu_annual.reindex(tickers)
    if mu.isna().all():
        raise ValueError("Expected returns contain only NaN")
    if mu.isna().any():
        mu = mu.fillna(mu.mean())
    risk_aversion = max(float(risk_alpha), 1e-6)
    mu_daily = np.expm1(np.log1p(mu.to_numpy(dtype=float)) / TRADING_DAYS_PER_YEAR)

    cov_values = cov.loc[tickers, tickers].to_numpy(dtype=float)
    try:
        weights = np.linalg.solve(cov_values, mu_daily / risk_aversion)
    except np.linalg.LinAlgError:
        ridge = cov_values + np.eye(len(tickers)) * 1e-6
        weights = np.linalg.solve(ridge, mu_daily / risk_aversion)

    weights = np.clip(weights, 0.0, None)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()

    effective_cap = float(max_pos_frac)
    if min_names_target and min_names_target > 0:
        effective_cap = min(effective_cap, 1.0 / float(min_names_target)) if effective_cap > 0 else 1.0 / float(min_names_target)
    if effective_cap <= 0:
        effective_cap = 1.0

    def _apply_weight_cap(vec: np.ndarray, cap: float) -> np.ndarray:
        if cap >= 1.0:
            return vec
        out = vec.copy()
        tol = 1e-9
        for _ in range(5 * len(out)):
            overflow = out > cap + tol
            if not overflow.any():
                break
            excess = out[overflow] - cap
            out[overflow] = cap
            remaining = ~overflow
            rem_sum = out[remaining].sum()
            if rem_sum <= tol:
                break
            out[remaining] += excess.sum() * (out[remaining] / rem_sum)
            out = np.clip(out, 0.0, None)
            total = out.sum()
            if total > tol:
                out /= total
        if out.sum() <= tol:
            out = vec / vec.sum()
        return out

    weights = _apply_weight_cap(weights, effective_cap)

    def _apply_sector_cap(vec: np.ndarray, cap: float) -> np.ndarray:
        if cap <= 0.0 or cap >= 1.0:
            return vec
        out = vec.copy()
        labels = list(tickers)
        tol = 1e-9
        for _ in range(10 * len(out)):
            sector_totals: Dict[str, float] = {}
            for idx, ticker in enumerate(labels):
                sector = sector_by_ticker.get(str(ticker)) if sector_by_ticker else None
                if not sector:
                    continue
                sector_totals[sector] = sector_totals.get(sector, 0.0) + float(out[idx])
            violators = {sec: tot for sec, tot in sector_totals.items() if tot > cap + tol}
            if not violators:
                break
            for sec, tot in violators.items():
                idxs = [i for i, ticker in enumerate(labels) if sector_by_ticker.get(str(ticker)) == sec]
                if not idxs:
                    continue
                scale = cap / max(tot, tol)
                out[idxs] = out[idxs] * scale
            out = np.clip(out, 0.0, None)
            total = out.sum()
            if total <= tol:
                break
            out /= total
        return out

    weights = _apply_sector_cap(weights, float(max_sector_frac))
    weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    if total <= 1e-9:
        raise ValueError("Mean-variance weights collapsed to zero")
    weights /= total
    return pd.Series(weights, index=tickers, dtype=float)


def compute_risk_parity_weights(
    cov: pd.DataFrame,
    *,
    max_iter: int = 500,
    tol: float = 1e-5,
) -> pd.Series:
    """Solve for risk-parity weights using multiplicative updates (Spinu, 2013).

    Returns
    -------
    pd.Series
        Normalised weights summing to 1 across the provided covariance matrix.
    """

    if cov is None or cov.empty:
        raise ValueError("Covariance matrix is empty")
    matrix = cov.to_numpy(dtype=float)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Covariance matrix must be square")
    n = matrix.shape[0]
    weights = np.ones(n, dtype=float) / n
    eps = 1e-12

    for _ in range(max_iter):
        portfolio_var = float(weights @ matrix @ weights)
        if portfolio_var <= eps:
            raise ValueError("Portfolio variance collapsed during risk parity iteration")
        mrc = matrix @ weights  # marginal risk contributions
        rc = weights * mrc
        target = portfolio_var / n
        if np.all(np.abs(rc - target) <= tol * target + eps):
            break
        update = target / np.clip(rc, eps, None)
        weights = weights * update
        weights = np.clip(weights, 0.0, None)
        total = weights.sum()
        if total <= eps:
            raise ValueError("Risk parity weights degenerated to zero")
        weights /= total
    else:  # pragma: no cover - convergence issues should be surfaced clearly
        raise ValueError("Risk parity solver failed to converge within max_iter")

    return pd.Series(weights, index=cov.index, dtype=float)
