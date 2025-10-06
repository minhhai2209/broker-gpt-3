import math
import unittest

import numpy as np
import pandas as pd
from unittest.mock import patch

from scripts.portfolio.portfolio_risk import (
    ExpectedReturnInputs,
    compute_cov_matrix,
    compute_expected_returns,
    compute_risk_parity_weights,
)


class TestPortfolioRisk(unittest.TestCase):
    def test_compute_cov_matrix_ledoit_wolf_returns_positive_definite(self):
        dates = pd.date_range("2024-01-01", periods=120, freq="B")
        rng = np.random.default_rng(42)
        data = pd.DataFrame(
            rng.normal(scale=[0.01, 0.015], size=(120, 2)),
            index=dates,
            columns=["AAA", "BBB"],
        )
        cov = compute_cov_matrix(data, reg=1e-4)
        self.assertEqual(cov.shape, (2, 2))
        # Cholesky should succeed for positive definite matrix
        np.linalg.cholesky(cov.to_numpy())

    def test_compute_cov_matrix_manual_fallback(self):
        dates = pd.date_range("2024-01-01", periods=80, freq="B")
        rng = np.random.default_rng(123)
        data = pd.DataFrame(
            rng.normal(scale=[0.02, 0.03], size=(80, 2)),
            index=dates,
            columns=["AAA", "BBB"],
        )
        from scripts.portfolio import portfolio_risk as pr

        with patch("scripts.portfolio.portfolio_risk._ledoit_wolf", None):
            cov = pr.compute_cov_matrix(data, reg=1e-4)
        np.linalg.cholesky(cov.to_numpy())

    def test_compute_expected_returns_adds_score_view(self):
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        base_prices = pd.DataFrame({
            "AAA": np.linspace(100, 120, len(dates)),
            "BBB": np.linspace(50, 45, len(dates)),
        }, index=dates)
        market_path = np.linspace(1000, 1100, len(dates)) + 5 * np.sin(np.linspace(0, 3.14, len(dates)))
        market = pd.Series(market_path, index=dates)
        inputs = ExpectedReturnInputs(
            prices=base_prices,
            market_index=market,
            rf_annual=0.05,
            market_premium_annual=0.08,
            score_view=pd.Series({"AAA": 1.0, "BBB": -1.0}),
            alpha_scale=0.02,
        )
        mu = compute_expected_returns(inputs)
        self.assertIn("AAA", mu.index)
        self.assertGreater(mu.loc["AAA"], mu.loc["BBB"])
        # Expected returns should be within sensible bounds [ -1, 1 )
        self.assertLess(mu.abs().max(), 1.0)

    def test_compute_risk_parity_weights_normalised(self):
        cov = pd.DataFrame(
            [[0.04, 0.01], [0.01, 0.09]],
            index=["AAA", "BBB"],
            columns=["AAA", "BBB"],
        )
        weights = compute_risk_parity_weights(cov)
        self.assertTrue(math.isclose(weights.sum(), 1.0, rel_tol=1e-9))
        self.assertTrue((weights >= 0.0).all())


if __name__ == "__main__":
    unittest.main()
