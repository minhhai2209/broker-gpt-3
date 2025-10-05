import pandas as pd

from backtest.engine_wrapper import SimulatedOrder
from backtest.fill_sim import FillResult
from backtest.portfolio import FeeConfig, Portfolio


def test_portfolio_apply_fill_updates_positions_and_cash():
    portfolio = Portfolio(initial_cash=1_000_000, positions=None, fee_config=FeeConfig())
    order = SimulatedOrder(symbol="AAA", side="BUY", type="LIMIT", qty=100, limit_price=10_000)
    fill = FillResult(order=order, executed_qty=100, executed_price=10_000, status="FILLED", notes={})
    portfolio.apply_fills(pd.Timestamp("2024-01-02").date(), [fill])
    state = portfolio.mark_to_market(pd.Timestamp("2024-01-02").date(), pd.DataFrame([
        {"symbol": "AAA", "close": 10_500}
    ]))
    assert "AAA" in portfolio.positions
    assert portfolio.positions["AAA"].quantity == 100
    assert state.nav > 0
    assert portfolio.cash == 1_000_000 - 100 * 10_000
