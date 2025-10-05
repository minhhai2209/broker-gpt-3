import pandas as pd

from backtest.engine_wrapper import SimulatedOrder
from backtest.fill_sim import execute_orders


def _make_frame(low: float, high: float, close: float, volume: float = 10000):
    return pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        ]
    )


def test_limit_buy_fill():
    order = SimulatedOrder(symbol="AAA", side="BUY", type="LIMIT", qty=1000, limit_price=10_000)
    frame = _make_frame(low=9_800, high=10_500, close=10_200)
    fills, residual = execute_orders(
        pd.Timestamp("2024-01-02").date(),
        [order],
        frame,
        {"type": "touch_full"},
        {"type": "none"},
    )
    assert len(fills) == 1
    assert fills[0].executed_qty == 1000
    assert not residual


def test_participation_cap():
    order = SimulatedOrder(symbol="AAA", side="BUY", type="LIMIT", qty=5000, limit_price=10_000)
    frame = _make_frame(low=9_700, high=10_400, close=10_100, volume=20_000)
    fills, residual = execute_orders(
        pd.Timestamp("2024-01-02").date(),
        [order],
        frame,
        {"type": "participation", "participation_cap": 0.1},
        {"type": "none"},
        lot_size=100,
    )
    assert len(fills) == 1
    assert fills[0].executed_qty <= order.qty
    assert fills[0].executed_qty == 2000  # 10% participation of 20k volume
    assert residual  # remaining quantity should stay open
