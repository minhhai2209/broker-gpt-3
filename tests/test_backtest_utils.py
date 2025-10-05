from backtest import utils


def test_tick_size_schedule():
    assert utils.tick_size("AAA", 5_000) == 10.0
    assert utils.tick_size("AAA", 20_000) == 50.0
    assert utils.tick_size("AAA", 100_000) == 100.0


def test_expand_params_grid():
    grid = {"a": [1, 2], "b": ["x"]}
    combos = utils.expand_params_grid(grid)
    assert len(combos) == 2
    normalized = {frozenset(c.items()) for c in combos}
    assert normalized == {
        frozenset({("a", 1), ("b", "x")}),
        frozenset({("a", 2), ("b", "x")}),
    }


def test_apply_slippage_bps():
    price = utils.apply_slippage(100.0, "BUY", utils.SlippageModel(type="bps", value=50), tick=1.0)
    assert price == 100.5
    price_sell = utils.apply_slippage(100.0, "SELL", utils.SlippageModel(type="bps", value=50), tick=1.0)
    assert price_sell == 99.5
