import unittest
import pandas as pd
from scripts.orders.order_engine import pick_limit_price, MarketRegime


class TestPricingPriorityByRegime(unittest.TestCase):
    def _regime(self, risk_on: bool, pricing: dict) -> MarketRegime:
        # Construct minimal regime object for pricing tests
        return MarketRegime(
            phase='morning', in_session=True,
            index_change_pct=0.0, breadth_hint=0.5, risk_on=risk_on,
            buy_budget_frac=0.1, top_sectors=[], add_max=5, new_max=5,
            weights={}, thresholds={}, sector_bias={}, ticker_bias={}, pricing=pricing, sizing={}, execution={}
        )

    def test_risk_on_buy_pref_order(self):
        snap = pd.Series({'Ticker': 'AAA', 'Price': 99.5})
        pre = pd.Series({
            'Ticker': 'AAA',
            'Aggr_Buy1_Tick': 100.0,
            'Bal_Buy1_Tick': 99.0,
            'Cons_Buy1_Tick': 98.0,
        })
        met = pd.Series({'Ticker': 'AAA'})
        pricing = {
            'risk_on_buy': ['Aggr','Bal','Cons','MR','Break'],
            'risk_on_sell': ['Cons','Bal','Break','MR','Aggr'],
            'risk_off_buy': ['Cons','MR','Bal','Aggr','Break'],
            'risk_off_sell': ['MR','Cons','Bal','Aggr','Break'],
            'atr_fallback_buy_mult': 0.25,
            'atr_fallback_sell_mult': 0.25,
        }
        reg = self._regime(True, pricing)
        lp = pick_limit_price('AAA', 'BUY', snap, pre, met, reg)
        self.assertAlmostEqual(lp, 99.0, places=2)

    def test_risk_off_sell_pref_order(self):
        snap = pd.Series({'Ticker': 'AAA', 'Price': 100.2})
        pre = pd.Series({
            'Ticker': 'AAA',
            'MR_Sell1_Tick': 101.0,
            'Cons_Sell1_Tick': 100.5,
            'Bal_Sell1_Tick': 102.0,
        })
        met = pd.Series({'Ticker': 'AAA'})
        pricing = {
            'risk_on_buy': ['Aggr','Bal','Cons','MR','Break'],
            'risk_on_sell': ['Cons','Bal','Break','MR','Aggr'],
            'risk_off_buy': ['Cons','MR','Bal','Aggr','Break'],
            'risk_off_sell': ['MR','Cons','Bal','Aggr','Break'],
            'atr_fallback_buy_mult': 0.25,
            'atr_fallback_sell_mult': 0.25,
        }
        reg = self._regime(False, pricing)
        lp = pick_limit_price('AAA', 'SELL', snap, pre, met, reg)
        self.assertAlmostEqual(lp, 100.5, places=2)

    def test_fallback_rounding_direction(self):
        # No valid candidates -> fallback by ATR; check rounding down/up to tick
        snap = pd.Series({'Ticker': 'AAA', 'Price': 100.0})
        pre = pd.Series({'Ticker': 'AAA', 'ATR14': 1.23})  # ATR in thousand
        met = pd.Series({'Ticker': 'AAA'})
        pricing = {
            'risk_on_buy': ['Aggr'], 'risk_on_sell': ['Cons'],
            'risk_off_buy': ['Cons'], 'risk_off_sell': ['MR'],
            'atr_fallback_buy_mult': 0.25,
            'atr_fallback_sell_mult': 0.25,
        }
        reg = self._regime(True, pricing)
        # BUY: 100 - 0.25*1.23 = 99.6925 -> floor to 99.60 (tick 0.10)
        buy_lp = pick_limit_price('AAA', 'BUY', snap, pre, met, reg)
        self.assertAlmostEqual(buy_lp, 99.60, places=2)
        # SELL: 100 + 0.25*1.23 = 100.3075 -> ceil to 100.40
        sell_lp = pick_limit_price('AAA', 'SELL', snap, pre, met, reg)
        self.assertAlmostEqual(sell_lp, 100.40, places=2)
