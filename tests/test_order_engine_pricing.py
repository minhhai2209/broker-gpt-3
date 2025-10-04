import unittest
import pandas as pd

from scripts.order_engine import MarketRegime, pick_limit_price


def make_regime(risk_on: bool):
    return MarketRegime(
        phase='morning',
        in_session=True,
        index_change_pct=0.0,
        breadth_hint=0.0,
        risk_on=risk_on,
        buy_budget_frac=0.1,
        top_sectors=[],
        add_max=5,
        new_max=5,
        weights={'w_trend':0,'w_momo':0,'w_liq':0,'w_vol_guard':0,'w_beta':0,'w_sector':0,'w_sector_sent':0,'w_ticker_sent':0,'w_roe':0,'w_earnings_yield':0},
        thresholds={'base_add':0,'base_new':0,'trim_th':0,'q_add':0.5,'q_new':0.5,'min_liq_norm':0,'near_ceiling_pct':0.98,'tp_pct':0.0,'sl_pct':1.0,'tp_atr_mult':None,'sl_atr_mult':None,'tp_floor_pct':0.0,'sl_floor_pct':0.0,'tp_trim_frac':0.3,'exit_on_ma_break':0},
        sector_bias={},
        ticker_bias={},
        pricing={
            'risk_on_buy': ["Aggr","Bal","Cons","MR","Break"],
            'risk_on_sell': ["Cons","Bal","Break","MR","Aggr"],
            'risk_off_buy': ["Cons","MR","Bal","Aggr","Break"],
            'risk_off_sell': ["MR","Cons","Bal","Aggr","Break"],
            'atr_fallback_buy_mult': 0.25,
            'atr_fallback_sell_mult': 0.25,
        },
        sizing={'softmax_tau':0.6,'add_share':0.6,'new_share':0.4,'min_lot':100,'risk_weighting':'score_softmax','risk_alpha':1.0,'max_pos_frac':0.5,'max_sector_frac':0.8,'reuse_sell_proceeds_frac':0.0,'leftover_redistribute':1,'min_ticket_k':0.0,'risk_blend':1.0,'cov_lookback_days':90,'cov_reg':0.0005,'risk_parity_floor':0.2,'dynamic_caps':{'enable':0,'pos_min':0.10,'pos_max':0.16,'sector_min':0.28,'sector_max':0.40,'blend':1.0,'override_static':0}},
        execution={},
    )


class TestPricing(unittest.TestCase):
    def test_pick_limit_price_buy_prefers_highest_le_price(self):
        snap = pd.Series({'Price': 20.15, 'ATR14': 0.50})
        pre = pd.Series({
            'BandFloor_Tick': 19.00,
            'BandCeiling_Tick': 21.00,
            'Aggr_Buy1_Tick': 20.00,
            'Aggr_Buy1_OutOfBand': 0,
            'Bal_Buy1_Tick': 19.95,
            'Bal_Buy1_OutOfBand': 0,
        })
        met = pd.Series({'TickSizeHOSE_Thousand': 0.05})
        regime = make_regime(risk_on=True)
        lp = pick_limit_price('AAA', 'BUY', snap, pre, met, regime)
        self.assertAlmostEqual(lp, 20.00)

    def test_pick_limit_price_sell_prefers_lowest_ge_price(self):
        snap = pd.Series({'Price': 20.15, 'ATR14': 0.50})
        pre = pd.Series({
            'BandFloor_Tick': 19.00,
            'BandCeiling_Tick': 21.00,
            'Cons_Sell1_Tick': 20.30,
            'Cons_Sell1_OutOfBand': 0,
            'Bal_Sell1_Tick': 20.20,
            'Bal_Sell1_OutOfBand': 0,
        })
        met = pd.Series({'TickSizeHOSE_Thousand': 0.05})
        regime = make_regime(risk_on=True)
        lp = pick_limit_price('AAA', 'SELL', snap, pre, met, regime)
        self.assertAlmostEqual(lp, 20.20)

    def test_pick_limit_price_fallback_buy_rounds_down(self):
        # No presets -> fallback by ATR using metrics ATR14_Pct
        snap = pd.Series({'Price': 20.15})
        pre = None
        met = pd.Series({'TickSizeHOSE_Thousand': 0.05, 'ATR14_Pct': 2.0})  # 2% of 20.15 ~ 0.403
        regime = make_regime(risk_on=True)
        lp = pick_limit_price('AAA', 'BUY', snap, pre, met, regime)
        # price - 0.25*0.403 = 20.0495 -> round down to 20.00
        self.assertAlmostEqual(lp, 20.00)

    def test_pick_limit_price_fallback_sell_rounds_up(self):
        snap = pd.Series({'Price': 20.15})
        pre = None
        met = pd.Series({'TickSizeHOSE_Thousand': 0.05, 'ATR14_Pct': 2.0})
        regime = make_regime(risk_on=True)
        lp = pick_limit_price('AAA', 'SELL', snap, pre, met, regime)
        # price + 0.25*0.403 = 20.2505 -> round up to 20.30
        self.assertAlmostEqual(lp, 20.30)


if __name__ == '__main__':
    unittest.main()
