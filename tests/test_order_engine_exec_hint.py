import unittest
import pandas as pd

from scripts.orders.order_engine import MarketRegime, _suggest_execution_window


def make_regime(**overrides):
    regime = MarketRegime(
        phase='morning',
        in_session=True,
        index_change_pct=0.0,
        breadth_hint=0.5,
        risk_on=False,
        buy_budget_frac=0.1,
        top_sectors=[],
        add_max=5,
        new_max=5,
        weights={
            'w_trend': 0,
            'w_momo': 0,
            'w_liq': 0,
            'w_vol_guard': 0,
            'w_beta': 0,
            'w_sector': 0,
            'w_sector_sent': 0,
            'w_ticker_sent': 0,
            'w_roe': 0,
            'w_earnings_yield': 0,
        },
        thresholds={
            'base_add': 0,
            'base_new': 0,
            'trim_th': 0,
            'q_add': 0.5,
            'q_new': 0.5,
            'min_liq_norm': 0,
            'near_ceiling_pct': 0.98,
            'tp_pct': 0.0,
            'sl_pct': 1.0,
            'tp_atr_mult': None,
            'sl_atr_mult': None,
            'tp_floor_pct': 0.0,
            'sl_floor_pct': 0.0,
            'tp_trim_frac': 0.3,
            'exit_on_ma_break': 0,
        },
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
        sizing={
            'softmax_tau': 0.6,
            'add_share': 0.6,
            'new_share': 0.4,
            'min_lot': 100,
            'risk_weighting': 'score_softmax',
            'risk_alpha': 1.0,
            'max_pos_frac': 0.5,
            'max_sector_frac': 0.8,
            'reuse_sell_proceeds_frac': 0.0,
            'leftover_redistribute': 1,
            'min_ticket_k': 0.0,
            'risk_blend': 1.0,
            'cov_lookback_days': 90,
            'cov_reg': 0.0005,
            'risk_parity_floor': 0.2,
            'dynamic_caps': {
                'enable': 0,
                'pos_min': 0.10,
                'pos_max': 0.16,
                'sector_min': 0.28,
                'sector_max': 0.40,
                'blend': 1.0,
                'override_static': 0,
            },
        },
        execution={},
    )
    for key, value in overrides.items():
        setattr(regime, key, value)
    return regime


class TestExecutionHint(unittest.TestCase):
    def test_hint_recognises_strong_market(self):
        regime = make_regime(
            risk_on=True,
            index_change_pct=1.20,
            breadth_hint=0.62,
            risk_on_probability=0.72,
            trend_strength=0.03,
            model_components={'breadth_long': 0.60},
            index_atr14_pct=0.015,
        )
        session = pd.DataFrame([{'SessionPhase': 'morning'}])
        hint = _suggest_execution_window(regime, session, 'morning')
        self.assertIn('Thị trường đang khỏe', hint)
        self.assertIn('VNINDEX +1.20%', hint)
        self.assertIn('Breadth>MA50 62%', hint)

    def test_hint_calls_out_strong_sell_pressure(self):
        regime = make_regime(
            risk_on=False,
            index_change_pct=-1.50,
            breadth_hint=0.40,
            risk_on_probability=0.35,
            trend_strength=-0.02,
            model_components={'breadth_long': 0.40},
            index_atr14_pct=0.015,
        )
        session = pd.DataFrame([{'SessionPhase': 'afternoon'}])
        hint = _suggest_execution_window(regime, session, 'afternoon')
        self.assertIn('Thị trường đang bị bán mạnh', hint)
        self.assertIn('VNINDEX -1.50%', hint)
        self.assertIn('Phiên chiều', hint)

    def test_hint_identifies_sideways_market(self):
        regime = make_regime(
            risk_on=False,
            index_change_pct=0.10,
            breadth_hint=0.50,
            risk_on_probability=0.48,
            trend_strength=0.0,
            model_components={'breadth_long': 0.50},
            index_atr14_pct=0.005,
        )
        session = pd.DataFrame([{'SessionPhase': 'morning'}])
        hint = _suggest_execution_window(regime, session, 'morning')
        self.assertIn('Thị trường đi ngang', hint)
        self.assertIn('VNINDEX +0.10%', hint)


if __name__ == '__main__':
    unittest.main()
