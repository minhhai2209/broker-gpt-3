import unittest

from scripts.orders.order_engine import MarketRegime, build_trade_suggestions


def make_regime(**overrides):
    regime = MarketRegime(
        phase='morning',
        in_session=True,
        index_change_pct=0.0,
        breadth_hint=0.50,
        risk_on=True,
        buy_budget_frac=0.12,
        top_sectors=['Financials','Materials'],
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
            'w_rs': 0,
        },
        thresholds={
            'base_add': 0,
            'base_new': 0,
            'trim_th': 0,
            'q_add': 0.5,
            'q_new': 0.5,
            'min_liq_norm': 0,
            'near_ceiling_pct': 0.98,
            'tp_pct': 0.05,
            'sl_pct': 0.03,
            'tp_atr_mult': None,
            'sl_atr_mult': None,
            'tp_floor_pct': 0.0,
            'sl_floor_pct': 0.0,
            'tp_trim_frac': 0.3,
            'exit_on_ma_break': 0,
            'cooldown_days': 0,
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
            'allocation_model': 'softmax',
            'risk_per_trade_frac': 0.0,
            'default_stop_atr_mult': 0.0,
        },
        execution={},
    )
    # Provide market_filter defaults used by suggestions
    regime.market_filter = {
        'risk_off_index_drop_pct': 0.5,
        'risk_off_trend_floor': 0.0,
        'risk_off_breadth_floor': 0.4,
        'market_score_soft_floor': 0.6,
        'market_score_hard_floor': 0.3,
        'leader_min_rsi': 50.0,
        'leader_min_mom_norm': 0.5,
        'leader_require_ma20': 1,
        'leader_require_ma50': 0,
        'leader_max': 10,
        'risk_off_drawdown_floor': 0.4,
        'index_atr_soft_pct': 0.8,
        'index_atr_hard_pct': 0.95,
    }
    for k, v in overrides.items():
        setattr(regime, k, v)
    return regime


class TestTradeSuggestions(unittest.TestCase):
    def test_core_sections_present(self):
        actions = {
            'AAA': 'new', 'BBB': 'add', 'CCC': 'trim', 'DDD': 'exit', 'EEE': 'hold', 'FFF': 'take_profit'
        }
        scores = {'AAA': 0.82, 'BBB': 0.76, 'CCC': 0.10, 'DDD': 0.05, 'FFF': 0.30}
        feats = {}
        regime = make_regime(risk_on=True, index_atr_percentile=0.70, buy_budget_frac_effective=0.10)

        lines = build_trade_suggestions(actions, scores, feats, regime, top_n=2)
        text = "\n".join(lines)
        self.assertIn("Ưu tiên mua", text)
        self.assertIn("AAA", text)
        self.assertIn("BBB", text)
        self.assertIn("Giảm tỷ trọng/Chốt lời", text)
        self.assertIn("CCC", text)
        self.assertIn("FFF", text)
        self.assertIn("Dừng lỗ/Thoát", text)
        self.assertIn("DDD", text)
        self.assertIn("Ngành nên ưu tiên", text)
        self.assertIn("~10% NAV", text)

    def test_volatility_caution_when_atr_high(self):
        actions = {'AAA': 'new'}
        scores = {'AAA': 0.9}
        feats = {}
        regime = make_regime(risk_on=True, index_atr_percentile=0.96)
        lines = build_trade_suggestions(actions, scores, feats, regime)
        text = "\n".join(lines)
        self.assertIn("Cảnh báo biến động", text)
        self.assertIn("ATR", text)


if __name__ == '__main__':
    unittest.main()

