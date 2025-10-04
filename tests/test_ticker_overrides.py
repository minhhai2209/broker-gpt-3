import unittest

from scripts.order_engine import MarketRegime, classify_action


class TestTickerOverrides(unittest.TestCase):
    def _base_regime(self):
        return MarketRegime(
            phase='morning',
            in_session=True,
            index_change_pct=0.0,
            breadth_hint=0.5,
            risk_on=True,
            buy_budget_frac=0.10,
            top_sectors=[],
            add_max=5,
            new_max=5,
            weights={
                'w_trend': 0,'w_momo': 0,'w_mom_ret': 0,'w_liq': 0,'w_vol_guard': 0,
                'w_beta': 0,'w_sector': 0,'w_sector_sent': 0,'w_ticker_sent': 0,
                'w_roe': 0,'w_earnings_yield': 0,'w_rs': 0,
            },
            thresholds={
                'base_add': 0.5,
                'base_new': 0.5,
                'trim_th': -1.0,  # avoid accidental trim
                'q_add': 0.5,
                'q_new': 0.5,
                'min_liq_norm': 0.0,
                'near_ceiling_pct': 0.98,
                'tp_pct': 0.10,
                'sl_pct': 0.10,
                'tp_atr_mult': None,
                'sl_atr_mult': None,
                'tp_floor_pct': 0.0,
                'sl_floor_pct': 0.0,
                'tp_trim_frac': 0.3,
                'exit_on_ma_break': 1,
                'exit_ma_break_score_gate': 0.0,
                'tilt_exit_downgrade_min': 0.05,
                'cooldown_days': 0,
                'exit_ma_break_rsi': 45.0,
                'trim_rsi_below_ma20': 30.0,
                'trim_rsi_macdh_neg': 30.0,
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
                'risk_blend': 1.0,
                'max_pos_frac': 0.5,
                'max_sector_frac': 0.8,
                'reuse_sell_proceeds_frac': 0.0,
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

    def test_exit_on_ma_break_can_be_disabled_per_ticker(self):
        regime = self._base_regime()
        # Features imply MA break with weak RSI but not SL/TP/trim
        feats = {
            'above_ma50': 0.0,
            'above_ma20': 1.0,
            'rsi': 40.0,
            'macdh_pos': 1.0,
            'pnl_pct': 0.0,
        }
        score = 0.10

        # Without overrides -> exit
        action_default = classify_action(True, score, feats, regime)
        self.assertEqual(action_default, 'exit')

        # With override disabling MA-break exit -> hold
        override = { 'exit_on_ma_break': 0 }
        action_override = classify_action(True, score, feats, regime, thresholds_override=override)
        self.assertEqual(action_override, 'hold')


if __name__ == '__main__':
    unittest.main()
