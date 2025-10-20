import unittest
from types import SimpleNamespace

from scripts.orders.orders_io import _apply_buy_ttl_policy


class TestOrdersTTLPolicy(unittest.TestCase):
    def _base_regime(self, **updates):
        attrs = {
            'neutral_state': {'guard_flags': {'guard_new': False, 'atr_hard': False, 'vol_hard': False, 'global_hard': False}},
            'index_change_pct': 0.01,
            'trend_strength': 0.02,
            'breadth_hint': 0.65,
            'market_score': 0.55,
            'risk_on_probability': 0.7,
            'index_atr_percentile': 0.30,
            'index_vol_annualized': 0.20,
            'epu_us_percentile': 0.10,
            'dxy_percentile': 0.10,
            'spx_drawdown_pct': 0.05,
        }
        attrs.update(updates)
        return SimpleNamespace(**attrs)

    def _mf_conf(self, **updates):
        conf = {
            'risk_off_index_drop_pct': 0.5,
            'risk_off_trend_floor': -0.02,
            'risk_off_breadth_floor': 0.30,
            'breadth_relax_margin': 0.0,
            'market_score_hard_floor': 0.25,
            'index_atr_soft_pct': 0.40,
            'index_atr_hard_pct': 0.90,
            'vol_ann_hard_ceiling': 0.80,
            'guard_behavior': 'scale_only',
        }
        conf.update(updates)
        return conf

    def test_buy_ttl_respects_floor_when_market_calm(self):
        regime = self._base_regime()
        mf_conf = self._mf_conf()
        orders_ui = {'buy_ttl_floor_minutes': 90, 'buy_ttl_reversal_minutes': 15}
        ttl = _apply_buy_ttl_policy(
            12,
            side='BUY',
            regime=regime,
            orders_ui=orders_ui,
            mf_conf=mf_conf,
            idx_atr_pctile=0.30,
        )
        self.assertEqual(ttl, 90)

    def test_buy_ttl_switches_to_reversal_minutes_under_guard(self):
        regime = self._base_regime(neutral_state={'guard_flags': {'guard_new': True}})
        mf_conf = self._mf_conf()
        orders_ui = {'buy_ttl_floor_minutes': 90, 'buy_ttl_reversal_minutes': 18}
        ttl = _apply_buy_ttl_policy(
            60,
            side='BUY',
            regime=regime,
            orders_ui=orders_ui,
            mf_conf=mf_conf,
            idx_atr_pctile=0.95,
        )
        self.assertEqual(ttl, 18)


if __name__ == '__main__':
    unittest.main()
