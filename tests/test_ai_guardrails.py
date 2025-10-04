import math
from datetime import date

from scripts.ai.guardrails import enforce_guardrails


def _state_entry(value: float, expiry: str) -> dict:
    return {'value': value, 'last_updated': '2025-10-01', 'expiry': expiry}


def test_buy_budget_bounds_and_rate_limit():
    baseline = {'buy_budget_frac': 0.10}
    overrides = {'buy_budget_frac': 0.5}
    sanitized, state = enforce_guardrails(
        overrides,
        baseline=baseline,
        metrics_count=30,
        state={},
        today=date(2025, 10, 2),
    )
    assert math.isclose(sanitized['buy_budget_frac'], 0.18, rel_tol=1e-6)
    assert math.isclose(state['buy_budget_frac']['value'], 0.18, rel_tol=1e-6)


def test_buy_budget_rate_limit():
    baseline = {'buy_budget_frac': 0.10}
    overrides = {'buy_budget_frac': 0.18}
    state = {'buy_budget_frac': _state_entry(0.10, '2025-10-05')}
    sanitized, new_state = enforce_guardrails(
        overrides,
        baseline=baseline,
        metrics_count=30,
        state=state,
        today=date(2025, 10, 3),
    )
    assert math.isclose(sanitized['buy_budget_frac'], 0.14, rel_tol=1e-6)
    assert math.isclose(new_state['buy_budget_frac']['value'], 0.14, rel_tol=1e-6)


def test_buy_budget_ttl_expired_reverts():
    baseline = {'buy_budget_frac': 0.10}
    overrides = {}
    state = {'buy_budget_frac': _state_entry(0.14, '2025-10-01')}
    sanitized, new_state = enforce_guardrails(
        overrides,
        baseline=baseline,
        metrics_count=30,
        state=state,
        today=date(2025, 10, 5),
    )
    assert 'buy_budget_frac' not in sanitized
    assert 'buy_budget_frac' not in new_state


def test_add_max_clamped_to_bound_and_rate_limit():
    baseline = {'add_max': 4}
    overrides = {'add_max': 9}
    state = {'add_max': _state_entry(4, '2025-10-05')}
    sanitized, new_state = enforce_guardrails(
        overrides,
        baseline=baseline,
        metrics_count=30,
        state=state,
        today=date(2025, 10, 3),
    )
    # Metrics count 30 => max ceil(30/6)=5, rate limit allows +2 from prev (4->6) but cap to 5
    assert sanitized['add_max'] == 5
    assert new_state['add_max']['value'] == 5


def test_sector_bias_decay_when_not_reaffirmed():
    baseline = {'sector_bias': {}}
    overrides = {}
    state = {
        'sector_bias': {
            'Energy': {
                'value': 0.10,
                'last_updated': '2025-09-30',
                'expiry': '2025-10-10',
            }
        }
    }
    sanitized, new_state = enforce_guardrails(
        overrides,
        baseline=baseline,
        metrics_count=30,
        state=state,
        today=date(2025, 10, 2),
    )
    assert math.isclose(sanitized['sector_bias']['Energy'], 0.08, rel_tol=1e-6)
    assert math.isclose(new_state['sector_bias']['Energy']['value'], 0.08, rel_tol=1e-6)


def test_ticker_bias_reaffirm_updates_expiry():
    baseline = {'ticker_bias': {}}
    overrides = {'ticker_bias': {'VNM': 0.05}}
    state = {
        'ticker_bias': {
            'VNM': {
                'value': 0.04,
                'last_updated': '2025-09-30',
                'expiry': '2025-10-02',
            }
        }
    }
    sanitized, new_state = enforce_guardrails(
        overrides,
        baseline=baseline,
        metrics_count=30,
        state=state,
        today=date(2025, 10, 2),
    )
    assert math.isclose(sanitized['ticker_bias']['VNM'], 0.05, rel_tol=1e-6)
    assert math.isclose(new_state['ticker_bias']['VNM']['value'], 0.05, rel_tol=1e-6)
    assert new_state['ticker_bias']['VNM']['expiry'] == '2025-10-07'
