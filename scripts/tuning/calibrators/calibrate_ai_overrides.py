from __future__ import annotations

"""
Calibrate AI overrides (sector_bias, ticker_bias) via Codex CLI.

Behavior
- Invokes the existing Codex-based generator to produce raw overrides,
  then merges allowed keys into the runtime policy at out/orders/policy_overrides.json.
- Allowed keys: sector_bias, ticker_bias. 'rationale' is ignored for runtime.

Inputs
- config/policy_default.json (for prompt/schema via generator)
- Runtime policy: out/orders/policy_overrides.json (created upstream)

Outputs
- Updates out/orders/policy_overrides.json (in-place) with AI overrides.

Fail-fast
- If Codex CLI or generator is unavailable, raises SystemExit with clear message.
"""

from pathlib import Path
import json
import re
from typing import Dict

BASE_DIR = Path(__file__).resolve().parents[3]
OUT_DIR = BASE_DIR / 'out'
ORDERS_DIR = OUT_DIR / 'orders'
CONFIG_DIR = BASE_DIR / 'config'

ALLOWED_KEYS = {'sector_bias', 'ticker_bias'}


def _strip_json_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(^|\s)//.*$", "", text, flags=re.M)
    text = re.sub(r"(^|\s)#.*$", "", text, flags=re.M)
    return text


def calibrate(*, write: bool = True) -> Dict:
    # Run the existing generator in-process; it will use Codex CLI and write
    # a temporary AI file. Then merge allowed keys into runtime policy.
    try:
        from scripts.tuning import codex_policy_budget_bias_tuner as tuner
    except Exception as exc:
        raise SystemExit(f"AI tuner unavailable: {exc}") from exc

    # Execute tuner main (runs Codex CLI) — may raise if CLI missing.
    tuner.main()

    ai_path = CONFIG_DIR / 'policy_ai_overrides.json'
    if not ai_path.exists():
        raise SystemExit('AI overrides not produced by tuner')
    raw = json.loads(_strip_json_comments(ai_path.read_text(encoding='utf-8')))
    # Filter to allowed keys only
    ai = {k: v for k, v in raw.items() if k in ALLOWED_KEYS}

    if write:
        pol_p = ORDERS_DIR / 'policy_overrides.json'
        if not pol_p.exists():
            raise SystemExit('Missing out/orders/policy_overrides.json before AI merge')
        pol = json.loads(_strip_json_comments(pol_p.read_text(encoding='utf-8')))
        for k, v in ai.items():
            pol[k] = v
        pol_p.write_text(json.dumps(pol, ensure_ascii=False, indent=2), encoding='utf-8')
    return ai


if __name__ == '__main__':
    calibrate(write=True)

