import re
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'scripts' / 'order_engine.py'
BASELINE = ROOT / 'config' / 'policy_default.json'


def _read(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def _strip_json_comments(s: str) -> str:
    # keep in sync with scripts/ai/generate_policy_overrides.py
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"(^|\s)//.*$", "", s, flags=re.M)
    s = re.sub(r"(^|\s)#.*$", "", s, flags=re.M)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s.strip()


def _extract_used_keys_py(code: str) -> dict:
    # Scan for keys referenced in engine code
    used = {
        'weights': set(),
        'thresholds': set(),
        'pricing': set(),
        'sizing': set(),
        'dynamic_caps': set(),
        'neutral_adaptive': set(),
    }
    # weights
    for pat in (r"weights\s*\[\s*'([\w_]+)'\s*\]", r"weights\.get\(\s*'([\w_]+)'\s*[,)]"):
        for m in re.finditer(pat, code):
            used['weights'].add(m.group(1))
    # thresholds
    for pat in (r"thresholds\s*\[\s*'([\w_]+)'\s*\]", r"thresholds\.get\(\s*'([\w_]+)'\s*[,)]"):
        for m in re.finditer(pat, code):
            used['thresholds'].add(m.group(1))
    # pricing
    for m in re.finditer(r"pricing(?:_conf)?\.get\(\s*'([\w_]+)'\s*[,)]", code):
        used['pricing'].add(m.group(1))
    # sizing
    for m in re.finditer(r"sizing\.get\(\s*'([\w_]+)'\s*[,)]", code):
        used['sizing'].add(m.group(1))
    # dynamic caps nested
    for m in re.finditer(r"dyn_caps\.get\(\s*'([\w_]+)'\s*[,)]", code):
        used['dynamic_caps'].add(m.group(1))
    # neutral adaptive keys
    for m in re.finditer(r"neutral_(?:config|state|adaptive)[^\n]*get\(\s*'([\w_]+)'\s*[,)]", code):
        used['neutral_adaptive'].add(m.group(1))
    for m in re.finditer(r"neutral_[\w]*\[['\"]([\w_]+)['\"]\]", code):
        used['neutral_adaptive'].add(m.group(1))
    return used


class TestPolicySampleSchemaSync(unittest.TestCase):
    def test_sample_contains_all_keys_used_in_engine(self):
        code = _read(ENGINE)
        used = _extract_used_keys_py(code)
        baseline_text = _strip_json_comments(_read(BASELINE))
        sample = json.loads(baseline_text or '{}')

        # Collect keys from sample
        weights_keys = set(sample.get('weights', {}).keys())
        thresholds_keys = set(sample.get('thresholds', {}).keys())
        pricing_keys = set(sample.get('pricing', {}).keys())
        sizing = sample.get('sizing', {}) or {}
        sizing_keys = set(sizing.keys())
        dyn = sizing.get('dynamic_caps', {}) or {}
        dyn_keys = set(dyn.keys())
        neutral_conf = sample.get('neutral_adaptive', {}) or {}
        neutral_keys = set(neutral_conf.keys())

        # Each used key should exist in the sample
        missing = {
            'weights': sorted(used['weights'] - weights_keys),
            'thresholds': sorted(used['thresholds'] - thresholds_keys),
            'pricing': sorted(used['pricing'] - pricing_keys),
            'sizing': sorted(used['sizing'] - sizing_keys),
            'dynamic_caps': sorted(used['dynamic_caps'] - dyn_keys),
            'neutral_adaptive': sorted(used['neutral_adaptive'] - neutral_keys),
        }

        problems = []
        for grp, keys in missing.items():
            if keys:
                problems.append(f"{grp}: {', '.join(keys)}")
        self.assertFalse(problems, f"Sample config missing keys used by engine -> { '; '.join(problems) }")


if __name__ == '__main__':
    unittest.main()
