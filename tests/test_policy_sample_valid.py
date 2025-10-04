import unittest
from pathlib import Path

from scripts.engine.schema import PolicyOverrides


class TestPolicySampleValid(unittest.TestCase):
    def test_baseline_contains_valid_regime_model(self):
        # Validate the merged baseline with schema (comments allowed)
        text = Path('config/policy_default.json').read_text(encoding='utf-8')
        import json, re
        def _strip(s: str) -> str:
            s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
            s = re.sub(r"(^|\s)//.*$", "", s, flags=re.M)
            s = re.sub(r"(^|\s)#.*$", "", s, flags=re.M)
            return s
        obj = json.loads(_strip(text) or '{}')
        model = PolicyOverrides.model_validate(obj)
        self.assertIsNotNone(model.regime_model, 'regime_model must be present in baseline')
        self.assertTrue(model.regime_model.components, 'regime_model.components must not be empty')


if __name__ == '__main__':
    unittest.main()
