import re
import unittest
from pathlib import Path

from scripts.engine.schema import PolicyOverrides


def _strip_json_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"(^|\s)//.*$", "", s, flags=re.M)
    s = re.sub(r"(^|\s)#.*$", "", s, flags=re.M)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s.strip()


class TestParseSamplePolicy(unittest.TestCase):
    def test_baseline_parses_with_schema(self):
        root = Path(__file__).resolve().parents[1]
        base = root / 'config' / 'policy_default.json'
        import json
        base_obj = json.loads(_strip_json_comments(base.read_text(encoding='utf-8')) or '{}')
        # Should not raise
        model = PolicyOverrides.model_validate(base_obj)
        obj = model.model_dump()
        # ticker_overrides exists (may be empty)
        self.assertIn('ticker_overrides', obj)
        # No strict requirement on concrete examples; presence of the object is enough


if __name__ == '__main__':
    unittest.main()
