import sys
from pathlib import Path

# Ensure repo root is importable
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.orders.order_engine import run

if __name__ == "__main__":
    run()
