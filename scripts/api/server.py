from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, make_response, request
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parents[2]
IN_PORTFOLIO_DIR = BASE_DIR / 'in' / 'portfolio'
OUT_DIR = BASE_DIR / 'out'

"""Note: The server exposes only on-demand endpoints.
All scheduling/automation is handled by GitHub Actions.
"""


@dataclass
class PolicyRunRecord:
    started_at: datetime
    finished_at: datetime
    scheduled_for: Optional[datetime]
    trigger: str
    ok: bool
    out: str

    def to_api(self) -> Dict[str, Any]:
        return {
            'ok': self.ok,
            'out': self.out,
            'trigger': self.trigger,
            'started_at': self.started_at.isoformat(),
            'finished_at': self.finished_at.isoformat(),
            'scheduled_for': self.scheduled_for.isoformat() if self.scheduled_for else None,
        }


# No background scheduler: policy refresh is handled via GitHub Actions or
# explicit CLI (`./broker.sh policy`).


def _resp(data: Dict[str, Any], status: int = 200):
    r = make_response(jsonify(data), status)
    r.headers['Cache-Control'] = 'no-store'
    return r


def ensure_dirs():
    IN_PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)


def reset_portfolio() -> Dict[str, Any]:
    ensure_dirs()
    removed: List[str] = []
    for p in IN_PORTFOLIO_DIR.iterdir():
        if p.is_file() and not p.name.startswith('.'):
            p.unlink(missing_ok=True)
            removed.append(p.name)
    return {"status": "ok", "removed": removed}


def write_csv_exact(name: str, content_text: str) -> Dict[str, Any]:
    ensure_dirs()
    safe = ''.join(ch for ch in (name or '') if ch.isalnum() or ch in ('-', '_')).strip('_')
    if not safe:
        safe = f"pf_{int(time.time())}"
    dest = IN_PORTFOLIO_DIR / f"{safe}.csv"
    data = content_text.encode('utf-8')
    with dest.open('wb') as f:
        f.write(data)
    return {
        "status": "ok",
        "saved": str(dest.relative_to(BASE_DIR)),
        "bytes": len(data),
    }


def run_cmd(cmd: list[str]) -> Dict[str, Any]:
    print(f"[srv] $ {' '.join(cmd)}", flush=True)
    out_lines: list[str] = []
    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end='')
        out_lines.append(line)
    rc = proc.wait()
    ok = (rc == 0)
    return {"ok": ok, "out": ''.join(out_lines)}
def finalize_and_run() -> Dict[str, Any]:
    """Run the order pipeline on the uploaded portfolio files."""
    ensure_dirs()
    files: List[Path] = [p for p in IN_PORTFOLIO_DIR.glob('*.csv') if p.is_file()]
    if not files:
        return {
            "status": "error",
            "error": "no_files",
            "hint": f"No CSV files found in {str(IN_PORTFOLIO_DIR.relative_to(BASE_DIR))}",
        }
    result = run_cmd(['bash', 'broker.sh', 'orders'])
    ok = bool(result.get('ok'))
    orders_dir = OUT_DIR / 'orders'
    outputs: List[str] = []
    if ok and orders_dir.exists():
        outputs = [str(p.relative_to(BASE_DIR)) for p in sorted(orders_dir.glob('*')) if p.is_file()]
    return {
        "status": "ok" if ok else "error",
        "inputs": [str(p.relative_to(BASE_DIR)) for p in sorted(files)],
        "outputs": outputs,
        "run": result,
    }


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    @app.route('/health', methods=['GET'])
    def health():
        return _resp({"status": "ok", "ts": datetime.utcnow().isoformat() + 'Z'})

    @app.route('/portfolio/reset', methods=['POST', 'OPTIONS'])
    def api_reset():
        if request.method == 'OPTIONS':
            return _resp({"ok": True})
        return _resp(reset_portfolio())

    @app.route('/portfolio/upload', methods=['POST', 'OPTIONS'])
    def api_upload():
        if request.method == 'OPTIONS':
            return _resp({"ok": True})
        js = request.get_json(force=True, silent=True)
        if js is None:
            return _resp({"status": "error", "error": "invalid_json", "hint": "expect {name, content}"}, 400)
        if not isinstance(js, dict):
            return _resp({"status": "error", "error": "json_schema"}, 400)
        name = js.get('name')
        content = js.get('content')
        if not isinstance(name, str) or not isinstance(content, str):
            return _resp({"status": "error", "error": "json_schema", "hint": "name:string, content:string"}, 400)
        return _resp(write_csv_exact(name, content))

    @app.route('/done', methods=['POST', 'OPTIONS'])
    def api_done():
        if request.method == 'OPTIONS':
            return _resp({"ok": True})
        # Determine skip_policy from query param or env BROKER_SKIP_POLICY
        return _resp(finalize_and_run())

    return app


if __name__ == '__main__':
    ensure_dirs()
    port = int(os.getenv('PORT', '8787'))
    app = create_app()
    app.run(host='0.0.0.0', port=port)
