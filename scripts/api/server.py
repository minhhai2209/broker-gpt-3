from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, make_response, request
from flask_cors import CORS
from zoneinfo import ZoneInfo, available_timezones

BASE_DIR = Path(__file__).resolve().parents[2]
IN_PORTFOLIO_DIR = BASE_DIR / 'in' / 'portfolio'
OUT_DIR = BASE_DIR / 'out'
# Folder that groups timestamped order runs
RUNS_DIR = BASE_DIR / 'runs'

# Current active archive timestamp folder (set on first upload after reset)
_CURRENT_STAMP: Optional[str] = None

def _stamp_now() -> str:
    tzname = os.getenv('BROKER_RUNS_TZ', os.getenv('BROKER_ARCHIVE_TZ', 'Asia/Ho_Chi_Minh'))
    try:
        tz = ZoneInfo(tzname)
    except Exception:
        tz = None
    now = datetime.now(tz) if tz else datetime.now()
    return now.strftime('%Y%m%d_%H%M%S')  # e.g., 20251004_093015

def _ensure_run_stamp() -> str:
    global _CURRENT_STAMP
    if not _CURRENT_STAMP:
        _CURRENT_STAMP = _stamp_now()
        (RUNS_DIR / _CURRENT_STAMP / 'portfolio').mkdir(parents=True, exist_ok=True)
        print(f"[srv] run stamp initialized: {_CURRENT_STAMP}", flush=True)
    return _CURRENT_STAMP

def _git_commit_push(paths: list[Path], message: str) -> Dict[str, Any]:
    added = []
    for p in paths:
        if p.exists():
            added.append(str(p.relative_to(BASE_DIR)))
    if not added:
        return {"ok": True, "out": "no changes"}
    cmds = [
        ['git', 'config', 'user.name', os.getenv('BROKER_GIT_USER', 'server-bot')],
        ['git', 'config', 'user.email', os.getenv('BROKER_GIT_EMAIL', 'server-bot@local')],
        ['git', 'add'] + added,
        ['git', 'commit', '-m', message],
        ['git', 'push'],
    ]
    out_all = []
    ok_overall = True
    for c in cmds:
        r = run_cmd(c)
        out_all.append(r.get('out', ''))
        ok_overall = ok_overall and bool(r.get('ok'))
        # If commit produced no changes, proceed to push to be safe
    return {"ok": ok_overall, "out": ''.join(out_all)}


def _parse_time_list(raw: str) -> List[dtime]:
    items: List[dtime] = []
    for chunk in raw.split(','):
        part = chunk.strip()
        if not part:
            continue
        if ':' not in part:
            raise ValueError(f"Invalid time entry '{part}'")
        hh, mm = part.split(':', 1)
        if not (hh.isdigit() and mm.isdigit()):
            raise ValueError(f"Invalid time entry '{part}'")
        hour = int(hh)
        minute = int(mm)
        if not (0 <= hour < 24 and 0 <= minute < 60):
            raise ValueError(f"Invalid time entry '{part}'")
        items.append(dtime(hour=hour, minute=minute))
    if not items:
        raise ValueError('At least one time entry required')
    return sorted(items)


def _parse_timezone(name: str) -> ZoneInfo:
    if name not in available_timezones():  # pragma: no cover - environment misconfiguration guardrail
        raise ValueError(f"Invalid timezone '{name}'")
    return ZoneInfo(name)


def _default_policy_times() -> List[dtime]:
    return _parse_time_list(
        os.getenv(
            'BROKER_POLICY_TIMES',
            '09:10,09:40,10:10,10:40,11:10,11:31,13:10,13:40,14:00,14:15',
        )
    )


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


@dataclass
class PolicyRunRequest:
    trigger: str
    scheduled_for: Optional[datetime]


class PolicyScheduler:
    def __init__(
        self,
        *,
        times: List[dtime],
        lead: timedelta,
        max_age: Optional[timedelta],
        timezone_name: str,
    ) -> None:
        self._times = sorted(times)
        self._lead = lead
        self._max_age = max_age
        self._tz = _parse_timezone(timezone_name)
        self._condition = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._latest: Optional[PolicyRunRecord] = None
        self._force_request: Optional[PolicyRunRequest] = None
        self._next_run_at: Optional[datetime] = None
        self._next_scheduled_slot: Optional[datetime] = None
        self._current_trigger: Optional[str] = None
        self._current_scheduled: Optional[datetime] = None

    def _now(self) -> datetime:
        return datetime.now(self._tz)

    def _compute_next_run(self, reference: datetime) -> tuple[datetime, Optional[datetime]]:
        for day_offset in range(0, 8):
            day = (reference + timedelta(days=day_offset)).date()
            for slot in self._times:
                scheduled_dt = datetime.combine(day, slot, tzinfo=self._tz)
                # If we are already inside the lead window, run immediately
                run_dt = scheduled_dt - self._lead
                if reference >= scheduled_dt:
                    continue
                if reference >= run_dt:
                    return reference, scheduled_dt
                return run_dt, scheduled_dt
        raise RuntimeError('Unable to compute next policy run window')

    def _queue_next_run_locked(self) -> None:
        reference = self._now()
        if self._latest and self._latest.scheduled_for is not None:
            delta = self._latest.scheduled_for - self._latest.finished_at
            if delta <= (self._lead / 2):
                reference = max(reference, self._latest.scheduled_for)
        self._next_run_at, self._next_scheduled_slot = self._compute_next_run(reference)
        self._condition.notify_all()

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                return
            self._next_run_at, self._next_scheduled_slot = self._compute_next_run(self._now())
            self._thread = threading.Thread(target=self._worker, name='PolicyScheduler', daemon=True)
            self._thread.start()

    def _worker(self) -> None:
        while True:
            with self._condition:
                while True:
                    if self._force_request and not self._running:
                        req = self._force_request
                        self._force_request = None
                        trigger = req.trigger
                        scheduled = req.scheduled_for if req.scheduled_for is not None else self._next_scheduled_slot
                        break
                    if self._next_run_at is None:
                        self._queue_next_run_locked()
                        continue
                    now = self._now()
                    if self._running:
                        self._condition.wait()
                        continue
                    if now >= self._next_run_at:
                        trigger = 'scheduled'
                        scheduled = self._next_scheduled_slot
                        break
                    wait_seconds = max(0.0, (self._next_run_at - now).total_seconds())
                    self._condition.wait(timeout=wait_seconds)
                self._running = True
                started_at = self._now()
                scheduled_for = scheduled
                current_trigger = trigger
                self._current_trigger = current_trigger
                self._current_scheduled = scheduled_for
            # Run outside the lock
            if scheduled_for is not None:
                print(
                    f"[srv] === policy job start [{current_trigger}] for {scheduled_for.isoformat()} ===",
                    flush=True,
                )
            else:
                print(f"[srv] === policy job start [{current_trigger}] ===", flush=True)
            result = run_cmd(['bash', 'broker.sh', 'policy'])
            finished_at = self._now()
            record = PolicyRunRecord(
                started_at=started_at,
                finished_at=finished_at,
                scheduled_for=scheduled_for,
                trigger=current_trigger,
                ok=bool(result.get('ok')),
                out=result.get('out', ''),
            )
            print(
                f"[srv] === policy job done [{record.trigger}] ok={record.ok} finished {record.finished_at.isoformat()} ===",
                flush=True,
            )
            with self._condition:
                self._latest = record
                self._running = False
                self._current_trigger = None
                self._current_scheduled = None
                self._queue_next_run_locked()

    def ensure_ready(self) -> PolicyRunRecord:
        target_min_finished = None
        if self._max_age is not None:
            target_min_finished = self._now() - self._max_age
        with self._condition:
            while True:
                if self._latest and (target_min_finished is None or self._latest.finished_at >= target_min_finished):
                    if not self._running:
                        return self._latest
                if not self._running and not self._force_request:
                    self._force_request = PolicyRunRequest(
                        trigger='ensure',
                        scheduled_for=self._next_scheduled_slot,
                    )
                    self._condition.notify_all()
                self._condition.wait()

    def status(self) -> Dict[str, Any]:
        with self._condition:
            latest = self._latest.to_api() if self._latest else None
            running = self._running
            next_run = self._next_run_at.isoformat() if self._next_run_at else None
            next_slot = self._next_scheduled_slot.isoformat() if self._next_scheduled_slot else None
            current = None
            if self._running:
                current = {
                    'trigger': self._current_trigger,
                    'scheduled_for': self._current_scheduled.isoformat() if self._current_scheduled else None,
                }
        return {
            'running': running,
            'next_run_at': next_run,
            'next_scheduled_slot': next_slot,
            'latest': latest,
            'current_run': current,
        }


# Behavioral env toggles removed: server runs in a single, stable mode with no
# automatic policy scheduler. Policy refresh is handled explicitly via CI
# workflows or the `broker.sh policy` command.
POLICY_SCHEDULER: Optional[PolicyScheduler] = None


def _resp(data: Dict[str, Any], status: int = 200):
    r = make_response(jsonify(data), status)
    r.headers['Cache-Control'] = 'no-store'
    return r


def ensure_dirs():
    IN_PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def reset_portfolio() -> Dict[str, Any]:
    ensure_dirs()
    removed: List[str] = []
    for p in IN_PORTFOLIO_DIR.iterdir():
        if p.is_file() and not p.name.startswith('.'):
            p.unlink(missing_ok=True)
            removed.append(p.name)
    # Clear current run stamp; next upload initializes a new timestamped folder
    global _CURRENT_STAMP
    _CURRENT_STAMP = None
    return {"status": "ok", "removed": removed, "runs_reset": True}


def write_csv_exact(name: str, content_text: str) -> Dict[str, Any]:
    ensure_dirs()
    safe = ''.join(ch for ch in (name or '') if ch.isalnum() or ch in ('-', '_')).strip('_')
    if not safe:
        safe = f"pf_{int(time.time())}"
    dest = IN_PORTFOLIO_DIR / f"{safe}.csv"
    data = content_text.encode('utf-8')
    with dest.open('wb') as f:
        f.write(data)
    # Also write to runs/<stamp>/portfolio and push to trigger pipeline
    stamp = _ensure_run_stamp()
    run_dest = RUNS_DIR / stamp / 'portfolio' / f"{safe}.csv"
    with run_dest.open('wb') as f:
        f.write(data)
    # Do NOT commit here; commit occurs at /done after all uploads finish
    return {
        "status": "ok",
        "saved": str(dest.relative_to(BASE_DIR)),
        "run_saved": str(run_dest.relative_to(BASE_DIR)),
        "bytes": len(data),
        "pending_commit": True,
        "run_stamp": stamp,
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


def _env_truth(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ('1', 'true', 'yes', 'y', 'on')


def finalize_and_run() -> Dict[str, Any]:
    """
    Finalize the current upload session by committing all files in
    runs/<stamp>/portfolio. This triggers the external CI to produce outputs.

    - Fails fast if there is no active stamp or no files to commit.
    - Resets the active stamp so a subsequent upload session starts fresh.
    """
    global _CURRENT_STAMP
    ensure_dirs()
    if not _CURRENT_STAMP:
        return {
            "status": "error",
            "error": "no_active_session",
            "hint": "Upload portfolio files first via /portfolio/upload",
        }
    stamp = _ensure_run_stamp()
    portfolio_dir = RUNS_DIR / stamp / 'portfolio'
    files: List[Path] = [p for p in portfolio_dir.glob('*.csv') if p.is_file()]
    if not files:
        return {
            "status": "error",
            "error": "no_files",
            "hint": f"No CSV files found in {str(portfolio_dir.relative_to(BASE_DIR))}",
        }
    git_res = _git_commit_push(files, f"runs: add portfolio batch for {stamp}")
    # Reset current stamp to ensure next upload batch starts a new run folder
    _CURRENT_STAMP = None
    scheduler_status = POLICY_SCHEDULER.status() if POLICY_SCHEDULER is not None else None
    return {
        "status": "ok" if git_res.get("ok") else "error",
        "committed": [str(p.relative_to(BASE_DIR)) for p in files],
        "git": git_res,
        "run_stamp": stamp,
        "policy_scheduler": scheduler_status,
    }


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    @app.route('/health', methods=['GET'])
    def health():
        return _resp({"status": "ok", "ts": datetime.utcnow().isoformat() + 'Z'})

    @app.route('/policy/status', methods=['GET'])
    def policy_status():
        if POLICY_SCHEDULER is None:
            return _resp({"auto_run": False})
        return _resp({"auto_run": True, "scheduler": POLICY_SCHEDULER.status()})

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
