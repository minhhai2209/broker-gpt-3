from __future__ import annotations

"""
Calibrate tilts (sector_bias, ticker_bias) via Codex CLI, calibrator style.

- Invokes Codex to generate a lightweight overrides JSON.
- Filters allowed keys (sector_bias, ticker_bias) and merges into
  out/orders/policy_overrides.json.
- Does not write legacy config/policy_ai_overrides.json or print legacy messages.
"""

from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parents[3]
OUT_DIR = BASE_DIR / 'out'
ORDERS_DIR = OUT_DIR / 'orders'
CONFIG_DIR = BASE_DIR / 'config'

ANALYSIS_FILENAME = 'analysis_round.txt'
OUTPUT_FILENAME = 'policy_overrides.generated.json'
ALLOWED_KEYS = {'sector_bias', 'ticker_bias'}
REASONING_EFFORT = 'high'  # fixed; avoid env-driven variants


def _strip_json_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(^|\s)//.*$", "", text, flags=re.M)
    text = re.sub(r"(^|\s)#.*$", "", text, flags=re.M)
    return text



def _read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8') if path.exists() else ''


def _build_prompt(sample_json: str, analysis_so_far: str, round_idx: int, max_rounds: int) -> str:
    VN_TZ = timezone(timedelta(hours=7))
    now_str = datetime.now(VN_TZ).strftime('%Y-%m-%d %H:%M:%S %Z%z')
    carry = ("\n\nPhân tích đã có (tích lũy từ các lượt trước, KHÔNG chứa JSON):\n" + analysis_so_far.strip()) if analysis_so_far.strip() else ""
    allowed_list = "\n".join([f"- {k}" for k in sorted(ALLOWED_KEYS)])
    return f"""
Bạn là chuyên gia cấu hình hệ thống giao dịch.

Nhiệm vụ: sinh policy_overrides NHẸ chỉ chứa các khoá sau và RATIONALE:
{allowed_list}

YÊU CẦU:
- Cung cấp 'rationale' (string) mô tả ngắn gọn lý do thay đổi, nguồn tin, TTL.
- KHÔNG thêm khoá ngoài danh sách.
- KHÔNG đụng vào các khoá do calibrator tính (market_filter, sizing, thresholds,...).

Chế độ nhiều lượt (multi-round):
- Có thể dùng nhiều lượt. Nếu cần thêm lượt, chỉ in ra CONTINUE.
- Khi hoàn tất và đã ghi file JSON {OUTPUT_FILENAME} vào CWD, chỉ in ra END.

Thời điểm (VN): {now_str}
Tối đa lượt: {max_rounds}
Lượt hiện tại: {round_idx}
{carry}

Schema tham chiếu (rút gọn):
{sample_json}
"""


def calibrate(*, write: bool = True) -> Dict:
    # Ensure baseline exists
    base_schema_path = CONFIG_DIR / 'policy_default.json'
    if not base_schema_path.exists():
        raise SystemExit(f'Missing {base_schema_path}')
    sample = _read_text(base_schema_path)

    # Find Codex CLI
    codex_bin = shutil.which('codex')
    if not codex_bin:
        raise SystemExit('Missing Codex CLI. Install with: npm install -g @openai/codex@latest')

    # Fixed number of rounds to avoid env-driven variability.
    max_rounds = 1
    analysis_accum = ''

    for round_idx in range(1, max_rounds + 1):
        prompt = _build_prompt(sample, analysis_accum, round_idx, max_rounds)
        cmd = [
            codex_bin,
            'exec',
            '--skip-git-repo-check',
            '--full-auto',
            '--model', 'gpt-5',
            '-c', 'tools.web_search=true',
            '-c', f'reasoning_effort={REASONING_EFFORT}',
            '-',
        ]
        with tempfile.TemporaryDirectory(prefix='codex_isolated_') as tmp_cwd:
            analysis_file_path = Path(tmp_cwd) / ANALYSIS_FILENAME
            output_file_path = Path(tmp_cwd) / OUTPUT_FILENAME
            analysis_file_path.unlink(missing_ok=True)
            output_file_path.unlink(missing_ok=True)
            proc = subprocess.run(
                cmd,
                input=prompt.encode('utf-8'),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                cwd=tmp_cwd,
            )
            text = proc.stdout.decode('utf-8', errors='replace')
            # Persist raw output for inspection
            ts = datetime.now(timezone.utc).astimezone().strftime('%Y%m%d_%H%M%S')
            debug_dir = OUT_DIR / 'debug'
            debug_dir.mkdir(parents=True, exist_ok=True)
            raw_path = debug_dir / f'codex_policy_raw_{ts}_r{round_idx}.txt'
            raw_path.write_text(text, encoding='utf-8')
            if proc.returncode != 0:
                err_path = debug_dir / f'codex_policy_error_{ts}_r{round_idx}.txt'
                err_path.write_text(text, encoding='utf-8')
                print(f"[codex] Command failed (exit {proc.returncode}); raw output saved to {err_path}")
                raise SystemExit(f"Codex CLI failed with exit code {proc.returncode}; see {err_path}")
            analysis_text = _read_text(analysis_file_path)
            # Track latest analysis
            (debug_dir / 'codex_analysis_latest.txt').write_text(analysis_text, encoding='utf-8')
            # Detect marker
            marker = None
            for line in reversed(text.splitlines()):
                s = line.strip()
                if s in ('CONTINUE', 'END'):
                    marker = s
                    break
            print(f"[codex] Detected marker: {marker if marker else '(none)'}")
            if marker == 'CONTINUE':
                analysis_accum = (analysis_accum + ("\n\n" if analysis_accum else "") + analysis_text)
                continue
            if marker == 'END':
                if not output_file_path.exists():
                    raw_path = debug_dir / f'codex_policy_raw_END_{round_idx}.txt'
                    raw_path.write_text(text, encoding='utf-8')
                    raise SystemExit(f"Missing generated file {OUTPUT_FILENAME}. Raw output saved to {raw_path}")
                raw_overrides = json.loads(output_file_path.read_text(encoding='utf-8'))
                # Filter allowed keys
                ai = {k: v for k, v in raw_overrides.items() if k in ALLOWED_KEYS}
                if write:
                    pol_p = ORDERS_DIR / 'policy_overrides.json'
                    if not pol_p.exists():
                        raise SystemExit('Missing out/orders/policy_overrides.json before AI merge')
                    pol = json.loads(_strip_json_comments(pol_p.read_text(encoding='utf-8')))
                    for k, v in ai.items():
                        pol[k] = v
                    pol_p.write_text(json.dumps(pol, ensure_ascii=False, indent=2), encoding='utf-8')
                return ai
            # No marker
            raise SystemExit('Codex output must contain a CONTINUE or END marker. Aborting.')
    raise SystemExit(f'Exceeded maximum analysis rounds without producing {OUTPUT_FILENAME}.')


if __name__ == '__main__':
    calibrate(write=True)
