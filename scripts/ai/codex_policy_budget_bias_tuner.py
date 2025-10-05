from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = BASE_DIR / "config"
OUT_DIR = BASE_DIR / "out"
ANALYSIS_FILENAME = "analysis_round.txt"
OUTPUT_FILENAME = "policy_overrides.generated.json"

# Ensure repo root is importable (match scripts/generate_orders.py behavior)
import sys as _sys
if str(BASE_DIR) not in _sys.path:
    _sys.path.insert(0, str(BASE_DIR))

from scripts.ai.guardrails import apply_guardrails


def _test_mode_enabled() -> bool:
    val = os.getenv('BROKER_TEST_MODE', '').strip().lower()
    return val not in {'', '0', 'false', 'no', 'off'}


def _resolve_reasoning_effort() -> str:
    override = os.getenv('BROKER_CX_REASONING', '').strip()
    if override:
        return override
    return 'low' if _test_mode_enabled() else 'high'


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""

def build_prompt(sample_json: str, analysis_so_far: str, round_idx: int, max_rounds: int) -> str:
    # Strong, explicit constraints so the model adheres to schema exactly
    # Use Vietnam time (UTC+7)
    VN_TZ = timezone(timedelta(hours=7))
    now_str = datetime.now(VN_TZ).strftime('%Y-%m-%d %H:%M:%S %Z%z')
    carry = ("\n\nPhân tích đã có (tích lũy từ các lượt trước, KHÔNG chứa JSON):\n" + analysis_so_far.strip()) if analysis_so_far.strip() else ""
    # Explicitly constrain the tunable override keys; keep the surface minimal
    # to reduce overlap with calibrators and improve stability/auditability.
    # Calibrators compute quantiles, ATR-based thresholds, market filters, sizing, etc.
    # Daily AI/script may only touch the following knobs:
    # Minimize tunable surface: no direct slot overrides from AI.
    # Slots may still be derived internally from 'news_risk_tilt' by guardrails,
    # but the generator must not set 'add_max' or 'new_max' explicitly.
    allowed_keys = [
        'buy_budget_frac',      # risk-on/off budget tilt
        'sector_bias',          # sector‑level tilts in [-0.20..0.20]
        'ticker_bias',          # ticker‑level tilts in [-0.20..0.20]
        'news_risk_tilt',       # optional helper input in [-1..+1]; mapped by guardrails
        'rationale',            # required: brief natural‑language justification for changes
    ]
    allowed_list = "\n".join([f"- {k}" for k in allowed_keys])
    prompt = f"""
Bạn là chuyên gia cấu hình hệ thống giao dịch.

Nhiệm vụ của bạn: sinh "policy_overrides" NHẸ chứa CHỈ các khoá được phép điều chỉnh thời gian thực để override lên "policy_default" của hệ thống. KHÔNG thay đổi các phần còn lại.

 Ràng buộc QUAN TRỌNG:
 - CHỈ điều chỉnh các khoá override được phép dưới đây. Không thêm bất kỳ khoá nào khác, không lặp lại các phần đã cố định trong default.
 - KHÔNG được ghi đè các khoá do calibrator tính toán (quantile, ATR, market filter, sizing...).
 - Tập khoá cho phép (chỉ các khoá bên dưới):
{allowed_list}

Bạn có thể dựa trên tin tức/sentiment mới nhất để điều chỉnh các khoá cho phép ở trên.
YÊU CẦU: phải cung cấp trường 'rationale' (string) mô tả ngắn gọn lý do thay đổi, nguồn tin và thời hạn hiệu lực (TTL).
Tuỳ chọn: nếu tiện, bạn có thể xuất 'news_risk_tilt' trong [-1..+1] (âm = risk‑off), script sẽ tự ánh xạ sang budget/slots theo guardrails.

Chế độ nhiều lượt (multi-round):
- Bạn có nhiều lượt phân tích. Lên kế hoạch cho tối ưu. Không được cố gắng hoàn thành trong một lượt.
- Bạn không hỏi xác nhận hay ý kiến gì từ người dùng. Bạn phải tự động cho phương án tối ưu.
- Khi cần thêm lượt, chỉ in ra "CONTINUE". "CONTINUE" phải là nội dung duy nhất trên dòng chứa nó.
- Ở mỗi lượt, bạn PHẢI ghi toàn bộ đoạn phân tích vào file "{ANALYSIS_FILENAME}" (ghi đè, UTF-8). Các đoạn phân tích ở các lượt sẽ được ghép với nhau làm context cho lượt phân tích tiếp theo.
- Khi đã hoàn tất mọi phân tích, bạn PHẢI ghi file JSON thuần (không code fence, không comment) vào đường dẫn "{OUTPUT_FILENAME}" (UTF-8), tuân thủ schema bên dưới.
- Sau khi đã ghi config ra file, bạn chỉ in ra "END". "END" phải là nội dung duy nhất trên dòng chứa nó.
- Khi đã in ra "CONTINUE" hoặc "END", bạn không được in thêm gì nữa.

Thời điểm (VN): {now_str}
Bạn có tối đa {max_rounds} lượt. Bạn không bắt buộc phải dùng hết số lượt.
Lượt hiện tại: {round_idx}
{carry}

Schema tham chiếu (để hiểu ngữ cảnh và giới hạn):
{sample_json}
"""
    return prompt


def _copy_generated(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(src.read_text(encoding='utf-8'), encoding='utf-8')
    print(f"Copied {src} -> {dest}")


def _write_analysis_dump(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_overrides_merged(target_path: Path, sanitized: Dict[str, object]) -> None:
    """Write sanitized runtime overrides while preserving non-runtime keys.

    Rationale:
    - Runtime deep-merge happens inside the engine (baseline + overrides -> out/orders/policy_overrides.json).
    - The generator should not blow away calibrations/metadata (e.g., TTL buckets) that may co-exist
      in config/policy_overrides.json. We therefore remove only the runtime knobs the generator owns
      and replace them with the sanitized values.
    """
    import json
    # Keys owned by the generator/guardrails at the top level
    runtime_keys = {
        'buy_budget_frac',
        'add_max',
        'new_max',
        'sector_bias',
        'ticker_bias',
        # Ephemeral inputs that should never persist if present
        'news_risk_tilt',
        'rationale',
    }
    existing: Dict[str, object] = {}
    if target_path.exists():
        try:
            existing = json.loads(target_path.read_text(encoding='utf-8'))
        except Exception:
            # Fail-fast: if file is corrupt, do not try to merge silently
            raise SystemExit(f'Invalid JSON in existing overrides: {target_path}')

    preserved = {k: v for k, v in (existing or {}).items() if k not in runtime_keys}

    def _deep_merge(dst: Dict[str, object], src: Dict[str, object]) -> Dict[str, object]:
        out = dict(dst)
        for k, v in (src or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = _deep_merge(out[k], v)  # type: ignore[arg-type]
            else:
                out[k] = v
        return out

    merged = _deep_merge(preserved, sanitized or {})
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Guardrails applied: deep-merged runtime overrides into {target_path} (preserved {len(preserved)} non-runtime keys)")


def main() -> None:
    # Prefer a complete default policy if present; fallback to sample schema
    default_path = CONFIG_DIR / 'policy_default.json'
    sample_path = CONFIG_DIR / 'policy_overrides.sample.json'
    base_schema_path = default_path if default_path.exists() else sample_path
    if not base_schema_path.exists():
        raise SystemExit(f'Missing {base_schema_path}')
    sample = _read_text(base_schema_path)
    analysis_accum = ""
    max_rounds = int(os.getenv('BROKER_CX_GEN_ROUNDS', '1'))
    analysis_dump_latest = (OUT_DIR / 'debug') / 'codex_analysis_latest.txt'

    for round_idx in range(1, max_rounds + 1):
        prompt = build_prompt(sample, analysis_accum, round_idx, max_rounds)

        print(f"[codex] Round {round_idx}/{max_rounds}: Generating via Codex CLI (gpt-5 + web_search)...")
        cmd = [
            'codex', 'exec',
            '--skip-git-repo-check',  # run outside trusted repo; we isolate in temp dir
            '--full-auto',
            '--model', 'gpt-5',
            '-c', 'tools.web_search=true',
            '-c', f'reasoning_effort={_resolve_reasoning_effort()}',
            '-',
        ]
        analysis_file_text: str = ""

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
                check=True,
                cwd=tmp_cwd,
            )
            text = proc.stdout.decode('utf-8', errors='replace')
            if not analysis_file_path.exists():
                analysis_file_text = ''

            ts = datetime.now(timezone.utc).astimezone().strftime('%Y%m%d_%H%M%S')
            debug_dir = OUT_DIR / 'debug'
            debug_dir.mkdir(parents=True, exist_ok=True)
            raw_path = debug_dir / f'codex_policy_raw_{ts}_r{round_idx}.txt'
            raw_path.write_text(text, encoding='utf-8')

            analysis_file_text = analysis_file_path.read_text(encoding='utf-8')
            # Persist latest analysis for the round
            _write_analysis_dump(analysis_dump_latest, analysis_file_text)

            # Detect END/CONTINUE marker
            lines = text.splitlines()
            marker = None
            for idx in range(len(lines) - 1, -1, -1):
                s = lines[idx].strip()
                if s == 'CONTINUE' or s == 'END':
                    marker = s
                    break
            print(f"[codex] Detected marker: {marker if marker else '(none)'}")

            if marker == 'CONTINUE':
                analysis_accum = (analysis_accum + ("\n\n" if analysis_accum else "") + analysis_file_text)
                continue

            if marker == 'END':
                # Require generated file, then copy to config/policy_overrides.json
                if not output_file_path.exists():
                    # Fallback: if model printed inline JSON instead of file, surface raw output
                    debug_dir = OUT_DIR / 'debug'
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    raw_path = debug_dir / f'codex_policy_raw_END_{round_idx}.txt'
                    raw_path.write_text(text, encoding='utf-8')
                    raise SystemExit(f"Missing generated file {OUTPUT_FILENAME}. Raw output saved to {raw_path}")
                raw_overrides = json.loads(output_file_path.read_text(encoding='utf-8'))
                sanitized = apply_guardrails(raw_overrides)
                target_path = CONFIG_DIR / 'policy_overrides.json'
                _write_overrides_merged(target_path, sanitized)
                print(f"[codex] Completed in round {round_idx} (file written).")
                return

        # Marker missing: abort with raw output for inspection
        raise SystemExit('Codex output must contain a CONTINUE or END marker. Aborting.')
    raise SystemExit(f'Exceeded maximum analysis rounds without producing {OUTPUT_FILENAME}.')



if __name__ == '__main__':
    main()
