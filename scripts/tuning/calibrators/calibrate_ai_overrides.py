from __future__ import annotations

"""Codex-driven AI overrides calibrator (pre-phase).

This module runs before the numeric calibrators to produce
``config/policy_ai_overrides.json`` with a strict whitelist and guardrails.
It supports direct injection of overrides (for testing) and performs
clamping, validation, and audit logging as described in the unified
specification.
"""

from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple, Set


BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_DIR = BASE_DIR / "config"
OUT_DIR = BASE_DIR / "out"
AI_OVERRIDES_PATH = CONFIG_DIR / "policy_ai_overrides.json"
AUDIT_PATH = OUT_DIR / "debug" / "policy_ai_overrides_audit.ndjson"


# Narrow Codex AI scope: only biases + event-style execution hints + optional per-ticker gates
ALLOWED_TOP_LEVEL = {
    "rationale",          # free-text audit note
    "market_bias",        # global lean in [-0.2..0.2]
    "sector_bias",        # map of sector -> bias in [-0.2..0.2]
    "ticker_bias",        # map of TICKER -> bias in [-0.2..0.2]
}

# (Deprecated) Older prompts asked Codex to output calibration_targets / budget knobs.
# We explicitly reject those now to keep AI intent narrow and auditable.
ALLOWED_CALIBRATION_TARGETS: dict = {}

# New execution keys allowed from Codex (event-style only)
# No execution/event keys accepted in this mode
ALLOWED_EXECUTION_EVENT: set = set()

BIAS_RANGE = (-0.20, 0.20)
EVENT_BUDGET_SCALE_RANGE = (0.0, 1.0)


def _strip_json_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(^|\s)//.*$", "", text, flags=re.M)
    text = re.sub(r"(^|\s)#.*$", "", text, flags=re.M)
    return text


def _clamp(value: float, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    return max(lo, min(hi, float(value)))


def _clamp_int(value: int, bounds: Tuple[int, int]) -> int:
    lo, hi = bounds
    return int(max(lo, min(hi, int(value))))


def _ensure_dict(obj: Any, context: str) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise SystemExit(f"{context} must be an object")
    return dict(obj)


def _normalise_bias_map(raw: Any, *, context: str) -> Dict[str, float]:
    mp = _ensure_dict(raw, context)
    out: Dict[str, float] = {}
    for key, value in mp.items():
        if not isinstance(key, str):
            raise SystemExit(f"{context} keys must be strings")
        try:
            val = float(value)
        except Exception as exc:
            raise SystemExit(f"{context}[{key}] must be numeric") from exc
        out[key.upper()] = _clamp(val, BIAS_RANGE)
    return out


def _normalise_calibration_targets(raw: Any) -> Dict[str, Dict[str, float]]:
    # Explicitly disallow calibration_targets from Codex now
    raise SystemExit("calibration_targets are not accepted from Codex AI pre-phase")


def _normalise_execution(raw: Any) -> Dict[str, Any]:
    # Execution is not accepted from Codex in this mode
    raise SystemExit("'execution' is not accepted from Codex AI pre-phase")


def _normalise_overrides(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise SystemExit("Codex output must be a JSON object")
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        if key not in ALLOWED_TOP_LEVEL:
            raise SystemExit(f"Unsupported override key '{key}'")
        if key == "rationale":
            if value is None:
                continue
            if not isinstance(value, str):
                raise SystemExit("rationale must be a string")
            out[key] = value.strip()
        elif key == "market_bias":
            try:
                out[key] = _clamp(float(value), BIAS_RANGE)
            except Exception as exc:
                raise SystemExit("market_bias must be numeric") from exc
        elif key == "sector_bias":
            out[key] = _normalise_bias_map(value, context="sector_bias")
        elif key == "ticker_bias":
            out[key] = _normalise_bias_map(value, context="ticker_bias")
    # Filter by internal universe (HOSE) after normalization
    tickers_u, sectors_u = _load_universe_sets()
    if out.get("ticker_bias"):
        out["ticker_bias"] = {k: v for k, v in out["ticker_bias"].items() if k in tickers_u}
    if out.get("sector_bias"):
        out["sector_bias"] = {k: v for k, v in out["sector_bias"].items() if k in sectors_u}
    return out


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return {}
    return json.loads(_strip_json_comments(raw))


def _flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    if isinstance(obj, dict):
        items: Dict[str, Any] = {}
        for key, value in obj.items():
            new_key = f"{prefix}.{key}" if prefix else str(key)
            items.update(_flatten(value, new_key))
        return items
    return {prefix: obj}


def _write_ai_overrides(new_data: Dict[str, Any]) -> None:
    prev = _read_json(AI_OVERRIDES_PATH)
    AI_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    AI_OVERRIDES_PATH.write_text(
        json.dumps(new_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    audit_dir = AUDIT_PATH.parent
    audit_dir.mkdir(parents=True, exist_ok=True)
    prev_flat = _flatten(prev)
    new_flat = _flatten(new_data)
    changed_keys = sorted(k for k in set(prev_flat) | set(new_flat) if prev_flat.get(k) != new_flat.get(k))
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": "codex",
        "keys_changed": changed_keys,
        "changes": {k: {"old": prev_flat.get(k), "new": new_flat.get(k)} for k in changed_keys},
    }
    rationale = new_data.get("rationale")
    if isinstance(rationale, str) and rationale.strip():
        entry["rationale"] = rationale.strip()
    with AUDIT_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_universe_sets() -> Tuple[Set[str], Set[str]]:
    """Return (tickers_set, sectors_set) from data/industry_map.csv."""
    import csv as _csv
    path = BASE_DIR / "data" / "industry_map.csv"
    tickers: Set[str] = set()
    sectors: Set[str] = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            rd = _csv.DictReader(fh)
            for row in rd:
                t = str(row.get("Ticker", "")).strip().upper()
                s = str(row.get("Sector", "")).strip()
                if t:
                    tickers.add(t)
                if s:
                    sectors.add(s)
    return tickers, sectors


def _build_prompt(sample_json: str) -> str:
    vn_tz = timezone(timedelta(hours=7))
    now_str = datetime.now(vn_tz).strftime("%Y-%m-%d %H:%M:%S %Z%z")
    example_json = (
        "{\n"
        "  \"rationale\": \"Tóm tắt tin tức và lí do (nguồn: VNExpress, Cafef, Bloomberg…).\",\n"
        "  \"market_bias\": 0.05,\n"
        "  \"sector_bias\": {\"Tài chính\": 0.04, \"Công nghệ thông tin\": 0.03},\n"
        "  \"ticker_bias\": {\"CTG\": 0.05, \"FPT\": 0.04}\n"
        "}"
    )
    return (
        "Chỉ in JSON hợp lệ (không kèm văn bản khác) VÀ ghi đúng nội dung vào file 'policy_overrides.generated.json' trong thư mục hiện tại.\n\n"
        "Nhiệm vụ duy nhất:\n"
        "- Đọc tin tức/nguồn công khai (VN/EN) 3–7 ngày gần đây; tổng hợp catalyst/risks.\n"
        "- Khuyến nghị NGÀNH (sector_bias) và MÃ (ticker_bias) trên sàn HOSE với bias ∈ [-0.20, 0.20].\n\n"
        "Ràng buộc đầu ra (JSON):\n"
        "- Khóa cấp 1: rationale, market_bias (tuỳ chọn), sector_bias, ticker_bias.\n"
        "- sector_bias: map {Sector -> bias}; ticker_bias: map {Ticker -> bias}.\n"
        "- Bỏ qua tất cả khóa khác; nếu không có thay đổi hãy để map rỗng nhưng vẫn có rationale.\n\n"
        f"Thời điểm (VN): {now_str}\n\n"
        "Ví dụ JSON: \n"
        f"{example_json}\n"
    )


def _invoke_codex(sample_json: str) -> Dict[str, Any]:
    codex_bin = shutil.which("codex")
    # Default required unless explicitly disabled via env
    require_codex = os.environ.get("BROKER_REQUIRE_CODEX", "1") == "1"
    if not codex_bin:
        if require_codex:
            raise SystemExit("Codex CLI not found but BROKER_REQUIRE_CODEX=1")
        print("[ai_overrides] Codex CLI not found; skipping AI overlay generation")
        return {}
    prompt = _build_prompt(sample_json)
    cmd = [
        codex_bin,
        "exec",
        "--skip-git-repo-check",
        "--yolo",
        "--model",
        "gpt-5",
        "-c",
        "tools.web_search=true",
        "-c",
        "reasoning_effort=high",
        "-",
    ]
    with tempfile.TemporaryDirectory(prefix="codex_ai_") as tmp:
        tmp_path = Path(tmp)
        gen_path = tmp_path / "policy_overrides.generated.json"
        proc = subprocess.run(
            cmd,
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            cwd=tmp_path,
        )
        output = proc.stdout.decode("utf-8", errors="replace")
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        debug_dir = OUT_DIR / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        raw_path = debug_dir / f"codex_policy_raw_{ts}.txt"
        raw_path.write_text(output, encoding="utf-8")
        if proc.returncode != 0:
            err_path = debug_dir / f"codex_policy_error_{ts}.txt"
            err_path.write_text(output, encoding="utf-8")
            if require_codex:
                raise SystemExit(f"Codex CLI failed with exit code {proc.returncode}; see {err_path}")
            print(f"[ai_overrides] Codex CLI failed (exit {proc.returncode}); skipping AI overlay. See {err_path}")
            return {}
        if not gen_path.exists():
            # Fallback: try to parse JSON from stdout and persist
            try:
                # naive extraction: take substring between first '{' and last '}'
                start = output.find('{')
                end = output.rfind('}')
                if start != -1 and end != -1 and end > start:
                    payload = json.loads(output[start:end+1])
                    gen_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
                else:
                    raise ValueError('no JSON braces found')
            except Exception as _exc_json:
                raise SystemExit("Codex run completed without generating policy_overrides.generated.json") from _exc_json
        return json.loads(gen_path.read_text(encoding="utf-8"))


def calibrate(*, write: bool = True, raw_overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    sample_path = CONFIG_DIR / "policy_default.json"
    if not sample_path.exists():
        raise SystemExit(f"Missing baseline policy at {sample_path}")
    sample = _strip_json_comments(sample_path.read_text(encoding="utf-8"))
    if raw_overrides is None:
        raw_overrides = _invoke_codex(sample)
        if not raw_overrides:
            return {}
    normalized = _normalise_overrides(raw_overrides)
    if write:
        _write_ai_overrides(normalized)
    return normalized


__all__ = ["calibrate"]
