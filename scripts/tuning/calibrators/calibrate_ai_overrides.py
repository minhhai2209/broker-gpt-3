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
from typing import Any, Dict, Tuple


BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_DIR = BASE_DIR / "config"
OUT_DIR = BASE_DIR / "out"
AI_OVERRIDES_PATH = CONFIG_DIR / "policy_ai_overrides.json"
AUDIT_PATH = OUT_DIR / "debug" / "policy_ai_overrides_audit.ndjson"


ALLOWED_TOP_LEVEL = {
    "rationale",
    "buy_budget_frac",
    "add_max",
    "new_max",
    "calibration_targets",
    "sector_bias",
    "ticker_bias",
    "execution",
}

ALLOWED_CALIBRATION_TARGETS = {
    "liquidity": {"adtv_multiple"},
    "market_filter": {
        "idx_drop_q",
        "vol_ann_q",
        "trend_floor_q",
        "atr_soft_q",
        "atr_hard_q",
        "ms_soft_q",
        "ms_hard_q",
        "dd_floor_q",
        "breadth_floor_q",
        "breadth_floor_half_life_days",
        "breadth_floor_min",
        "breadth_floor_max",
        "leader_rsi_q",
        "leader_mom_q",
    },
    "dynamic_caps": {"enp_total_target", "sector_limit_target"},
    "sizing": {"enp_target", "enp_target_add", "enp_target_new"},
    "thresholds": {"near_ceiling_q"},
}

ALLOWED_EXECUTION_FILL = {
    "horizon_s",
    "window_sigma_s",
    "window_vol_s",
    "target_prob",
    "max_chase_ticks",
    "cancel_ratio_per_min",
    "joiner_factor",
    "no_cross",
}

BIAS_RANGE = (-0.20, 0.20)
BUY_BUDGET_FRAC_RANGE = (0.02, 0.30)
NEW_ADD_RANGE = (0, 20)
MAX_CHASE_RANGE = (0, 2)
HORIZON_RANGE = (10, 240)
WINDOW_SIGMA_RANGE = (15, 240)
WINDOW_VOL_RANGE = (30, 300)
TARGET_PROB_RANGE = (0.0, 0.95)
CANCEL_RATIO_RANGE = (0.0, 0.90)
JOINER_RANGE = (0.0, 0.50)


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
    targets = _ensure_dict(raw, "calibration_targets")
    out: Dict[str, Dict[str, float]] = {}
    for section, allowed_keys in ALLOWED_CALIBRATION_TARGETS.items():
        if section not in targets:
            continue
        sec_raw = _ensure_dict(targets[section], f"calibration_targets.{section}")
        sec_out: Dict[str, float] = {}
        for key, value in sec_raw.items():
            if key not in allowed_keys:
                raise SystemExit(f"Unsupported calibration_targets.{section}.{key}")
            try:
                sec_out[key] = float(value)
            except Exception as exc:
                raise SystemExit(
                    f"calibration_targets.{section}.{key} must be numeric"
                ) from exc
        out[section] = sec_out
    for section in targets.keys():
        if section not in ALLOWED_CALIBRATION_TARGETS:
            raise SystemExit(f"Unsupported calibration_targets section '{section}'")
    return out


def _normalise_execution(raw: Any) -> Dict[str, Any]:
    execution = _ensure_dict(raw, "execution")
    out: Dict[str, Any] = {}
    if "fill" in execution:
        fill_raw = _ensure_dict(execution["fill"], "execution.fill")
        # Provide an explicit diagnostic for a common mistake: nesting under a 'base' alias.
        if "base" in fill_raw:
            raise SystemExit(
                "Unsupported alias 'execution.fill.base'; use flat execution.fill keys: "
                "horizon_s, window_sigma_s, window_vol_s, target_prob, max_chase_ticks, "
                "cancel_ratio_per_min, joiner_factor, no_cross"
            )
        fill_out: Dict[str, Any] = {}
        for key, value in fill_raw.items():
            if key not in ALLOWED_EXECUTION_FILL:
                raise SystemExit(f"Unsupported execution.fill.{key}")
            if key == "no_cross":
                fill_out[key] = bool(value)
            elif key == "max_chase_ticks":
                try:
                    fill_out[key] = _clamp_int(value, MAX_CHASE_RANGE)
                except Exception as exc:
                    raise SystemExit("execution.fill.max_chase_ticks must be int") from exc
            elif key in {"horizon_s", "window_sigma_s", "window_vol_s"}:
                try:
                    ivalue = int(value)
                except Exception as exc:
                    raise SystemExit(f"execution.fill.{key} must be int") from exc
                bounds = (
                    HORIZON_RANGE if key == "horizon_s"
                    else WINDOW_SIGMA_RANGE if key == "window_sigma_s"
                    else WINDOW_VOL_RANGE
                )
                fill_out[key] = _clamp_int(ivalue, bounds)
            elif key == "target_prob":
                try:
                    fill_out[key] = _clamp(float(value), TARGET_PROB_RANGE)
                except Exception as exc:
                    raise SystemExit("execution.fill.target_prob must be numeric") from exc
            elif key == "cancel_ratio_per_min":
                try:
                    fill_out[key] = _clamp(float(value), CANCEL_RATIO_RANGE)
                except Exception as exc:
                    raise SystemExit("execution.fill.cancel_ratio_per_min must be numeric") from exc
            elif key == "joiner_factor":
                try:
                    fill_out[key] = _clamp(float(value), JOINER_RANGE)
                except Exception as exc:
                    raise SystemExit("execution.fill.joiner_factor must be numeric") from exc
        out["fill"] = fill_out
    for key in execution.keys():
        if key != "fill":
            raise SystemExit(f"Unsupported execution key '{key}'")
    return out


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
        elif key == "buy_budget_frac":
            try:
                out[key] = _clamp(float(value), BUY_BUDGET_FRAC_RANGE)
            except Exception as exc:
                raise SystemExit("buy_budget_frac must be numeric") from exc
        elif key in {"add_max", "new_max"}:
            try:
                out[key] = _clamp_int(value, NEW_ADD_RANGE)
            except Exception as exc:
                raise SystemExit(f"{key} must be integer") from exc
        elif key == "calibration_targets":
            out[key] = _normalise_calibration_targets(value)
        elif key == "sector_bias":
            out[key] = _normalise_bias_map(value, context="sector_bias")
        elif key == "ticker_bias":
            out[key] = _normalise_bias_map(value, context="ticker_bias")
        elif key == "execution":
            exec_norm = _normalise_execution(value)
            if exec_norm:
                out[key] = exec_norm
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


def _build_prompt(sample_json: str) -> str:
    allowed = "\n".join(f"- {k}" for k in sorted(ALLOWED_TOP_LEVEL))
    vn_tz = timezone(timedelta(hours=7))
    now_str = datetime.now(vn_tz).strftime("%Y-%m-%d %H:%M:%S %Z%z")
    return f"""
Bạn là chuyên gia điều chỉnh policy giao dịch cho Broker GPT.

Yêu cầu:
- Chỉ xuất JSON tại file policy_overrides.generated.json chứa các khóa sau:
{allowed}
- Các khóa lồng nhau chỉ sử dụng đúng whitelist trong đặc tả (liquidity/market_filter/dynamic_caps/sizing/thresholds, execution.fill,...).
- Mọi giá trị phải nằm trong guardrail: buy_budget_frac ∈ [0.02,0.30], add_max/new_max ∈ [0,20], bias ∈ [-0.20,0.20], execution.fill.* tuân thủ clamp.
- Luôn cung cấp "rationale" (chuỗi, nêu rõ bối cảnh/tin tức, TTL, tác động tới rủi ro).
- Tuyệt đối KHÔNG ghi thêm khóa ngoài whitelist.
- Nếu không cần thay đổi, vẫn ghi rationale và đặt khóa rỗng (hoặc bỏ qua) theo chuẩn JSON object.

 Ràng buộc QUAN TRỌNG cho execution.fill:
 - Không được lồng thêm cấp alias như "base"; mọi khóa phải phẳng dưới execution.fill.
 - Danh sách khóa hợp lệ của execution.fill: horizon_s, window_sigma_s, window_vol_s, target_prob, max_chase_ticks, cancel_ratio_per_min, joiner_factor, no_cross.
 - Ví dụ hợp lệ:
   "execution": {{
     "fill": {{
       "horizon_s": 75,
       "window_sigma_s": 75,
       "window_vol_s": 120,
       "target_prob": 0.55,
       "max_chase_ticks": 2,
       "cancel_ratio_per_min": 0.15,
       "joiner_factor": 0.25,
       "no_cross": true
     }}
   }}

Thời điểm (VN): {now_str}

Schema tham chiếu (rút gọn):
{sample_json}
"""


def _invoke_codex(sample_json: str) -> Dict[str, Any]:
    codex_bin = shutil.which("codex")
    if not codex_bin:
        if os.environ.get("BROKER_REQUIRE_CODEX") == "1":
            raise SystemExit("Codex CLI not found but BROKER_REQUIRE_CODEX=1")
        print("[ai_overrides] Codex CLI not found; skipping AI overlay generation")
        return {}
    prompt = _build_prompt(sample_json)
    cmd = [
        codex_bin,
        "exec",
        "--skip-git-repo-check",
        "--full-auto",
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
            raise SystemExit(f"Codex CLI failed with exit code {proc.returncode}; see {err_path}")
        if not gen_path.exists():
            raise SystemExit("Codex run completed without generating policy_overrides.generated.json")
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
