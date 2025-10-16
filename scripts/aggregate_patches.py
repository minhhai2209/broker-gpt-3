#!/usr/bin/env python3
"""Merge policy patches from multiple calibrators.

The runtime policy exposed to the order engine is derived from:

* The merged baseline policy (``out/orders/policy_overrides.json``)
* Optional calibrator patches located under ``out/orders``

This module applies the merge contract described in
``specs/ml_calibrator_integration_spec.txt``.  It can be executed as a
standalone script or imported (``aggregate_to_runtime``) by other
modules such as the order engine runner.

Key behaviours:

* Budgets and limits are merged via ``min`` across sources after
  clamping to guardrails.
* Bias contributions are summed then clamped to ``[-0.2, 0.2]``.
* Execution knobs from calibrators are surfaced via a dedicated
  ``_runtime_overrides`` block without mutating the baseline schema.
* Patches with expired TTLs are ignored.  When ``out/orders/.policy_lock``
  exists the previously committed runtime policy is kept verbatim.
* A structured NDJSON log is appended to
  ``out/debug/policy_merge.log`` for auditability.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "out"
OUT_ORDERS_DIR = OUT_DIR / "orders"
OUT_DEBUG_DIR = OUT_DIR / "debug"

PATCH_PATHS = (
    OUT_ORDERS_DIR / "patch_market.json",
    OUT_ORDERS_DIR / "patch_tune.json",
    OUT_ORDERS_DIR / "patch_ml.json",
)

RUNTIME_PATH = OUT_ORDERS_DIR / "policy_runtime.json"
MERGE_LOG_PATH = OUT_DEBUG_DIR / "policy_merge.log"
LOCK_PATH = OUT_ORDERS_DIR / ".policy_lock"

CLAMP_BUDGET = (0.0, 0.30)
CLAMP_BIAS = (-0.20, 0.20)


@dataclass
class PatchInfo:
    """Container describing a patch prior to merge."""

    path: Path
    payload: Dict[str, object]
    source: str
    ttl: Optional[datetime]
    valid: bool
    reason: Optional[str] = None


class PatchMergeError(RuntimeError):
    """Raised when merge cannot be completed in a recoverable manner."""


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    try:
        text = path.read_text(encoding="utf-8")
        return json.loads(text) if text.strip() else {}
    except json.JSONDecodeError as exc:  # pragma: no cover - explicit message
        raise PatchMergeError(f"Invalid JSON in {path}: {exc}") from exc


def _load_patch(path: Path, now: datetime) -> PatchInfo:
    if not path.exists():
        return PatchInfo(path=path, payload={}, source=str(path.name), ttl=None, valid=False, reason="missing")
    try:
        payload = _load_json(path)
    except PatchMergeError as exc:
        return PatchInfo(path=path, payload={}, source=str(path.name), ttl=None, valid=False, reason=str(exc))
    meta = payload.get("meta") if isinstance(payload, dict) else None
    source = "unknown"
    ttl: Optional[datetime] = None
    if isinstance(meta, dict):
        source = str(meta.get("source") or source)
        ttl_raw = meta.get("ttl")
        if ttl_raw:
            try:
                ttl = datetime.fromisoformat(str(ttl_raw))
            except ValueError:
                return PatchInfo(
                    path=path,
                    payload=payload,
                    source=source,
                    ttl=None,
                    valid=False,
                    reason=f"invalid ttl {ttl_raw!r}",
                )
    if ttl and ttl.tzinfo is None:
        ttl = ttl.replace(tzinfo=timezone.utc)
    valid = True
    reason = None
    if ttl and now > ttl:
        valid = False
        reason = "expired"
    return PatchInfo(path=path, payload=payload, source=source, ttl=ttl, valid=valid, reason=reason)


def _clamp(value: float, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    return max(lo, min(hi, float(value)))


def _ensure_dirs() -> None:
    OUT_ORDERS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def _baseline_policy(path: Path) -> Dict[str, object]:
    try:
        return _load_json(path)
    except FileNotFoundError as exc:
        raise PatchMergeError(
            "Missing baseline runtime policy at out/orders/policy_overrides.json. "
            "Ensure ensure_policy_override_file() has been executed first."
        ) from exc


def _merge_budget(
    base_value: Optional[float],
    patches: Iterable[PatchInfo],
    winners: List[Dict[str, object]],
) -> Optional[float]:
    candidates: List[Tuple[float, str]] = []
    if base_value is not None:
        candidates.append((_clamp(float(base_value), CLAMP_BUDGET), "baseline"))
    for patch in patches:
        if not patch.valid:
            continue
        try:
            raw = patch.payload.get("set", {}) if isinstance(patch.payload, dict) else {}
            if isinstance(raw, dict) and "buy_budget_frac" in raw:
                candidates.append((_clamp(float(raw["buy_budget_frac"]), CLAMP_BUDGET), patch.source))
        except Exception:
            continue
    if not candidates:
        return base_value
    winner_value, winner_source = min(candidates, key=lambda x: x[0])
    winners.append(
        {
            "key": "buy_budget_frac",
            "old": float(base_value) if base_value is not None else None,
            "candidate": winner_value,
            "source_winner": winner_source,
            "rule": "min",
        }
    )
    return winner_value


def _merge_limit(
    key: str,
    base_value: Optional[int],
    patches: Iterable[PatchInfo],
    winners: List[Dict[str, object]],
) -> Optional[int]:
    candidates: List[Tuple[int, str]] = []
    if base_value is not None:
        candidates.append((int(base_value), "baseline"))
    for patch in patches:
        if not patch.valid:
            continue
        payload = patch.payload
        if not isinstance(payload, dict):
            continue
        limits = payload.get("limits")
        if not isinstance(limits, dict):
            continue
        if key not in limits:
            continue
        try:
            candidates.append((int(limits[key]), patch.source))
        except Exception:
            continue
    if not candidates:
        return base_value
    winner_value, winner_source = min(candidates, key=lambda x: x[0])
    winners.append(
        {
            "key": key,
            "old": int(base_value) if base_value is not None else None,
            "candidate": winner_value,
            "source_winner": winner_source,
            "rule": "min",
        }
    )
    return winner_value


def _merge_biases(
    bias_map: Dict[str, float],
    patch_biases: Dict[str, float],
    winners: List[Dict[str, object]],
    source: str,
) -> None:
    for key, value in patch_biases.items():
        try:
            delta = float(value)
        except Exception:
            continue
        old_val = float(bias_map.get(key, 0.0) or 0.0)
        new_val = _clamp(old_val + delta, CLAMP_BIAS)
        bias_map[key] = new_val
        winners.append(
            {
                "key": key,
                "old": old_val,
                "candidate": new_val,
                "source_winner": source,
                "rule": "sum+clamp",
            }
        )


def _merge_exec(overrides: Dict[str, float], exec_patch: Dict[str, object], source: str, winners: List[Dict[str, object]]) -> None:
    if not isinstance(exec_patch, dict):
        return
    for key, value in exec_patch.items():
        try:
            val = float(value)
        except Exception:
            continue
        old_val = overrides.get(key)
        if key.endswith("_cap"):
            new_val = min(val if old_val is None else old_val, val)
            rule = "min"
        else:
            new_val = max(val if old_val is None else old_val, val)
            rule = "max"
        overrides[key] = new_val
        winners.append(
            {
                "key": f"exec.{key}",
                "old": old_val,
                "candidate": new_val,
                "source_winner": source,
                "rule": rule,
            }
        )


def aggregate_policy(
    base_policy: Dict[str, object],
    patches: Iterable[PatchInfo],
    *,
    now: Optional[datetime] = None,
) -> Tuple[Dict[str, object], Dict[str, object], List[Dict[str, object]]]:
    """Merge patches into a runtime-ready policy.

    Returns ``(runtime_policy, runtime_details, winners_log)``.
    ``runtime_details`` is serialisable and stored under ``_runtime_overrides``
    for downstream consumers.
    """

    now = now or datetime.now(timezone.utc)
    winners: List[Dict[str, object]] = []

    # 1. Budget + limits
    buy_budget = base_policy.get("buy_budget_frac")
    merged_budget = _merge_budget(buy_budget, patches, winners)
    if merged_budget is not None:
        base_policy["buy_budget_frac"] = merged_budget

    for limit_key in ("new_max", "add_max"):
        base_policy_val = base_policy.get(limit_key)
        merged_limit = _merge_limit(limit_key, base_policy_val, patches, winners)
        if merged_limit is not None:
            base_policy[limit_key] = merged_limit

    # 2. Biases (sector/ticker). Represent keys as dotted prefix inside runtime block.
    sector_bias = dict(base_policy.get("sector_bias", {}) or {})
    ticker_bias = dict(base_policy.get("ticker_bias", {}) or {})
    bias_record: Dict[str, float] = {}
    for patch in patches:
        if not patch.valid:
            continue
        patch_bias = patch.payload.get("bias") if isinstance(patch.payload, dict) else None
        if not isinstance(patch_bias, dict):
            continue
        for key, value in patch_bias.items():
            try:
                delta = float(value)
            except Exception:
                continue
            if key.startswith("sector_bias."):
                sector = key.split(".", 1)[1]
                old_val = float(sector_bias.get(sector, 0.0) or 0.0)
                new_val = _clamp(old_val + delta, CLAMP_BIAS)
                sector_bias[sector] = new_val
                bias_record[f"sector_bias.{sector}"] = new_val
                winners.append(
                    {
                        "key": f"sector_bias.{sector}",
                        "old": old_val,
                        "candidate": new_val,
                        "source_winner": patch.source,
                        "rule": "sum+clamp",
                    }
                )
            elif key.startswith("ticker_bias."):
                ticker = key.split(".", 1)[1]
                old_val = float(ticker_bias.get(ticker, 0.0) or 0.0)
                new_val = _clamp(old_val + delta, CLAMP_BIAS)
                ticker_bias[ticker] = new_val
                bias_record[f"ticker_bias.{ticker}"] = new_val
                winners.append(
                    {
                        "key": f"ticker_bias.{ticker}",
                        "old": old_val,
                        "candidate": new_val,
                        "source_winner": patch.source,
                        "rule": "sum+clamp",
                    }
                )
    base_policy["sector_bias"] = sector_bias
    base_policy["ticker_bias"] = ticker_bias

    # 3. Gate merge (last-write wins for diagnostics, actual gating handled downstream)
    gate: Dict[str, Dict[str, object]] = {}
    for patch in patches:
        if not patch.valid:
            continue
        payload = patch.payload
        if not isinstance(payload, dict):
            continue
        gate_patch = payload.get("gate")
        if not isinstance(gate_patch, dict):
            continue
        for ticker, obj in gate_patch.items():
            if not isinstance(obj, dict):
                continue
            t_key = str(ticker)
            entry = gate.setdefault(t_key, {"sources": []})
            entry["sources"].append(patch.source)
            if "p_succ" in obj:
                try:
                    entry["p_succ"] = float(obj.get("p_succ"))
                except Exception:
                    pass
            if "p_gate" in obj:
                try:
                    entry["p_gate"] = float(obj.get("p_gate"))
                except Exception:
                    pass
            if "decision" in obj:
                decision = str(obj.get("decision")).lower()
                if decision == "block":
                    entry["decision"] = "block"
                    if obj.get("reason"):
                        entry.setdefault("reasons", []).append(str(obj.get("reason")))
                elif decision == "allow" and "decision" not in entry:
                    entry["decision"] = "allow"
            if "allow" in obj and isinstance(obj.get("allow"), bool):
                if not obj.get("allow"):
                    entry["decision"] = "block"
            if obj.get("reason") and obj.get("reason") not in entry.get("reasons", []):
                entry.setdefault("reasons", []).append(str(obj.get("reason")))

    # 4. Execution overrides aggregated for downstream consumption
    exec_overrides: Dict[str, float] = {}
    for patch in patches:
        if not patch.valid:
            continue
        payload = patch.payload
        if not isinstance(payload, dict):
            continue
        exec_patch = payload.get("exec")
        if not isinstance(exec_patch, dict):
            continue
        _merge_exec(exec_overrides, exec_patch, patch.source, winners)

    # 5. Guard flags (boolean OR across patches)
    guard_flags: Dict[str, bool] = {}
    for patch in patches:
        if not patch.valid:
            continue
        guard_patch = patch.payload.get("guard") if isinstance(patch.payload, dict) else None
        if not isinstance(guard_patch, dict):
            continue
        for key, value in guard_patch.items():
            if isinstance(value, bool):
                guard_flags[key] = bool(guard_flags.get(key)) or value

    metadata: Dict[str, object] = {}
    for patch in patches:
        if not patch.valid:
            continue
        meta = patch.payload.get("meta") if isinstance(patch.payload, dict) else None
        if isinstance(meta, dict):
            metadata[patch.source] = meta

    runtime_details = {
        "timestamp": now.isoformat(),
        "gate": gate,
        "exec": exec_overrides,
        "bias": bias_record,
        "guard": guard_flags,
        "meta": metadata,
    }
    return base_policy, runtime_details, winners


def _serialise_runtime(policy: Dict[str, object], runtime_details: Dict[str, object]) -> Dict[str, object]:
    merged = dict(policy)
    merged["_runtime_overrides"] = runtime_details
    return merged


def aggregate_to_runtime(
    *,
    now: Optional[datetime] = None,
    baseline_path: Optional[Path] = None,
    patch_paths: Iterable[Path] = PATCH_PATHS,
) -> Path:
    """Public helper used by the order engine.

    Returns the path to ``policy_runtime.json``.  Raises ``PatchMergeError``
    when the baseline policy cannot be loaded.
    """

    _ensure_dirs()
    now = now or datetime.now(timezone.utc)

    if LOCK_PATH.exists() and RUNTIME_PATH.exists():
        # Lock engaged: keep existing runtime snapshot
        _append_log(
            {
                "timestamp": now.isoformat(),
                "action": "lock_active",
                "lock_path": str(LOCK_PATH),
                "runtime_kept": str(RUNTIME_PATH),
            }
        )
        return RUNTIME_PATH

    baseline_path = baseline_path or (OUT_ORDERS_DIR / "policy_overrides.json")
    base_policy = _baseline_policy(baseline_path)

    patch_infos: List[PatchInfo] = []
    for path in patch_paths:
        patch_infos.append(_load_patch(path, now))

    # Filter valid patches only for merge operations
    valid_patches = [p for p in patch_infos if p.valid and isinstance(p.payload, dict) and p.payload]

    runtime_policy, runtime_details, winners = aggregate_policy(base_policy, valid_patches, now=now)
    runtime_details["sources"] = [p.source for p in valid_patches]
    runtime_details["patch_count"] = len(valid_patches)

    payload = _serialise_runtime(runtime_policy, runtime_details)
    RUNTIME_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _append_log(
        {
            "timestamp": now.isoformat(),
            "baseline": str(baseline_path),
            "runtime": str(RUNTIME_PATH),
            "patches": [
                {
                    "path": str(info.path),
                    "source": info.source,
                    "valid": info.valid,
                    "reason": info.reason,
                    "ttl": info.ttl.isoformat() if info.ttl else None,
                }
                for info in patch_infos
            ],
            "winners": winners,
            "result": {
                "buy_budget_frac": runtime_policy.get("buy_budget_frac"),
                "add_max": runtime_policy.get("add_max"),
                "new_max": runtime_policy.get("new_max"),
                "gate_size": len(runtime_details.get("gate", {})),
            },
        }
    )
    return RUNTIME_PATH


def _append_log(record: Dict[str, object]) -> None:
    OUT_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    with MERGE_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    try:
        aggregate_to_runtime()
    except PatchMergeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
