from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def repo_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur.parent, *cur.parents]:
        if (cand / ".git").exists():
            return cand
    return Path.cwd()


def load_presets(config_path: Path) -> Dict[str, Dict[str, object]]:
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    presets = cfg.get("presets", {}) or {}
    if not isinstance(presets, dict) or not presets:
        raise SystemExit("No presets found in config")
    return presets


def scan_profiles(portfolios_dir: Path) -> List[str]:
    if not portfolios_dir.exists():
        return []
    names = []
    for f in sorted(portfolios_dir.glob("*.csv")):
        if f.is_file():
            names.append(f.stem)
    return names


def build_file_list(profile: str, preset_names: List[str]) -> str:
    lines: List[str] = []
    i = 1
    lines.append(
        f"{i}) out/market/technical_snapshot.csv — ảnh chụp kỹ thuật theo mã (giá hiện tại, thay đổi %, SMA/EMA/RSI/ATR, MACD, Z‑score, returns, ADV, 52w range)"
    )
    for p in preset_names:
        i += 1
        lines.append(f"{i}) out/presets/{p}.csv — mức bậc mua/bán theo preset {p}")
    i += 1
    lines.append(f"{i}) data/portfolios/{profile}.csv — danh mục hiện tại (Ticker, Quantity, AvgPrice)")
    i += 1
    lines.append(f"{i}) out/portfolios/{profile}_positions.csv — PnL theo mã, MarketValue/CostBasis/Unrealized")
    i += 1
    lines.append(f"{i}) out/portfolios/{profile}_sector.csv — tổng hợp PnL theo ngành")
    i += 1
    lines.append(f"{i}) data/order_history/{profile}_fills.csv — các lệnh đã khớp trong hôm nay")
    return "\n".join(lines)


def build_preset_descriptions(presets: Dict[str, Dict[str, object]]) -> str:
    lines: List[str] = []
    for name in sorted(presets.keys()):
        raw = presets[name] or {}
        desc = raw.get("description") if isinstance(raw, dict) else None
        if not isinstance(desc, str) or not desc.strip():
            desc = "(chưa có mô tả; sử dụng các mức Buy_i/Sell_i trong file preset)"
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def render_template(template_text: str, profile: str, presets: Dict[str, Dict[str, object]]) -> str:
    text = template_text
    preset_names = sorted(presets.keys())
    file_list = build_file_list(profile, preset_names)
    preset_desc = build_preset_descriptions(presets)
    text = text.replace("{{PROFILE}}", profile)
    text = text.replace("{{FILE_LIST}}", file_list)
    text = text.replace("{{PRESET_DESCRIPTIONS}}", preset_desc)
    return text


def main(argv: Optional[List[str]] = None) -> int:
    rroot = repo_root()
    parser = argparse.ArgumentParser(description="Generate per-profile prompts from template")
    parser.add_argument("--template", default=str(rroot / "prompts" / "SAMPLE_PROMPT.txt"))
    parser.add_argument("--config", default=str(rroot / "config" / "data_engine.yaml"))
    parser.add_argument("--profiles", default=None, help="Comma-separated profile names; defaults to scanning data/portfolios")
    parser.add_argument("--portfolios-dir", default=str(rroot / "data" / "portfolios"))
    parser.add_argument("--outdir", default=str(rroot / "prompts"))
    args = parser.parse_args(argv)

    template_path = Path(args.template)
    config_path = Path(args.config)
    portfolios_dir = Path(args.portfolios_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not template_path.exists():
        raise SystemExit(f"Template not found: {template_path}")
    template_text = template_path.read_text(encoding="utf-8")
    presets = load_presets(config_path)

    if args.profiles:
        profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    else:
        profiles = scan_profiles(portfolios_dir)
    if not profiles:
        print("[prompts] No profiles found; create data/portfolios/<profile>.csv first", file=sys.stderr)
        return 2

    for profile in profiles:
        rendered = render_template(template_text, profile, presets)
        dest = outdir / f"prompt_{profile}.txt"
        dest.write_text(rendered, encoding="utf-8")
        print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

