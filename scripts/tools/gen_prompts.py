from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def repo_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur.parent, *cur.parents]:
        if (cand / ".git").exists():
            return cand
    return Path.cwd()

def scan_profiles(portfolios_dir: Path) -> List[str]:
    if not portfolios_dir.exists():
        return []
    names: List[str] = []
    for profile_dir in sorted(portfolios_dir.iterdir()):
        if not profile_dir.is_dir():
            continue
        if (profile_dir / "portfolio.csv").exists():
            names.append(profile_dir.name)
    return names

def render_template(template_text: str, profile: str) -> str:
    return template_text.replace("{{PROFILE}}", profile)


def main(argv: Optional[List[str]] = None) -> int:
    rroot = repo_root()
    parser = argparse.ArgumentParser(description="Generate per-profile prompts from template (replace only {{PROFILE}})")
    parser.add_argument("--template", default=str(rroot / "prompts" / "SAMPLE_PROMPT.txt"))
    parser.add_argument("--profiles", default=None, help="Comma-separated profile names; defaults to scanning data/portfolios")
    parser.add_argument("--portfolios-dir", default=str(rroot / "data" / "portfolios"))
    parser.add_argument("--outdir", default=str(rroot / "prompts"))
    args = parser.parse_args(argv)

    template_path = Path(args.template)
    portfolios_dir = Path(args.portfolios_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not template_path.exists():
        raise SystemExit(f"Template not found: {template_path}")
    template_text = template_path.read_text(encoding="utf-8")
    if args.profiles:
        profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    else:
        profiles = scan_profiles(portfolios_dir)
    if not profiles:
        print(
            "[prompts] No profiles found; create data/portfolios/<profile>/portfolio.csv first",
            file=sys.stderr,
        )
        return 2

    for profile in profiles:
        rendered = render_template(template_text, profile)
        dest = outdir / f"prompt_{profile}.txt"
        dest.write_text(rendered, encoding="utf-8")
        print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
