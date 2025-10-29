from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence


def repo_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur.parent, *cur.parents]:
        if (cand / ".git").exists():
            return cand
    return Path.cwd()

def main(argv: Optional[Sequence[str]] = None) -> int:
    rroot = repo_root()
    parser = argparse.ArgumentParser(
        description="Expose the canonical prompt file; optionally copy it to a destination."
    )
    parser.add_argument("--template", default=str(rroot / "prompts" / "PROMPT.txt"))
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional directory to copy the prompt into as 'prompt.txt'.",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Optional explicit destination path for the copied prompt.",
    )
    parser.add_argument(
        "--profiles",
        default=None,
        help="Deprecated; retained for compatibility but ignored.",
    )
    args = parser.parse_args(argv)

    template_path = Path(args.template)

    if not template_path.exists():
        raise SystemExit(f"Template not found: {template_path}")
    dest: Optional[Path]
    if args.dest and args.outdir:
        raise SystemExit("Specify either --dest or --outdir, not both")
    if args.dest:
        dest = Path(args.dest)
    elif args.outdir:
        dest = Path(args.outdir) / "prompt.txt"
    else:
        dest = None

    if dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(dest)
    else:
        print(template_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
