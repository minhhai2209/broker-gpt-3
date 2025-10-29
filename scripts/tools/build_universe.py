"""Utility to derive the trading universe from the industry map."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


class UniverseBuilderError(RuntimeError):
    """Raised when the universe CSV cannot be materialised."""


def build_universe(industry_map: Path, output: Path) -> Path:
    """Create ``output`` with the distinct tickers present in ``industry_map``.

    Parameters
    ----------
    industry_map:
        CSV file containing at least a ``Ticker`` column.
    output:
        Destination CSV path. The parent directory is created on demand.

    Returns
    -------
    Path
        The resolved ``output`` path.
    """

    industry_map = industry_map.resolve()
    output = output.resolve()

    if not industry_map.exists():
        raise UniverseBuilderError(f"Industry map not found: {industry_map}")

    try:
        df = pd.read_csv(industry_map)
    except Exception as exc:  # pragma: no cover - pandas will raise informative errors
        raise UniverseBuilderError(f"Failed to read industry map {industry_map}") from exc

    if "Ticker" not in df.columns:
        raise UniverseBuilderError("Industry map must contain a 'Ticker' column")

    tickers = (
        df["Ticker"].dropna().astype(str).str.strip().str.upper().loc[lambda s: s != ""]
    )
    unique = pd.Series(sorted(set(tickers)))
    if unique.empty:
        raise UniverseBuilderError("Industry map does not contain any valid tickers")

    output.parent.mkdir(parents=True, exist_ok=True)
    unique.to_frame(name="Ticker").to_csv(output, index=False)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build data/universe/vn100.csv from industry map")
    parser.add_argument(
        "--industry-map",
        type=Path,
        default=Path("data/industry_map.csv"),
        help="Source CSV with columns including Ticker",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/universe/vn100.csv"),
        help="Destination CSV path for the derived universe",
    )
    args = parser.parse_args(argv)

    build_universe(args.industry_map, args.output)
    print(f"[universe] Wrote {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

