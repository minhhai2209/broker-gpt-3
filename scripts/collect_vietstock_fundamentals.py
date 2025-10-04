#!/usr/bin/env python3
"""Fetch Vietstock financial ratios for VN100 tickers using Playwright."""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

import pandas as pd

BASE_URL = "https://finance.vietstock.vn/{ticker}/tai-chinh.htm?tab=BCTT"
# Default output placed in tracked data/ so results có thể commit/cache
DEFAULT_OUT = Path("data/fundamentals_vietstock.csv")
DEFAULT_INDUSTRY_MAP = Path("data/industry_map.csv")

SKIP_LABELS = {
    "Giai đoạn",
    "Hợp nhất",
    "Công ty kiểm toán",
    "Ý kiến kiểm toán",
    "Kiểm toán",
}


def load_tickers(industry_map: Path, limit: Optional[int] = None) -> List[str]:
    df = pd.read_csv(industry_map)
    tickers = df["Ticker"].astype(str).str.upper().unique().tolist()
    if limit is not None:
        tickers = tickers[:limit]
    return tickers


NUMERIC_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def parse_value(text: str) -> Optional[float]:
    txt = text.strip()
    if not txt or txt in {"--", "-", "N/A"}:
        return None
    normalized = txt.replace(",", "")
    if normalized.endswith("%"):
        normalized = normalized[:-1]
    if not NUMERIC_PATTERN.match(normalized):
        return None
    value = float(normalized)
    return value


@dataclass
class MetricRecord:
    ticker: str
    metric: str
    unit: str
    period: str
    value: Optional[float]
    raw: str


def parse_ratios(html: str, ticker: str) -> List[MetricRecord]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="tbl-data-BCTT-CSTC")
    if table is None:
        raise RuntimeError("Không tìm thấy bảng chỉ số tài chính")

    header_th = [th.get_text(strip=True) for th in table.find("thead").find_all("th")]
    records: List[MetricRecord] = []

    tbody = table.find("tbody")
    if tbody is None:
        return records

    for row in tbody.find_all("tr"):
        cols = row.find_all("td")
        if not cols:
            continue
        label = cols[0].get_text(strip=True)
        if not label or label in SKIP_LABELS:
            continue
        unit = cols[2].get_text(strip=True) if len(cols) > 2 else ""
        value_cells = cols[4:]
        if not value_cells:
            continue
        periods = [h for h in header_th[-len(value_cells):]]
        for period, cell in zip(periods, value_cells):
            raw = cell.get_text(strip=True)
            value = parse_value(raw)
            record = MetricRecord(
                ticker=ticker,
                metric=label,
                unit=unit,
                period=period,
                value=value,
                raw=raw,
            )
            records.append(record)
    return records


def fetch_ratios(page, ticker: str, delay: float) -> List[MetricRecord]:
    url = BASE_URL.format(ticker=ticker)
    page.goto(url, wait_until="domcontentloaded", timeout=60000)
    # Try to stabilize network and dynamic rendering
    try:
        page.wait_for_load_state("networkidle", timeout=60000)
    except Exception:
        pass
    # Robust selector wait with fallbacks
    selectors = [
        "#tbl-data-BCTT-CSTC tbody tr",
        "table#tbl-data-BCTT-CSTC tr",
        "table:has(#tbl-data-BCTT-CSTC) tr",
        "table:has(thead) tbody tr",
    ]
    ok = False
    for sel in selectors:
        try:
            page.wait_for_selector(sel, timeout=45000)
            ok = True
            break
        except Exception:
            continue
    if not ok:
        # Dump a small HTML excerpt for diagnostics and raise
        html = page.content()
        snippet = html[:1000]
        raise RuntimeError(f"Không tìm thấy bảng chỉ số tài chính cho {ticker}; HTML head: {snippet[:200]}")
    page.wait_for_timeout(2000)
    records = parse_ratios(page.content(), ticker)
    if delay:
        time.sleep(delay)
    return records


def write_csv(records: Iterable[MetricRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ticker", "metric", "unit", "period", "value", "raw"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def run(tickers: List[str], out_path: Path, delay: float) -> None:
    all_records: List[MetricRecord] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        for idx, ticker in enumerate(tickers, start=1):
            records = fetch_ratios(page, ticker, delay)
            all_records.extend(records)
            print(f"[{idx}/{len(tickers)}] {ticker}: {len(records)} điểm dữ liệu")
        browser.close()
    write_csv(all_records, out_path)
    print(f"Đã ghi {len(all_records)} dòng vào {out_path}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Thu thập chỉ số tài chính từ Vietstock")
    parser.add_argument("--tickers", nargs="*", help="Danh sách mã cụ thể. Nếu bỏ qua sẽ dùng data/industry_map.csv")
    parser.add_argument("--industry-map", type=Path, default=DEFAULT_INDUSTRY_MAP, help="Đường dẫn file industry_map.csv")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số mã xử lý")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="File CSV đầu ra")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay (giây) giữa các lần gọi để tránh chặn")
    args = parser.parse_args(argv)

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = load_tickers(args.industry_map, args.limit)

    if not tickers:
        print("Không có mã nào để xử lý", file=sys.stderr)
        return 1

    run(tickers, args.out, args.delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
