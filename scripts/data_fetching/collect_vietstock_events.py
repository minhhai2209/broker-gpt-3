#!/usr/bin/env python3
"""Fetch corporate events for VN100 tickers from Vietstock.

This utility navigates to the Vietstock events page per ticker and parses the
events table using Playwright (Chromium, headless). It then writes a compact
events calendar CSV used by ``calendar_loader.load_events``.

Why Playwright here instead of jsdom:
- Vietstock renders the events table via client JavaScript with cross‑origin
  requests to their datacenter. A plain HTML fetch or lightweight DOM emulation
  often misses the populated table. Playwright gives us a reliable, low‑friction
  renderer already available in this repo's Python toolchain.

Output schema (columns):
- Ticker (str, upper)
- Date (date) — chosen as Ex‑rights date if present, else Record date, else
  Execution date
- Type (str) — Vietstock "Loại Sự kiện"

Additional columns are included for audit/troubleshooting:
- Exchange, ExDate, RecordDate, ExecutionDate, Content, SourceURL

Fail‑fast policy: if the DOM structure cannot be located for a ticker, the run
aborts with a clear error message and non‑zero exit code.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from playwright.sync_api import sync_playwright

import pandas as pd

DEFAULT_OUT = Path("data/events_calendar.csv")
DEFAULT_INDUSTRY_MAP = Path("data/industry_map.csv")


def load_tickers(industry_map: Path, limit: Optional[int] = None) -> List[str]:
    df = pd.read_csv(industry_map)
    tickers = (
        df["Ticker"].astype(str).str.upper().unique().tolist()
    )
    # Exclude index labels if present
    tickers = [t for t in tickers if t not in {"VNINDEX", "VN30", "VN100"}]
    if limit is not None:
        tickers = tickers[:limit]
    return tickers


def _parse_vn_date(s: str) -> Optional[str]:
    s = (s or "").strip()
    if not s or s in {"--", "-", "N/A"}:
        return None
    # Expected dd/mm/yyyy
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            d = datetime.strptime(s, fmt).date()
            return d.isoformat()
        except Exception:
            continue
    return None


@dataclass
class EventRow:
    ticker: str
    exchange: str
    ex_date: Optional[str]
    record_date: Optional[str]
    execution_date: Optional[str]
    content: str
    type_label: str
    source_url: str

    def main_date(self) -> Optional[str]:
        return self.ex_date or self.record_date or self.execution_date


def parse_table_rows(page, ticker: str, source_url: str) -> List[EventRow]:
    # The events table lives inside a container whose class contains "event".
    # Header columns observed: STT | Mã CK | Sàn | Ngày GDKHQ | Ngày ĐKCC | Ngày thực hiện | Nội dung sự kiện | Loại Sự kiện
    table = page.query_selector("div[class*='event'] table")
    if table is None:
        raise RuntimeError(f"Không tìm thấy bảng sự kiện cho {ticker} — selector div[class*='event'] table")
    trs = table.query_selector_all("tbody tr") or []
    rows: List[EventRow] = []
    for tr in trs:
        tds = tr.query_selector_all("td")
        if not tds or len(tds) < 8:
            continue
        # Columns by index (0‑based): 0 STT, 1 Mã CK, 2 Sàn, 3 GDKHQ, 4 ĐKCC, 5 Thực hiện, 6 Nội dung, 7 Loại
        code = (tds[1].text_content() or "").strip().upper()
        exch = (tds[2].text_content() or "").strip()
        ex_date = _parse_vn_date(tds[3].text_content())
        record_date = _parse_vn_date(tds[4].text_content())
        execution_date = _parse_vn_date(tds[5].text_content())
        content = (tds[6].text_content() or "").strip()
        type_label = (tds[7].text_content() or "").strip()
        if not code:
            continue
        rows.append(
            EventRow(
                ticker=code,
                exchange=exch,
                ex_date=ex_date,
                record_date=record_date,
                execution_date=execution_date,
                content=content,
                type_label=type_label,
                source_url=source_url,
            )
        )
    return rows


def fetch_events_for_ticker(page, ticker: str, delay: float) -> List[EventRow]:
    url = f"https://finance.vietstock.vn/lich-su-kien.htm?page=1&tab=1&code={ticker}"
    page.goto(url, wait_until="domcontentloaded", timeout=60000)
    try:
        page.wait_for_load_state("networkidle", timeout=60000)
    except Exception:
        pass
    # Wait for the table container to render
    page.wait_for_selector("div[class*='event'] table", timeout=30000)
    rows = parse_table_rows(page, ticker, url)
    if delay:
        page.wait_for_timeout(int(delay * 1000))
    return rows


def write_csv(rows: Iterable[EventRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Ticker",
        "Date",
        "Type",
        "Exchange",
        "ExDate",
        "RecordDate",
        "ExecutionDate",
        "Content",
        "SourceURL",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "Ticker": r.ticker,
                    "Date": r.main_date() or "",
                    "Type": r.type_label,
                    "Exchange": r.exchange,
                    "ExDate": r.ex_date or "",
                    "RecordDate": r.record_date or "",
                    "ExecutionDate": r.execution_date or "",
                    "Content": r.content,
                    "SourceURL": r.source_url,
                }
            )


def run(tickers: List[str], out_path: Path, delay: float) -> None:
    all_rows: List[EventRow] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        for idx, t in enumerate(tickers, start=1):
            rows = fetch_events_for_ticker(page, t, delay)
            all_rows.extend(rows)
            print(f"[{idx}/{len(tickers)}] {t}: {len(rows)} sự kiện")
        browser.close()
    # Fail fast if nothing collected
    if not all_rows:
        raise SystemExit("Không thu được sự kiện nào — kiểm tra DOM selector hoặc kết nối mạng")
    write_csv(all_rows, out_path)
    print(f"Đã ghi {len(all_rows)} dòng vào {out_path}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Tải sự kiện doanh nghiệp từ Vietstock cho rổ VN100")
    ap.add_argument("--tickers", nargs="*", help="Danh sách mã cụ thể (mặc định: đọc từ data/industry_map.csv)")
    ap.add_argument("--industry-map", type=Path, default=DEFAULT_INDUSTRY_MAP)
    ap.add_argument("--limit", type=int, default=None, help="Giới hạn số mã để chạy thử")
    ap.add_argument("--delay", type=float, default=0.3, help="Delay giữa các lần gọi (giây)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="CSV đầu ra cho calendar loader")
    args = ap.parse_args(argv)

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = load_tickers(args.industry_map, args.limit)

    if not tickers:
        print("Không có mã nào để xử lý", file=sys.stderr)
        return 2

    run(tickers, args.out, args.delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
