"""Playwright-based scraper to fetch portfolio from TCInvest (TCBS).

Usage:
  python -m scripts.scrapers.tcbs [--profile <name>] [--headful]

Credentials:
  - Read from environment variables `TCBS_USERNAME` and `TCBS_PASSWORD`.
  - If a `.env` file exists at repo root, it will be loaded automatically
    when python-dotenv is available (optional dependency).

Output:
  - Writes `data/portfolios/<profile>.csv` with columns: Ticker,Quantity,AvgPrice

Fingerprint persistence:
  - Uses a persistent Chromium user data directory at `.playwright/tcbs-user-data`.
    The first run may require device/OTP confirmation. Run with `--headful` to
    complete verification once; subsequent runs reuse the same profile.

Profile name:
  - If `--profile` is omitted, reads `TCBS_PROFILE` from environment/.env.
    Defaults to `tcbs`.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


def _repo_root() -> Path:
    cur = Path(__file__).resolve()
    for candidate in [cur.parent, *cur.parents]:
        if (candidate / ".git").exists():
            return candidate
    return Path.cwd()


def _load_env_if_present(root: Path) -> None:
    env_path = root / ".env"
    if not env_path.exists():
        return
    try:
        import dotenv  # type: ignore

        dotenv.load_dotenv(env_path)
    except Exception:
        # Optional dependency; proceed if not available
        pass


def _require_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        raise RuntimeError(f"Missing environment variable: {name}")
    return val


def _norm_number(text: str) -> float:
    """Parse a locale-ish number like '1,600' or '51.24' to float.

    - Remove all non-digit/non-dot/non-minus characters (commas, spaces, units).
    - Keep the last dot as decimal separator.
    """
    raw = (text or "").strip()
    if not raw:
        return 0.0
    # Remove thousand separators and non-numeric symbols (keep - and .)
    cleaned = "".join(ch for ch in raw if ch.isdigit() or ch in {"-", "."})
    if cleaned in {"", "-", ".", "-."}:
        return 0.0
    try:
        return float(cleaned)
    except Exception:
        return 0.0


@dataclass
class TableMapping:
    symbol_idx: int
    quantity_idx: int
    avgprice_idx: int


def build_mapping(headers: List[str]) -> TableMapping:
    """Derive column indices from header texts.

    Expected Vietnamese labels (robust to whitespace/casing):
      - Ticker: 'Mã'
      - Quantity: prefer 'SL Tổng', fallback to 'Được GD' (tradable)
      - AvgPrice: 'Giá vốn'
    """
    norm = [h.strip().lower() for h in headers]

    def idx_of(*candidates: str) -> int:
        for label in candidates:
            for i, h in enumerate(norm):
                if label in h:
                    return i
        raise RuntimeError(f"Missing required column header among: {candidates}")

    symbol_idx = idx_of("mã")
    # Prefer SL Tổng (total quantity), fallback to Được GD
    quantity_idx = idx_of("sl tổng", "được gd", "sl tổng =", "sl tổng")
    avgprice_idx = idx_of("giá vốn")
    return TableMapping(symbol_idx, quantity_idx, avgprice_idx)


def parse_tcbs_table(headers: List[str], rows: List[List[str]]) -> pd.DataFrame:
    mapping = build_mapping(headers)
    out_rows: List[Dict[str, object]] = []

    def _clean_ticker(raw: str) -> str:
        s = (raw or "").strip().upper()
        # Collapse whitespace/newlines
        s = re.sub(r"\s+", " ", s)
        # Pick the first all-caps alnum token (1-6 chars), typical VN tickers
        for tok in re.split(r"\s+|,", s):
            if re.fullmatch(r"[A-Z0-9]{1,6}", tok):
                return tok
        # Fallback: remove non-alnum underscores
        s2 = re.sub(r"[^A-Z0-9]", "", s)
        return s2[:6] if s2 else s
    for r in rows:
        if not r or len(r) <= max(mapping.symbol_idx, mapping.quantity_idx, mapping.avgprice_idx):
            continue
        ticker = _clean_ticker(r[mapping.symbol_idx])
        qty = _norm_number(r[mapping.quantity_idx])
        avg = _norm_number(r[mapping.avgprice_idx])
        if not ticker or qty <= 0:
            continue
        out_rows.append({"Ticker": ticker, "Quantity": int(qty), "AvgPrice": float(avg)})
    return pd.DataFrame(out_rows, columns=["Ticker", "Quantity", "AvgPrice"]) if out_rows else pd.DataFrame(
        columns=["Ticker", "Quantity", "AvgPrice"]
    )


def parse_statement_table(headers: List[str], rows: List[List[str]]) -> pd.DataFrame:
    """Parse statementStock table rows to standardized columns.

    Output columns: Date (YYYY-MM-DD), Ticker, Side (BUY/SELL), ExecQtty (int), ExecPrice (float, thousands VND)
    """
    norm = [h.strip().lower() for h in headers]

    def idx(label_substr: str) -> int:
        for i, h in enumerate(norm):
            if label_substr in h:
                return i
        raise RuntimeError(f"Missing column containing: {label_substr}")

    i_symbol = idx("mã")
    i_date = idx("ngày gd")
    i_side = idx("lệnh")
    i_exec_qty = idx("kl khớp")
    i_exec_price = idx("giá kh")

    out: List[Dict[str, object]] = []
    for r in rows:
        if len(r) <= max(i_symbol, i_date, i_side, i_exec_qty, i_exec_price):
            continue
        raw_date = (r[i_date] or "").strip()
        # Expect dd/mm/YYYY
        m = re.match(r"(\d{2})/(\d{2})/(\d{4})", raw_date)
        if not m:
            continue
        dd, mm, yyyy = m.groups()
        date_iso = f"{yyyy}-{mm}-{dd}"
        side_txt = (r[i_side] or "").strip().lower()
        side = "BUY" if "mua" in side_txt else ("SELL" if "bán" in side_txt else "")
        if not side:
            continue
        symbol = r[i_symbol]
        symbol = re.sub(r"\s+", " ", symbol).strip().upper()
        # pick first alnum token
        sym_tok = None
        for tok in re.split(r"\s+|,", symbol):
            if re.fullmatch(r"[A-Z0-9]{1,6}", tok):
                sym_tok = tok
                break
        symbol = sym_tok or re.sub(r"[^A-Z0-9]", "", symbol)[:6]
        qty = _norm_number(r[i_exec_qty])
        price_vnd = _norm_number(r[i_exec_price])
        # Convert to thousands VND
        price_thousand = price_vnd / 1000.0
        if not symbol or qty <= 0 or price_thousand <= 0:
            continue
        out.append(
            {
                "Date": date_iso,
                "Ticker": symbol,
                "Side": side,
                "ExecQtty": int(qty),
                "ExecPrice": float(round(price_thousand, 3)),
            }
        )
    return pd.DataFrame(out, columns=["Date", "Ticker", "Side", "ExecQtty", "ExecPrice"]) if out else pd.DataFrame(
        columns=["Date", "Ticker", "Side", "ExecQtty", "ExecPrice"]
    )


def _ensure_playwright_installed(pybin: str) -> None:
    # Best-effort: ensure Chromium is installed; skip on failure (user can install manually)
    try:
        import subprocess

        subprocess.run([pybin, "-m", "playwright", "install", "chromium"], check=False, stdout=subprocess.DEVNULL)
    except Exception:
        pass


def fetch_tcbs_portfolio(
    profile: str, headless: bool = True, timeout_ms: int = 300000, with_fills_today: bool = False
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """Launch persistent Chromium, log in to TCBS, navigate to portfolio table, parse and write CSV.

    Returns path to `data/portfolios/<profile>.csv`.
    """
    root = _repo_root()
    _load_env_if_present(root)
    username = _require_env("TCBS_USERNAME")
    password = _require_env("TCBS_PASSWORD")

    # Prepare output dirs
    portfolios_dir = (root / "data" / "portfolios").resolve()
    portfolios_dir.mkdir(parents=True, exist_ok=True)
    out_path = portfolios_dir / f"{profile}.csv"
    fills_out: Optional[Path] = None
    fills_all_out: Optional[Path] = None

    # Playwright runtime
    pybin = sys.executable
    _ensure_playwright_installed(pybin)

    from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout

    user_data_dir = (root / ".playwright" / "tcbs-user-data").resolve()
    user_data_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        def _launch():
            return p.chromium.launch_persistent_context(
                user_data_dir=str(user_data_dir),
                headless=headless,
                viewport={"width": 1400, "height": 900},
                args=["--disable-blink-features=AutomationControlled"],
            )
        try:
            context = _launch()
        except Exception as exc:
            msg = str(exc)
            if "ProcessSingleton" in msg or "SingletonLock" in msg:
                lock = user_data_dir / "SingletonLock"
                try:
                    if lock.exists():
                        print(f"[tcbs] Removing stale lock: {lock}")
                        lock.unlink()
                except Exception:
                    pass
                context = _launch()
            else:
                raise
        page = context.new_page()
        page.set_default_timeout(timeout_ms)

        def attempt_login() -> None:
            print("[tcbs] Filling login form…")
            # Prefer placeholder-based locators; fallback to formcontrolname
            user_candidates = [
                lambda: page.get_by_placeholder(re.compile(r"Email|Số\s*tài\s*khoản|Điện\s*thoại", re.I)),
                lambda: page.locator('input[formcontrolname="username"]'),
                lambda: page.locator('input[type="text"]'),
            ]
            pass_candidates = [
                lambda: page.get_by_placeholder(re.compile(r"Mật\s*khẩu|Password", re.I)),
                lambda: page.locator('input[formcontrolname="password"]'),
                lambda: page.locator('input[type="password"]'),
            ]

            def pick(cands):
                for c in cands:
                    try:
                        loc = c()
                        if loc.count() > 0:
                            return loc.first
                    except Exception:
                        continue
                return None

            user_input = pick(user_candidates)
            pass_input = pick(pass_candidates)
            if not user_input or not pass_input:
                print("[tcbs] Inputs not found; possibly already logged in.")
                return

            user_input.wait_for(state="visible", timeout=timeout_ms)
            user_input.click()
            user_input.fill(username)
            pass_input.click()
            pass_input.fill(password)

            # Try multiple click strategies for the login button
            for attempt in range(4):
                try:
                    print(f"[tcbs] Clicking login button, strategy {attempt+1}…")
                    if attempt == 0:
                        page.get_by_role("button", name=re.compile(r"đăng\s*nhập", re.I)).first.click()
                    elif attempt == 1:
                        page.locator("button.btn-login").first.click()
                    elif attempt == 2:
                        page.locator("button:has-text('Đăng nhập')").first.click()
                    else:
                        pass_input.press("Enter")
                    break
                except Exception:
                    continue

        # Step 0: Open home; if redirected to login, perform login
        page.goto("https://tcinvest.tcbs.com.vn/home", wait_until="domcontentloaded")
        if "guest/login" in page.url:
            attempt_login()

        # Always navigate explicitly to my-asset (site may not redirect)
        page.goto("https://tcinvest.tcbs.com.vn/my-asset", wait_until="domcontentloaded")
        # If redirected back to login, the user likely needs OTP/device confirm; allow manual action in headful mode
        if "guest/login" in page.url:
            print("[tcbs] Still on login page after submit; complete OTP/device confirm if prompted, then continuing…")
            # Give user time (headful) then re-attempt navigate to my-asset
            page.wait_for_timeout(5000)
            attempt_login()
            page.goto("https://tcinvest.tcbs.com.vn/my-asset", wait_until="domcontentloaded")
        # Avoid blocking on long-lived sockets
        page.wait_for_load_state("domcontentloaded")

        # Always navigate explicitly to my-asset (site may not redirect)
        page.goto("https://tcinvest.tcbs.com.vn/my-asset", wait_until="domcontentloaded")
        # If redirected back to login, the user likely needs OTP/device confirm; allow manual action in headful mode
        if "guest/login" in page.url:
            print("[tcbs] Still on login page after submit; complete OTP/device confirm if prompted, then continuing…")
            # Give user time (headful) then re-attempt navigate to my-asset
            page.wait_for_timeout(5000)
            page.goto("https://tcinvest.tcbs.com.vn/my-asset", wait_until="domcontentloaded")
        # Avoid blocking on long-lived sockets
        page.wait_for_load_state("domcontentloaded")

        # Step 2: Open tabs: 'Cổ phiếu' then 'Tài sản'
        try:
            # Try role-based first, then fallback to text locator
            page.get_by_role("tab", name=re.compile(r"cổ\s*phiếu", re.I)).first.click()
        except Exception:
            page.locator("text=Cổ phiếu").first.click()
        try:
            page.locator("text=Tài sản").first.click()
        except Exception:
            pass  # sometimes the sub-tab is default active

        # Step 3: Locate the data table
        table = page.locator("table[role=table]").first
        table.wait_for(state="visible", timeout=timeout_ms)

        # Extract headers
        headers = [h.inner_text().strip() for h in table.locator("thead th").all()]
        # Extract rows (visible only)
        body_rows = table.locator("tbody tr[role=row]").all()
        rows: List[List[str]] = []
        for tr in body_rows:
            cells = tr.locator("td[role=cell]").all()
            rows.append([c.inner_text().strip() for c in cells])

        df = parse_tcbs_table(headers, rows)
        if df.empty:
            # Take a diagnostic screenshot to help debugging selectors/state
            diag_dir = (root / "out" / "diagnostics").resolve()
            diag_dir.mkdir(parents=True, exist_ok=True)
            shot = diag_dir / "tcbs_table_empty.png"
            try:
                page.screenshot(path=str(shot), full_page=True)
                print(f"[tcbs] Saved diagnostic screenshot at {shot}")
            except Exception:
                pass
            raise RuntimeError("TCBS portfolio table parsed empty; review selectors or login state")

        df["Ticker"] = df["Ticker"].astype(str).str.upper()
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0).astype(int)
        df["AvgPrice"] = pd.to_numeric(df["AvgPrice"], errors="coerce").fillna(0.0)

        df.to_csv(out_path, index=False)
        # Optionally fetch today's executed orders after portfolio
        if with_fills_today:
            fills_out, fills_all_out = _fetch_statement_today(page, root, profile)
        context.close()
        return out_path, fills_out, fills_all_out


def _fetch_statement_today(page, root: Path, profile: str) -> Tuple[Path, Path]:
    """Navigate to statementStock, click TRA CỨU, parse table, and write today's fills CSV.

    Writes data/order_history/<profile>_fills.csv with columns:
      timestamp (ISO, VN timezone at 00:00), ticker, side, quantity, price (thousand VND)
    Only today's rows are kept (idempotent per day).
    """
    from datetime import datetime, timezone, timedelta

    VN_TZ = timezone(timedelta(hours=7))
    today_vn = datetime.now(VN_TZ).date()

    page.goto("https://tcinvest.tcbs.com.vn/lookup?tabName=statementStock", wait_until="domcontentloaded")
    # Click search
    try:
        page.get_by_role("button", name=re.compile(r"tra\s*cứu", re.I)).first.click()
    except Exception:
        page.locator("button.btn-lookup").first.click()
    # Wait table
    table = page.locator("table[role=table]").first
    table.wait_for(state="visible")
    headers = [h.inner_text().strip() for h in table.locator("thead th").all()]
    body_rows = table.locator("tbody tr[role=row]").all()
    rows: List[List[str]] = []
    for tr in body_rows:
        cells = tr.locator("td[role=cell]").all()
        rows.append([c.inner_text().strip() for c in cells])

    df_all = parse_statement_table(headers, rows)
    if df_all.empty:
        raise RuntimeError("TCBS statement table parsed empty; review selectors or filters")
    # Filter today (VN)
    df_all["Date"] = pd.to_datetime(df_all["Date"]).dt.date
    df_today = df_all[df_all["Date"] == today_vn]
    # Map to output schema
    out_rows = []
    iso_date = today_vn.strftime("%Y-%m-%d") + "T00:00:00+07:00"
    for r in df_today.itertuples(index=False):
        out_rows.append(
            {
                "timestamp": iso_date,
                "ticker": getattr(r, "Ticker"),
                "side": getattr(r, "Side"),
                "quantity": int(getattr(r, "ExecQtty")),
                "price": float(getattr(r, "ExecPrice")),
            }
        )
    dest_dir = (root / "data" / "order_history").resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path_today = dest_dir / f"{profile}_fills.csv"
    out_path_all = dest_dir / f"{profile}_fills_all.csv"
    # Idempotent per-day: keep only today's set
    pd.DataFrame(out_rows, columns=["timestamp", "ticker", "side", "quantity", "price"]).to_csv(out_path_today, index=False)
    # Always overwrite the full normalized table (keep normalized columns and units)
    df_all.rename(
        columns={
            "Date": "date",
            "Ticker": "ticker",
            "Side": "side",
            "ExecQtty": "quantity",
            "ExecPrice": "price",
        },
        inplace=True,
    )
    # Convert back to string ISO date for readability
    df_all["date"] = pd.to_datetime(df_all["date"]).dt.strftime("%Y-%m-%d")
    df_all.to_csv(out_path_all, index=False)
    return out_path_today, out_path_all


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch portfolio from TCInvest (TCBS) via Playwright")
    parser.add_argument("--profile", default=None, help="Profile name for output CSV in data/portfolios/")
    parser.add_argument("--headful", action="store_true", help="Run browser with UI for first-time device confirmation")
    parser.add_argument("--timeout-ms", type=int, default=300000, help="Global Playwright default timeout in milliseconds")
    parser.add_argument("--fills", action="store_true", help="Also fetch today's executed orders and write data/order_history/<profile>_fills.csv")
    args = parser.parse_args(argv)
    env_profile = os.environ.get("TCBS_PROFILE", "").strip()
    profile = (args.profile or env_profile or "tcbs").strip()
    p_path, f_path, f_all_path = fetch_tcbs_portfolio(
        profile, headless=not args.headful, timeout_ms=int(args.timeout_ms), with_fills_today=bool(args.fills)
    )
    print(str(p_path))
    if f_path:
        print(str(f_path))
    if f_all_path:
        print(str(f_all_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
