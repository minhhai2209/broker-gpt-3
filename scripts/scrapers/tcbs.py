"""Playwright-based scraper to fetch portfolio from TCInvest (TCBS).

Usage:
  python -m scripts.scrapers.tcbs [--profile <name>] [--headful]

Credentials:
  - Read from environment variables `TCBS_USERNAME` and `TCBS_PASSWORD`.
  - If a `.env` file exists at repo root, it will be loaded automatically
    when python-dotenv is available (optional dependency).

Output:
  - Writes `data/portfolios/<profile>/portfolio.csv` with columns: Ticker,Quantity,AvgPrice

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
import logging
import time
from contextlib import contextmanager

import pandas as pd


def _repo_root() -> Path:
    cur = Path(__file__).resolve()
    for candidate in [cur.parent, *cur.parents]:
        if (candidate / ".git").exists():
            return candidate
    return Path.cwd()


# Module logger (keep lightweight; respect app config if present)
LOGGER = logging.getLogger("tcbs")


def _ensure_logging_configured() -> None:
    """Configure basic logging if application hasn't done so.

    Uses a concise format and INFO level by default.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(name)s: %(message)s")


@contextmanager
def _log_step(name: str, **fields: object):
    """Log step start/end with duration and structured fields."""
    meta = " ".join(f"{k}={v}" for k, v in fields.items()) if fields else ""
    LOGGER.info("▶ %s%s", name, (" " + meta) if meta else "")
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        dt = time.perf_counter() - t0
        LOGGER.exception("✖ %s failed after %.3fs", name, dt)
        raise
    else:
        dt = time.perf_counter() - t0
        LOGGER.info("✓ %s done in %.3fs", name, dt)


def _log_url_after_goto(page, tag: str) -> None:
    """Log the current page URL immediately after goto and after a short settle.

    Accept a Playwright `page` to make this usable across functions.
    """
    try:
        LOGGER.info("current_url=%s tag=%s", page.url, f"{tag}.after_goto")
        page.wait_for_load_state("domcontentloaded")
    except Exception:
        pass
    try:
        page.wait_for_timeout(700)
    except Exception:
        pass
    LOGGER.info("current_url=%s tag=%s", page.url, f"{tag}.after_settle")


def _load_env_if_present(root: Path) -> None:
    env_path = root / ".env"
    if not env_path.exists():
        return
    try:
        import dotenv  # type: ignore

        dotenv.load_dotenv(env_path)
        LOGGER.debug("Loaded .env from %s", env_path)
    except Exception:
        # Optional dependency; proceed if not available
        LOGGER.debug("python-dotenv unavailable; skip .env load")


def _require_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        raise RuntimeError(f"Missing environment variable: {name}")
    # Do not log secrets; only presence/length
    LOGGER.debug("Env %s present (len=%d)", name, len(val))
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
    LOGGER.debug("parse_tcbs_table: headers=%s", headers)
    mapping = build_mapping(headers)
    LOGGER.debug(
        "Column mapping: symbol=%d quantity=%d avg=%d",
        mapping.symbol_idx,
        mapping.quantity_idx,
        mapping.avgprice_idx,
    )
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
    df = pd.DataFrame(out_rows, columns=["Ticker", "Quantity", "AvgPrice"]) if out_rows else pd.DataFrame(
        columns=["Ticker", "Quantity", "AvgPrice"]
    )
    LOGGER.info("Parsed portfolio rows: %d (raw=%d)", len(df), len(rows))
    if LOGGER.isEnabledFor(logging.DEBUG) and not df.empty:
        LOGGER.debug("Sample: %s", df.head(10).to_dict(orient="records"))
    return df


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
    df = pd.DataFrame(out, columns=["Date", "Ticker", "Side", "ExecQtty", "ExecPrice"]) if out else pd.DataFrame(
        columns=["Date", "Ticker", "Side", "ExecQtty", "ExecPrice"]
    )
    LOGGER.info("Parsed statement rows: %d (raw=%d)", len(df), len(rows))
    if LOGGER.isEnabledFor(logging.DEBUG) and not df.empty:
        LOGGER.debug("Statement sample: %s", df.head(10).to_dict(orient="records"))
    return df


def _ensure_playwright_installed(pybin: str) -> None:
    # Best-effort: ensure Chromium is installed; skip on failure (user can install manually)
    try:
        import subprocess
        LOGGER.debug("Ensure Playwright Chromium (via %s)", pybin)
        subprocess.run(
            [pybin, "-m", "playwright", "install", "chromium"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        LOGGER.debug("Skip Playwright installation check (subprocess error)")


def fetch_tcbs_portfolio(
    profile: str,
    headless: bool = True,
    timeout_ms: int = 300000,
    with_fills_today: bool = True,
    slow_mo_ms: Optional[int] = None,
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """Launch persistent Chromium, log in to TCBS, navigate to portfolio table, parse and write CSV.

    Returns:
      - portfolio CSV path
      - today's fills CSV path (or None if disabled)
      - full normalized fills CSV path (or None if disabled)
    """
    _ensure_logging_configured()
    with _log_step("setup"):
        root = _repo_root()
        LOGGER.info("repo_root=%s", root)
        _load_env_if_present(root)
        username = _require_env("TCBS_USERNAME")
        password = _require_env("TCBS_PASSWORD")

    # Prepare output dirs
    portfolios_root = (root / "data" / "portfolios").resolve()
    profile_dir = (portfolios_root / profile).resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    out_path = profile_dir / "portfolio.csv"
    LOGGER.info("output_file=%s", out_path)
    fills_out: Optional[Path] = None
    fills_all_out: Optional[Path] = None

    # Playwright runtime
    pybin = sys.executable
    _ensure_playwright_installed(pybin)

    from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout

    user_data_dir = (root / ".playwright" / "tcbs-user-data").resolve()
    user_data_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("user_data_dir=%s headless=%s timeout_ms=%d with_fills_today=%s", user_data_dir, headless, timeout_ms, with_fills_today)

    with sync_playwright() as p:
        # Determine pacing: default to slower actions in headful mode
        sm = int(slow_mo_ms) if slow_mo_ms is not None else (250 if not headless else 0)
        sm = max(0, sm)
        def _launch():
            return p.chromium.launch_persistent_context(
                user_data_dir=str(user_data_dir),
                headless=headless,
                viewport={"width": 1400, "height": 900},
                slow_mo=sm or 0,
                args=["--disable-blink-features=AutomationControlled"],
            )
        try:
            with _log_step("launch_chromium", headless=headless):
                context = _launch()
        except Exception as exc:
            msg = str(exc)
            if "ProcessSingleton" in msg or "SingletonLock" in msg:
                lock = user_data_dir / "SingletonLock"
                try:
                    if lock.exists():
                        LOGGER.warning("Removing stale lock: %s", lock)
                        lock.unlink()
                except Exception:
                    LOGGER.debug("Failed to remove lock; retrying launch anyway")
                with _log_step("launch_chromium_retry", reason="singleton lock"):
                    context = _launch()
            else:
                raise
        page = context.new_page()
        page.set_default_timeout(timeout_ms)
        try:
            page.set_default_navigation_timeout(timeout_ms)
        except Exception:
            pass
        LOGGER.info("pacing slow_mo_ms=%d", sm)

        def _has_login_ui() -> bool:
            """Heuristic: detect presence of TCBS login inputs/buttons.

            Prefer feature detection over relying on page.url to handle client-side redirects.
            """
            try:
                loc = page.get_by_placeholder(re.compile(r"Email|Số\s*tài\s*khoản|Điện\s*thoại", re.I))
                if loc.count() > 0 and loc.first.is_visible():
                    return True
            except Exception:
                pass
            selectors = [
                'input[formcontrolname="username"]',
                'input[formcontrolname="password"]',
                'input[type="password"]',
                "button.btn-login",
                "button:has-text('Đăng nhập')",
            ]
            for sel in selectors:
                try:
                    loc = page.locator(sel)
                    if loc.count() > 0 and loc.first.is_visible():
                        return True
                except Exception:
                    continue
            return False

        # URL logging moved to module scope as _log_url_after_goto(page, tag)

        def attempt_login() -> None:
            LOGGER.info("login: attempt begin")
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
                LOGGER.info("login: inputs not found; maybe already logged in")
                return

            with _log_step("login_fill"):
                user_input.wait_for(state="visible", timeout=timeout_ms)
                user_input.click()
                user_input.fill(username)
                pass_input.click()
                pass_input.fill(password)
                # Small pause so any oninput validation can enable the button
                try:
                    page.wait_for_timeout(max(200, min(800, (sm or 0) * 2)))
                except Exception:
                    pass

            # Try multiple click strategies for the login button
            for attempt in range(4):
                try:
                    LOGGER.info("login: click button strategy=%d", attempt + 1)
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
        with _log_step("navigate", url="/home"):
            # Use 'commit' to return as soon as headers arrive; SPA may redirect afterward
            resp = page.goto("https://tcinvest.tcbs.com.vn/home", wait_until="commit")
            try:
                LOGGER.info("goto_response_url=%s status=%s", getattr(resp, "url", None), getattr(resp, "status", None))
            except Exception:
                pass
            _log_url_after_goto(page, "/home")
        if _has_login_ui():
            attempt_login()

        # Always navigate explicitly to my-asset (site may not redirect)
        with _log_step("navigate", url="/my-asset"):
            resp = page.goto("https://tcinvest.tcbs.com.vn/my-asset", wait_until="commit")
            try:
                LOGGER.info("goto_response_url=%s status=%s", getattr(resp, "url", None), getattr(resp, "status", None))
            except Exception:
                pass
            _log_url_after_goto(page, "/my-asset")
        # If redirected back to login, the user likely needs OTP/device confirm; allow manual action in headful mode
        if _has_login_ui():
            LOGGER.warning("login: still on login; complete OTP/device confirm if prompted")
            # Give user time (headful) then re-attempt navigate to my-asset
            page.wait_for_timeout(5000)
            attempt_login()
            with _log_step("navigate", url="/my-asset"):
                resp = page.goto("https://tcinvest.tcbs.com.vn/my-asset", wait_until="commit")
                try:
                    LOGGER.info("goto_response_url=%s status=%s", getattr(resp, "url", None), getattr(resp, "status", None))
                except Exception:
                    pass
                _log_url_after_goto(page, "/my-asset.retry")
        # Avoid blocking on long-lived sockets
        page.wait_for_load_state("domcontentloaded")

        # Always navigate explicitly to my-asset (site may not redirect)
        with _log_step("navigate", url="/my-asset"):
            resp = page.goto("https://tcinvest.tcbs.com.vn/my-asset", wait_until="commit")
            try:
                LOGGER.info("goto_response_url=%s status=%s", getattr(resp, "url", None), getattr(resp, "status", None))
            except Exception:
                pass
            _log_url_after_goto(page, "/my-asset.2")
        # If redirected back to login, the user likely needs OTP/device confirm; allow manual action in headful mode
        if _has_login_ui():
            LOGGER.warning("login: still on login; complete OTP/device confirm if prompted")
            # Give user time (headful) then re-attempt navigate to my-asset
            page.wait_for_timeout(5000)
            with _log_step("navigate", url="/my-asset"):
                resp = page.goto("https://tcinvest.tcbs.com.vn/my-asset", wait_until="commit")
                try:
                    LOGGER.info("goto_response_url=%s status=%s", getattr(resp, "url", None), getattr(resp, "status", None))
                except Exception:
                    pass
                _log_url_after_goto(page, "/my-asset.3")
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
        with _log_step("locate_table", selector="table[role=table]"):
            table = page.locator("table[role=table]").first
            table.wait_for(state="visible", timeout=timeout_ms)

        # Extract headers
        headers = [h.inner_text().strip() for h in table.locator("thead th").all()]
        # Extract rows (visible only)
        body_rows = table.locator("tbody tr[role=row]").all()
        LOGGER.info("table: headers=%s rows_found=%d", headers, len(body_rows))
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
                LOGGER.warning("Saved diagnostic screenshot at %s", shot)
            except Exception:
                pass
            raise RuntimeError("TCBS portfolio table parsed empty; review selectors or login state")

        df["Ticker"] = df["Ticker"].astype(str).str.upper()
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0).astype(int)
        df["AvgPrice"] = pd.to_numeric(df["AvgPrice"], errors="coerce").fillna(0.0)

        with _log_step("write_portfolio_csv", rows=len(df), path=str(out_path)):
            df.to_csv(out_path, index=False)
        # Optionally fetch today's executed orders after portfolio
        if with_fills_today:
            fills_out, fills_all_out = _fetch_statement_today(page, root, profile)
        context.close()
        return out_path, fills_out, fills_all_out


def _fetch_statement_today(page, root: Path, profile: str) -> Tuple[Path, Path]:
    """Navigate to statementStock, click TRA CỨU, parse table, and write today's fills CSV.

    Writes data/order_history/<profile>/fills.csv with columns:
      timestamp (ISO, VN timezone at 00:00), ticker, side, quantity, price (thousand VND)
    Only today's rows are kept (idempotent per day).
    """
    from datetime import datetime, timezone, timedelta

    VN_TZ = timezone(timedelta(hours=7))
    today_vn = datetime.now(VN_TZ).date()

    with _log_step("navigate", url="/lookup?tabName=statementStock"):
        resp = page.goto("https://tcinvest.tcbs.com.vn/lookup?tabName=statementStock", wait_until="domcontentloaded")
        try:
            LOGGER.info("goto_response_url=%s status=%s", getattr(resp, "url", None), getattr(resp, "status", None))
        except Exception:
            pass
        _log_url_after_goto(page, "/lookup")
    # Click search
    with _log_step("click_search"):
        try:
            page.get_by_role("button", name=re.compile(r"tra\s*cứu", re.I)).first.click()
        except Exception:
            page.locator("button.btn-lookup").first.click()
    # Wait table
    with _log_step("locate_statement_table"):
        table = page.locator("table[role=table]").first
        table.wait_for(state="visible")
        headers = [h.inner_text().strip() for h in table.locator("thead th").all()]
        body_rows = table.locator("tbody tr[role=row]").all()
        LOGGER.info("statement: headers=%s rows_found=%d", headers, len(body_rows))
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
    dest_dir = (root / "data" / "order_history" / profile).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path_today = dest_dir / "fills.csv"
    out_path_all = dest_dir / "fills_all.csv"
    # Idempotent per-day: keep only today's set
    with _log_step("write_fills_today_csv", rows=len(out_rows), path=str(out_path_today)):
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
    with _log_step("write_fills_all_csv", rows=len(df_all), path=str(out_path_all)):
        df_all.to_csv(out_path_all, index=False)
    return out_path_today, out_path_all


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ensure_logging_configured()
    parser = argparse.ArgumentParser(description="Fetch portfolio from TCInvest (TCBS) via Playwright")
    parser.add_argument("--profile", default=None, help="Profile name (folder under data/portfolios/)")
    parser.add_argument("--headful", action="store_true", help="Run browser with UI for first-time device confirmation")
    parser.add_argument("--timeout-ms", type=int, default=300000, help="Global Playwright default timeout in milliseconds")
    # Fills collection: default ON; --no-fills can disable. Keep --fills for back-compat.
    parser.add_argument("--fills", dest="fills", action="store_true", default=None, help="(Default ON) Also fetch today's executed orders and write data/order_history/<profile>/fills.csv")
    parser.add_argument("--no-fills", dest="fills", action="store_false", help="Disable fetching today's executed orders")
    parser.add_argument("--slow-mo-ms", type=int, default=None, help="Delay each Playwright action by N ms (defaults to 250 in headful; 0 in headless)")
    args = parser.parse_args(argv)
    env_profile = os.environ.get("TCBS_PROFILE", "").strip()
    profile = (args.profile or env_profile or "tcbs").strip()
    # Compute default pacing for logging visibility (actual decision is inside fetch)
    computed_slow = args.slow_mo_ms if args.slow_mo_ms is not None else (250 if args.headful else 0)
    fills_enabled = True if args.fills is None else bool(args.fills)
    LOGGER.info(
        "args: profile=%s headful=%s timeout_ms=%d fills=%s slow_mo_ms=%s",
        profile,
        args.headful,
        int(args.timeout_ms),
        fills_enabled,
        computed_slow,
    )
    p_path, f_path, f_all_path = fetch_tcbs_portfolio(
        profile,
        headless=not args.headful,
        timeout_ms=int(args.timeout_ms),
        with_fills_today=fills_enabled,
        slow_mo_ms=args.slow_mo_ms,
    )
    LOGGER.info("portfolio_csv=%s", p_path)
    print(str(p_path))
    if f_path:
        LOGGER.info("fills_today_csv=%s", f_path)
        print(str(f_path))
    if f_all_path:
        LOGGER.info("fills_all_csv=%s", f_all_path)
        print(str(f_all_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
