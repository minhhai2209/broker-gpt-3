"""Minimal API for uploading account portfolios."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from flask import Flask, jsonify, make_response, request
from flask_cors import CORS

from scripts.engine.data_engine import EngineConfig

LOGGER = logging.getLogger(__name__)


def _create_response(payload: Dict[str, object], status: int = 200):
    resp = make_response(jsonify(payload), status)
    resp.headers["Cache-Control"] = "no-store"
    return resp


class PortfolioStorage:
    """Handle persistence of portfolios and order history per profile."""

    def __init__(self, portfolio_dir: Path, order_history_dir: Path) -> None:
        self._portfolio_dir = portfolio_dir
        self._order_history_dir = order_history_dir
        self._portfolio_dir.mkdir(parents=True, exist_ok=True)
        self._order_history_dir.mkdir(parents=True, exist_ok=True)

    def save_portfolio(self, profile: str, rows: list[dict]) -> Path:
        if not rows:
            raise ValueError("portfolio list is empty")
        df = pd.DataFrame(rows)
        required = {"Ticker", "Quantity", "AvgPrice"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"missing columns in portfolio: {sorted(missing)}")
        df["Ticker"] = df["Ticker"].astype(str).str.upper()
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0)
        df["AvgPrice"] = pd.to_numeric(df["AvgPrice"], errors="coerce").fillna(0.0)
        path = self._portfolio_dir / f"{profile}.csv"
        df.to_csv(path, index=False)
        return path

    def append_fills(self, profile: str, rows: list[dict]) -> Optional[Path]:
        if not rows:
            return None
        df = pd.DataFrame(rows)
        if df.empty:
            return None
        if "timestamp" not in df.columns:
            df["timestamp"] = datetime.now(timezone.utc).isoformat()
        df["timestamp"] = df["timestamp"].astype(str)
        df["ticker"] = df.get("ticker", df.get("Ticker", "")).astype(str).str.upper()
        df["side"] = df.get("side", "").astype(str).str.upper()
        df["quantity"] = pd.to_numeric(df.get("quantity", df.get("Quantity", 0.0)), errors="coerce").fillna(0.0)
        df["price"] = pd.to_numeric(df.get("price", df.get("Price", 0.0)), errors="coerce").fillna(0.0)
        path = self._order_history_dir / f"{profile}_fills.csv"
        header = not path.exists()
        df.to_csv(path, mode="a", index=False, header=header)
        return path


def create_app(config_path: Optional[Path] = None) -> Flask:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    cfg_path = config_path or Path(os.environ.get("DATA_ENGINE_CONFIG", "config/data_engine.yaml"))
    engine_cfg = EngineConfig.from_yaml(cfg_path)
    storage = PortfolioStorage(engine_cfg.portfolio_dir, engine_cfg.order_history_dir)

    app = Flask(__name__)
    CORS(app)

    @app.post("/upload")
    def upload():
        payload = request.get_json(force=True, silent=True)
        if not isinstance(payload, dict):
            return _create_response({"status": "error", "error": "invalid_json"}, 400)
        profile = str(payload.get("profile", "")).strip()
        if not profile:
            return _create_response({"status": "error", "error": "missing_profile"}, 400)
        safe_profile = "".join(ch for ch in profile if ch.isalnum() or ch in ("-", "_")).strip()
        if not safe_profile:
            return _create_response({"status": "error", "error": "invalid_profile"}, 400)
        portfolio_rows = payload.get("portfolio", [])
        if not isinstance(portfolio_rows, list):
            return _create_response({"status": "error", "error": "portfolio_must_be_list"}, 400)
        fills_rows = payload.get("fills", [])
        if fills_rows is not None and not isinstance(fills_rows, list):
            return _create_response({"status": "error", "error": "fills_must_be_list"}, 400)
        try:
            portfolio_path = storage.save_portfolio(safe_profile, portfolio_rows)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("Failed to persist portfolio %s: %s", safe_profile, exc)
            return _create_response({"status": "error", "error": str(exc)}, 400)
        fills_path = None
        try:
            fills_path = storage.append_fills(safe_profile, fills_rows or [])
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("Failed to persist fills %s: %s", safe_profile, exc)
            return _create_response({"status": "error", "error": str(exc)}, 400)
        response = {
            "status": "ok",
            "profile": safe_profile,
            "portfolio_path": str(portfolio_path),
        }
        if fills_path is not None:
            response["fills_path"] = str(fills_path)
        return _create_response(response, 200)

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8787"))
    create_app().run(host="0.0.0.0", port=port)
