"""Wrapper for the production order engine to support simulation runs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

from .utils import ConfigError, ensure_timezone


@dataclass
class SimulatedOrder:
    symbol: str
    side: str
    type: str
    qty: int
    limit_price: float | None = None
    ttl_days: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "type": self.type,
            "qty": self.qty,
            "limit_price": self.limit_price,
            "ttl_days": self.ttl_days,
            "meta": self.meta,
        }


class EngineWrapper:
    """Call into the order engine with simulation flags and capture the orders."""

    def __init__(self, flags: Optional[Dict[str, Any]] = None):
        self.flags = {"simulate": True, "no_side_effects": True}
        if flags:
            self.flags.update(flags)

    def build_orders_for_day(self, day: date, context: Dict[str, Any]) -> List[SimulatedOrder]:
        engine_result = self._call_engine(day, context)
        orders = []
        for order_dict in engine_result:
            orders.append(self._normalize_order(order_dict, day, context))
        return orders

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_engine(self, day: date, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        custom_builder = context.get("order_builder")
        if callable(custom_builder):
            result = custom_builder(day, context, self.flags)
            return self._coerce_orders(result)
        try:
            from scripts import order_engine
        except Exception as exc:  # pragma: no cover - import failure is critical
            raise ConfigError("Failed to import production order engine") from exc
        # Pass simulation flag to the engine. Context is provided via environment variable
        os.environ.setdefault("BROKER_TEST_MODE", "1")
        result = order_engine.run(simulate=True, context=context, flags=self.flags)
        return self._coerce_orders(result.get("orders") if isinstance(result, dict) else result)

    def _coerce_orders(self, value: Any) -> List[Dict[str, Any]]:
        if value is None:
            return []
        orders: List[Dict[str, Any]] = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    orders.append(dict(item))
                else:
                    # Attempt to read attributes from the engine Order class
                    attrs = {
                        "symbol": getattr(item, "ticker", getattr(item, "symbol", None)),
                        "side": getattr(item, "side", None),
                        "qty": getattr(item, "quantity", getattr(item, "qty", None)),
                        "limit_price": getattr(item, "limit_price", getattr(item, "price", None)),
                        "ttl_days": getattr(item, "ttl_days", None),
                        "meta": getattr(item, "meta", {}),
                    }
                    orders.append(attrs)
        elif isinstance(value, dict):
            orders.append(dict(value))
        else:
            raise ConfigError(f"Unsupported order payload from engine: {type(value)!r}")
        return orders

    def _normalize_order(self, payload: Dict[str, Any], day: date, context: Dict[str, Any]) -> SimulatedOrder:
        symbol = payload.get("symbol") or payload.get("ticker")
        if not symbol:
            raise ConfigError("Engine order is missing a symbol")
        side = (payload.get("side") or "").upper()
        if side not in {"BUY", "SELL"}:
            raise ConfigError(f"Engine order has unsupported side: {side!r}")
        qty = int(payload.get("qty") or payload.get("quantity") or 0)
        if qty <= 0:
            raise ConfigError(f"Engine order must have positive quantity: {payload}")
        limit_price = payload.get("limit_price") or payload.get("price")
        order_type = (payload.get("type") or "LIMIT").upper()
        ttl_days = payload.get("ttl_days")
        meta = dict(payload.get("meta") or {})
        meta.setdefault("engine_date", ensure_timezone(day))
        if context.get("watchlist") is not None:
            meta.setdefault("watchlist_size", len(context["watchlist"]))
        return SimulatedOrder(
            symbol=symbol,
            side=side,
            type=order_type,
            qty=qty,
            limit_price=limit_price,
            ttl_days=ttl_days,
            meta=meta,
        )
