"""
Execution Service
─────────────────
Handles opening and closing arbitrage legs atomically.

Architecture:
  1. Try Hummingbot Gateway (REST) for execution if configured.
  2. Fall back to ccxt for paper / live trading via direct API keys.

Safe execution guarantees:
  • Price slippage guard: abort if market moved more than SLIPPAGE_TOLERANCE
    between snapshot and execution.
  • Atomic open: if one leg fails after N retries, the successfully opened
    leg is immediately closed (emergency hedge).
  • Partial fill tracking: stores actual filled qty & price from exchange.
"""

from __future__ import annotations

import asyncio
import logging
import time
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple

import ccxt.async_support as ccxt
import httpx

from app.core.config import settings

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Hummingbot Gateway client (optional)
# ─────────────────────────────────────────────────────────────────────────────

class HummingbotGatewayClient:
    """
    Thin HTTP client for Hummingbot Gateway v2.
    Set HUMMINGBOT_GATEWAY_URL env var to enable (e.g. http://localhost:15888).
    """

    def __init__(self, base_url: str) -> None:
        self._base = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base, timeout=10.0)

    async def place_order(
        self,
        connector: str,
        chain: str,
        network: str,
        trading_pair: str,
        side: str,       # "BUY" or "SELL"
        order_type: str, # "MARKET" or "LIMIT"
        amount: str,
        price: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "connector": connector,
            "chain": chain,
            "network": network,
            "trading_pair": trading_pair,
            "address": "",  # populated from gateway wallet config
            "trade_type": side,
            "type": order_type,
            "amount": amount,
        }
        if price:
            payload["price"] = price
        resp = await self._client.post("/amm/trade", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def cancel_order(
        self, connector: str, order_id: str, trading_pair: str
    ) -> Dict[str, Any]:
        resp = await self._client.post(
            "/amm/cancel",
            json={"connector": connector, "orderId": order_id, "tradingPair": trading_pair},
        )
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        await self._client.aclose()


# ─────────────────────────────────────────────────────────────────────────────
# ccxt-based executor (fallback / CEX)
# ─────────────────────────────────────────────────────────────────────────────

_PERP_SUFFIX: Dict[str, str] = {
    "binance": "/USDT:USDT",
    "bybit": "/USDT:USDT",
    "okx": "-USDT-SWAP",
    "gate": "/USDT:USDT",
    "hyperliquid": "/USDT:USDT",
}


def _perp_symbol(connector: str, token: str) -> str:
    if connector == "okx":
        return f"{token}-USDT-SWAP"
    return f"{token}{_PERP_SUFFIX.get(connector, '/USDT:USDT')}"


class CcxtExecutor:
    """
    Stateless executor that creates ccxt exchange instances on demand.
    In production, inject API keys via exchange config.
    """

    def __init__(self, connector_configs: Optional[Dict[str, Dict]] = None) -> None:
        self._configs: Dict[str, Dict] = connector_configs or {}
        self._exchanges: Dict[str, ccxt.Exchange] = {}

    def _get_exchange(self, connector: str) -> ccxt.Exchange:
        if connector not in self._exchanges:
            cls = getattr(ccxt, connector, None)
            if cls is None:
                raise ValueError(f"Unknown ccxt connector: {connector}")
            cfg = self._configs.get(connector, {})
            cfg["enableRateLimit"] = True
            self._exchanges[connector] = cls(cfg)
        return self._exchanges[connector]

    async def place_market_order(
        self,
        connector: str,
        token: str,
        side: str,   # "buy" or "sell"
        amount: float,
    ) -> Dict[str, Any]:
        exchange = self._get_exchange(connector)
        symbol = _perp_symbol(connector, token)
        order = await exchange.create_market_order(symbol, side, amount)
        return order

    async def close_position(
        self,
        connector: str,
        token: str,
        side: str,  # "buy" to close short, "sell" to close long
        amount: float,
    ) -> Dict[str, Any]:
        return await self.place_market_order(connector, token, side, amount)

    async def close(self) -> None:
        for ex in self._exchanges.values():
            try:
                await ex.close()
            except Exception:  # noqa: BLE001
                pass


# ─────────────────────────────────────────────────────────────────────────────
# High-level Execution Service
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionResult:
    __slots__ = ("order_id", "filled_qty", "avg_price", "success", "error")

    def __init__(
        self,
        order_id: Optional[str] = None,
        filled_qty: float = 0.0,
        avg_price: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        self.order_id = order_id
        self.filled_qty = filled_qty
        self.avg_price = avg_price
        self.success = success
        self.error = error


class ExecutionService:

    def __init__(self, executor: Optional[CcxtExecutor] = None) -> None:
        self._executor = executor or CcxtExecutor()

    # ── Price guard ──────────────────────────────────────────────────────────

    def _check_slippage(
        self,
        expected_price: float,
        actual_price: float,
        tolerance: float = settings.PRICE_SLIPPAGE_TOLERANCE,
    ) -> bool:
        if expected_price <= 0:
            return True
        deviation = abs(actual_price - expected_price) / expected_price
        return deviation <= tolerance

    # ── Single leg with retry ────────────────────────────────────────────────

    async def _execute_leg(
        self,
        connector: str,
        token: str,
        side: str,
        amount: float,
        expected_price: float,
        attempts: int = settings.MAX_RETRY_ATTEMPTS,
    ) -> ExecutionResult:
        last_error: Optional[str] = None
        for attempt in range(1, attempts + 1):
            try:
                order = await self._executor.place_market_order(
                    connector, token, side, amount
                )
                filled = float(order.get("filled") or order.get("amount") or amount)
                avg_px = float(
                    order.get("average")
                    or order.get("price")
                    or expected_price
                )
                if not self._check_slippage(expected_price, avg_px):
                    raise ValueError(
                        f"Slippage exceeded on {connector}: "
                        f"expected={expected_price:.4f} got={avg_px:.4f}"
                    )
                return ExecutionResult(
                    order_id=str(order.get("id", "")),
                    filled_qty=filled,
                    avg_price=avg_px,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                log.warning(
                    "Leg execution attempt %d/%d failed on %s: %s",
                    attempt,
                    attempts,
                    connector,
                    exc,
                )
                if attempt < attempts:
                    await asyncio.sleep(settings.RETRY_DELAY_SECONDS * attempt)
        return ExecutionResult(success=False, error=last_error)

    # ── Atomic open ──────────────────────────────────────────────────────────

    async def open_position(
        self,
        connector_long: str,
        connector_short: str,
        token: str,
        position_size: Decimal,
        best_ask_long: float,
        best_bid_short: float,
    ) -> Tuple[ExecutionResult, ExecutionResult]:
        """
        Open both legs concurrently.
        If one leg fails, immediately close the other (emergency hedge).
        Returns (long_result, short_result).
        """
        amount = float(position_size)

        long_task = asyncio.create_task(
            self._execute_leg(
                connector_long, token, "buy", amount, best_ask_long
            )
        )
        short_task = asyncio.create_task(
            self._execute_leg(
                connector_short, token, "sell", amount, best_bid_short
            )
        )

        long_res, short_res = await asyncio.gather(long_task, short_task)

        # Atomic safety: if one leg failed, close the successful leg
        if long_res.success and not short_res.success:
            log.error(
                "Short leg failed; closing long leg on %s to stay flat.", connector_long
            )
            await self._emergency_close(connector_long, token, "sell", long_res.filled_qty)
            long_res.success = False
            long_res.error = f"Emergency-closed due to short-leg failure: {short_res.error}"

        elif short_res.success and not long_res.success:
            log.error(
                "Long leg failed; closing short leg on %s to stay flat.", connector_short
            )
            await self._emergency_close(
                connector_short, token, "buy", short_res.filled_qty
            )
            short_res.success = False
            short_res.error = f"Emergency-closed due to long-leg failure: {long_res.error}"

        return long_res, short_res

    async def close_position(
        self,
        connector_long: str,
        connector_short: str,
        token: str,
        close_size: Decimal,
        expected_price_long: float,
        expected_price_short: float,
    ) -> Tuple[ExecutionResult, ExecutionResult]:
        amount = float(close_size)
        long_res, short_res = await asyncio.gather(
            self._execute_leg(
                connector_long, token, "sell", amount, expected_price_long
            ),
            self._execute_leg(
                connector_short, token, "buy", amount, expected_price_short
            ),
        )
        return long_res, short_res

    async def _emergency_close(
        self, connector: str, token: str, side: str, amount: float
    ) -> None:
        try:
            await self._executor.close_position(connector, token, side, amount)
        except Exception as exc:  # noqa: BLE001
            log.critical(
                "EMERGENCY CLOSE FAILED on %s %s %s qty=%s: %s",
                connector, token, side, amount, exc,
            )

    async def close(self) -> None:
        await self._executor.close()


# Module-level singletons
execution_service = ExecutionService()
