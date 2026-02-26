"""
Market Data Service
───────────────────
Aggregates order books and funding rates from exchanges via ccxt.
Provides spread calculations and a snapshot builder.

Runs a background refresh loop; results are cached in-memory so the
WebSocket broadcaster can serve them without extra I/O.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple

import ccxt.async_support as ccxt

from app.core.config import settings
from app.schemas.market import (
    FundingRate,
    MarketSnapshot,
    OrderBook,
    OrderBookLevel,
    SpreadSnapshot,
)

log = logging.getLogger(__name__)

# symbol suffix for perpetual futures used by each connector
_PERP_SUFFIX: Dict[str, str] = {
    "binance": "/USDT:USDT",
    "bybit": "/USDT:USDT",
    "okx": "-USDT-SWAP",
    "gate": "/USDT:USDT",
    "hyperliquid": "/USDT:USDT",
}


def _perp_symbol(connector: str, token: str) -> str:
    suffix = _PERP_SUFFIX.get(connector, "/USDT:USDT")
    if connector == "okx":
        return f"{token}-USDT-SWAP"
    return f"{token}{suffix}"


def _make_exchange(connector: str) -> ccxt.Exchange:
    cls = getattr(ccxt, connector, None)
    if cls is None:
        raise ValueError(f"Unknown connector: {connector}")
    return cls({"enableRateLimit": True})


class MarketDataService:
    """Thin async wrapper around ccxt that caches the latest snapshot."""

    def __init__(self) -> None:
        self._exchanges: Dict[str, ccxt.Exchange] = {}
    # Cache: (connector, symbol) → last fetch result; refreshed on every call.
    # No TTL – the WS broadcaster drives the refresh frequency via WS_BROADCAST_INTERVAL.        self._ob_cache: Dict[Tuple[str, str], OrderBook] = {}
        self._fr_cache: Dict[Tuple[str, str], FundingRate] = {}

    def _get_exchange(self, connector: str) -> ccxt.Exchange:
        if connector not in self._exchanges:
            self._exchanges[connector] = _make_exchange(connector)
        return self._exchanges[connector]

    # ──────────────────────────────────────────────────────────────────────
    # Order book
    # ──────────────────────────────────────────────────────────────────────

    async def fetch_orderbook(
        self, connector: str, token: str, depth: int = settings.ORDERBOOK_DEPTH
    ) -> OrderBook:
        exchange = self._get_exchange(connector)
        symbol = _perp_symbol(connector, token)
        raw = await exchange.fetch_order_book(symbol, limit=depth)
        ob = OrderBook(
            connector=connector,
            symbol=symbol,
            timestamp=raw.get("timestamp") or time.time() * 1000,
            bids=[OrderBookLevel(price=b[0], qty=b[1]) for b in raw["bids"][:depth]],
            asks=[OrderBookLevel(price=a[0], qty=a[1]) for a in raw["asks"][:depth]],
        )
        self._ob_cache[(connector, token)] = ob
        return ob

    # ──────────────────────────────────────────────────────────────────────
    # Funding rate
    # ──────────────────────────────────────────────────────────────────────

    async def fetch_funding_rate(self, connector: str, token: str) -> FundingRate:
        exchange = self._get_exchange(connector)
        symbol = _perp_symbol(connector, token)
        try:
            raw = await exchange.fetch_funding_rate(symbol)
            rate = float(raw.get("fundingRate") or raw.get("rate") or 0.0)
            nft = raw.get("fundingDatetime") or raw.get("nextFundingTime")
            if isinstance(nft, str):
                nft = exchange.parse8601(nft)
        except Exception as exc:  # noqa: BLE001
            log.warning("funding_rate fetch failed for %s %s: %s", connector, token, exc)
            rate = 0.0
            nft = None
        fr = FundingRate(
            connector=connector,
            symbol=symbol,
            rate=rate,
            next_funding_time=nft,
        )
        self._fr_cache[(connector, token)] = fr
        return fr

    # ──────────────────────────────────────────────────────────────────────
    # Spread
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_spread(ob_long: OrderBook, ob_short: OrderBook) -> SpreadSnapshot:
        """
        Long position buys at ask on connector_long.
        Short position sells at bid on connector_short.
        Spread = bid_short - ask_long  (positive ⟹ instant profit on entry).
        """
        best_ask_long = ob_long.asks[0].price if ob_long.asks else 0.0
        best_bid_short = ob_short.bids[0].price if ob_short.bids else 0.0
        if not best_ask_long or not best_bid_short:
            spread_abs = 0.0
            spread_pct = 0.0
        else:
            spread_abs = best_bid_short - best_ask_long
            spread_pct = spread_abs / best_ask_long * 100
        return SpreadSnapshot(
            connector_long=ob_long.connector,
            connector_short=ob_short.connector,
            symbol=ob_long.symbol,
            timestamp=time.time() * 1000,
            best_ask_long=best_ask_long,
            best_bid_short=best_bid_short,
            spread_abs=spread_abs,
            spread_pct=spread_pct,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Full snapshot
    # ──────────────────────────────────────────────────────────────────────

    async def get_snapshot(
        self,
        connector_long: str,
        connector_short: str,
        token: str,
    ) -> MarketSnapshot:
        ob_long, ob_short, fr_long, fr_short = await asyncio.gather(
            self.fetch_orderbook(connector_long, token),
            self.fetch_orderbook(connector_short, token),
            self.fetch_funding_rate(connector_long, token),
            self.fetch_funding_rate(connector_short, token),
        )
        spread = self.compute_spread(ob_long, ob_short)
        from app.services.funding import FundingCalculator

        net_funding = FundingCalculator.net_funding(fr_long.rate, fr_short.rate)
        mid_price = (spread.best_ask_long + spread.best_bid_short) / 2
        expected_carry = FundingCalculator.expected_carry_8h(net_funding, 1.0, mid_price)
        return MarketSnapshot(
            spread=spread,
            orderbook_long=ob_long,
            orderbook_short=ob_short,
            funding_long=fr_long,
            funding_short=fr_short,
            net_funding=net_funding,
            expected_carry_8h=expected_carry,
        )

    async def close(self) -> None:
        for ex in self._exchanges.values():
            try:
                await ex.close()
            except Exception:  # noqa: BLE001
                pass
        self._exchanges.clear()


# Module-level singleton used by FastAPI lifespan
market_data_service = MarketDataService()
