"""
Market data REST endpoints.
"""

from fastapi import APIRouter, HTTPException, Query

from app.core.config import settings
from app.schemas.market import FundingRate, MarketSnapshot, OrderBook
from app.services.market_data import market_data_service

router = APIRouter(prefix="/market", tags=["market"])


@router.get("/connectors", summary="List supported connectors")
async def list_connectors() -> list[str]:
    return settings.SUPPORTED_CONNECTORS


@router.get("/tokens", summary="List supported tokens")
async def list_tokens() -> list[str]:
    return settings.SUPPORTED_TOKENS


@router.get("/orderbook", response_model=OrderBook)
async def get_orderbook(
    connector: str = Query(...),
    token: str = Query(...),
) -> OrderBook:
    if connector not in settings.SUPPORTED_CONNECTORS:
        raise HTTPException(400, f"Unsupported connector: {connector}")
    if token not in settings.SUPPORTED_TOKENS:
        raise HTTPException(400, f"Unsupported token: {token}")
    return await market_data_service.fetch_orderbook(connector, token)


@router.get("/funding", response_model=FundingRate)
async def get_funding(
    connector: str = Query(...),
    token: str = Query(...),
) -> FundingRate:
    if connector not in settings.SUPPORTED_CONNECTORS:
        raise HTTPException(400, f"Unsupported connector: {connector}")
    if token not in settings.SUPPORTED_TOKENS:
        raise HTTPException(400, f"Unsupported token: {token}")
    return await market_data_service.fetch_funding_rate(connector, token)


@router.get("/snapshot", response_model=MarketSnapshot)
async def get_snapshot(
    connector_long: str = Query(...),
    connector_short: str = Query(...),
    token: str = Query(...),
) -> MarketSnapshot:
    for c in (connector_long, connector_short):
        if c not in settings.SUPPORTED_CONNECTORS:
            raise HTTPException(400, f"Unsupported connector: {c}")
    if token not in settings.SUPPORTED_TOKENS:
        raise HTTPException(400, f"Unsupported token: {token}")
    return await market_data_service.get_snapshot(connector_long, connector_short, token)
