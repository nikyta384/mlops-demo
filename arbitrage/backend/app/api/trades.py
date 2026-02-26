"""
Trade REST endpoints (open / close / history).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.crud import close_trade, create_trade, get_trade, list_trades
from app.db.session import get_db
from app.models.trade import TradeStatus
from app.schemas.market import (
    CloseTradeRequest,
    OpenTradeRequest,
    TradeListResponse,
    TradeResponse,
)
from app.services.execution import execution_service
from app.services.market_data import market_data_service

router = APIRouter(prefix="/trades", tags=["trades"])


@router.post("/open", response_model=TradeResponse, status_code=201)
async def open_trade(
    req: OpenTradeRequest,
    db: AsyncSession = Depends(get_db),
) -> TradeResponse:
    # 1. Fetch current market snapshot for price reference & funding
    try:
        snapshot = await market_data_service.get_snapshot(
            req.connector_long, req.connector_short, req.token
        )
    except Exception as exc:
        raise HTTPException(502, f"Market data unavailable: {exc}") from exc

    best_ask_long = snapshot.spread.best_ask_long
    best_bid_short = snapshot.spread.best_bid_short

    # 2. Execute both legs atomically
    long_res, short_res = await execution_service.open_position(
        connector_long=req.connector_long,
        connector_short=req.connector_short,
        token=req.token,
        position_size=req.position_size,
        best_ask_long=best_ask_long,
        best_bid_short=best_bid_short,
    )

    if not long_res.success or not short_res.success:
        error_msg = long_res.error or short_res.error
        raise HTTPException(503, f"Execution failed: {error_msg}")

    # 3. Persist trade record
    trade = await create_trade(
        db=db,
        req=req,
        entry_price_long=Decimal(str(long_res.avg_price)),
        entry_price_short=Decimal(str(short_res.avg_price)),
        funding_rate_long=snapshot.funding_long.rate,
        funding_rate_short=snapshot.funding_short.rate,
        net_funding=snapshot.net_funding,
        order_id_long=long_res.order_id,
        order_id_short=short_res.order_id,
    )

    return TradeResponse.model_validate(trade)


@router.post("/close", response_model=TradeResponse)
async def close_trade_endpoint(
    req: CloseTradeRequest,
    db: AsyncSession = Depends(get_db),
) -> TradeResponse:
    trade = await get_trade(db, req.trade_id)
    if trade is None:
        raise HTTPException(404, "Trade not found")
    if trade.status != TradeStatus.OPEN:
        raise HTTPException(409, f"Trade is already {trade.status}")

    close_size = req.close_size or trade.position_size

    # Fetch current prices for slippage guard
    try:
        snapshot = await market_data_service.get_snapshot(
            trade.connector_long, trade.connector_short, trade.token
        )
    except Exception as exc:
        raise HTTPException(502, f"Market data unavailable: {exc}") from exc

    long_res, short_res = await execution_service.close_position(
        connector_long=trade.connector_long,
        connector_short=trade.connector_short,
        token=trade.token,
        close_size=close_size,
        expected_price_long=snapshot.orderbook_long.bids[0].price if snapshot.orderbook_long.bids else snapshot.spread.best_ask_long,   # selling long → bid on long exchange
        expected_price_short=snapshot.orderbook_short.asks[0].price if snapshot.orderbook_short.asks else snapshot.spread.best_bid_short,  # buying back short → ask on short exchange
    )

    if not long_res.success or not short_res.success:
        error_msg = long_res.error or short_res.error
        raise HTTPException(503, f"Close execution failed: {error_msg}")

    # Calculate realized PnL
    ep_long = float(trade.entry_price_long or 0)
    ep_short = float(trade.entry_price_short or 0)
    qty = float(close_size)
    pnl_long = (long_res.avg_price - ep_long) * qty
    pnl_short = (ep_short - short_res.avg_price) * qty
    realized_pnl = Decimal(str(pnl_long + pnl_short))

    updated = await close_trade(
        db=db,
        trade=trade,
        exit_price_long=Decimal(str(long_res.avg_price)),
        exit_price_short=Decimal(str(short_res.avg_price)),
        realized_pnl=realized_pnl,
        order_id_long=long_res.order_id,
        order_id_short=short_res.order_id,
    )

    return TradeResponse.model_validate(updated)


@router.get("/", response_model=TradeListResponse)
async def list_trades_endpoint(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    status: Optional[TradeStatus] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> TradeListResponse:
    total, items = await list_trades(db, offset=offset, limit=limit, status=status)
    return TradeListResponse(
        total=total,
        items=[TradeResponse.model_validate(t) for t in items],
    )


@router.get("/{trade_id}", response_model=TradeResponse)
async def get_trade_endpoint(
    trade_id: int,
    db: AsyncSession = Depends(get_db),
) -> TradeResponse:
    trade = await get_trade(db, trade_id)
    if trade is None:
        raise HTTPException(404, "Trade not found")
    return TradeResponse.model_validate(trade)
