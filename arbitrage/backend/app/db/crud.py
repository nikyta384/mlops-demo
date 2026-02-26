"""
Persistence Layer
─────────────────
CRUD helpers for ArbitrageTrade.
All functions are async and accept an AsyncSession.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.trade import ArbitrageTrade, TradeStatus
from app.schemas.market import OpenTradeRequest


async def create_trade(
    db: AsyncSession,
    req: OpenTradeRequest,
    entry_price_long: Optional[Decimal],
    entry_price_short: Optional[Decimal],
    funding_rate_long: Optional[float],
    funding_rate_short: Optional[float],
    net_funding: Optional[float],
    order_id_long: Optional[str],
    order_id_short: Optional[str],
) -> ArbitrageTrade:
    trade = ArbitrageTrade(
        opened_at=datetime.now(tz=timezone.utc),
        connector_long=req.connector_long,
        connector_short=req.connector_short,
        token=req.token,
        position_size=req.position_size,
        entry_price_long=entry_price_long,
        entry_price_short=entry_price_short,
        funding_rate_long=funding_rate_long,
        funding_rate_short=funding_rate_short,
        net_funding=net_funding,
        order_id_long=order_id_long,
        order_id_short=order_id_short,
        status=TradeStatus.OPEN,
    )
    db.add(trade)
    await db.flush()  # populate id
    return trade


async def close_trade(
    db: AsyncSession,
    trade: ArbitrageTrade,
    exit_price_long: Decimal,
    exit_price_short: Decimal,
    realized_pnl: Decimal,
    order_id_long: Optional[str] = None,
    order_id_short: Optional[str] = None,
) -> ArbitrageTrade:
    trade.closed_at = datetime.now(tz=timezone.utc)
    trade.exit_price_long = exit_price_long
    trade.exit_price_short = exit_price_short
    trade.realized_pnl = realized_pnl
    trade.status = TradeStatus.CLOSED
    if order_id_long:
        trade.order_id_long = order_id_long
    if order_id_short:
        trade.order_id_short = order_id_short
    await db.flush()
    return trade


async def get_trade(db: AsyncSession, trade_id: int) -> Optional[ArbitrageTrade]:
    result = await db.execute(
        select(ArbitrageTrade).where(ArbitrageTrade.id == trade_id)
    )
    return result.scalar_one_or_none()


async def list_trades(
    db: AsyncSession,
    offset: int = 0,
    limit: int = 50,
    status: Optional[TradeStatus] = None,
) -> tuple[int, List[ArbitrageTrade]]:
    q = select(ArbitrageTrade)
    if status:
        q = q.where(ArbitrageTrade.status == status)
    q_total = select(ArbitrageTrade)
    if status:
        q_total = q_total.where(ArbitrageTrade.status == status)

    from sqlalchemy import func as sqlfunc

    total_result = await db.execute(
        select(sqlfunc.count()).select_from(q_total.subquery())
    )
    total = total_result.scalar_one()

    items_result = await db.execute(
        q.order_by(desc(ArbitrageTrade.opened_at)).offset(offset).limit(limit)
    )
    return total, list(items_result.scalars().all())
