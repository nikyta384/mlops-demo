from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from app.models.trade import TradeStatus


# ──────────────────────────── Market data ──────────────────────────────────

class OrderBookLevel(BaseModel):
    price: float
    qty: float


class OrderBook(BaseModel):
    connector: str
    symbol: str
    timestamp: float
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]


class FundingRate(BaseModel):
    connector: str
    symbol: str
    rate: float           # fraction per period, e.g. 0.0001
    next_funding_time: Optional[float] = None


class SpreadSnapshot(BaseModel):
    connector_long: str
    connector_short: str
    symbol: str
    timestamp: float
    best_ask_long: float   # price to buy on long side
    best_bid_short: float  # price to sell on short side
    spread_abs: float      # best_bid_short - best_ask_long
    spread_pct: float      # spread_abs / best_ask_long * 100


class MarketSnapshot(BaseModel):
    spread: SpreadSnapshot
    orderbook_long: OrderBook
    orderbook_short: OrderBook
    funding_long: FundingRate
    funding_short: FundingRate
    net_funding: float
    expected_carry_8h: float   # net_funding * position_size * mark_price (placeholder)


# ──────────────────────────── Trade schemas ────────────────────────────────

class OpenTradeRequest(BaseModel):
    connector_long: str = Field(..., examples=["binance"])
    connector_short: str = Field(..., examples=["bybit"])
    token: str = Field(..., examples=["BTC"])
    position_size: Decimal = Field(..., gt=0, examples=["0.01"])


class CloseTradeRequest(BaseModel):
    trade_id: int
    close_size: Optional[Decimal] = Field(
        None, gt=0, description="Size to close; omit for full close"
    )


class TradeResponse(BaseModel):
    id: int
    opened_at: datetime
    closed_at: Optional[datetime]
    connector_long: str
    connector_short: str
    token: str
    position_size: Decimal
    entry_price_long: Optional[Decimal]
    entry_price_short: Optional[Decimal]
    exit_price_long: Optional[Decimal]
    exit_price_short: Optional[Decimal]
    funding_rate_long: Optional[Decimal]
    funding_rate_short: Optional[Decimal]
    net_funding: Optional[Decimal]
    funding_collected: Optional[Decimal]
    realized_pnl: Optional[Decimal]
    status: TradeStatus
    order_id_long: Optional[str]
    order_id_short: Optional[str]

    model_config = {"from_attributes": True}


class TradeListResponse(BaseModel):
    total: int
    items: List[TradeResponse]


# ──────────────────────────── WebSocket messages ───────────────────────────

class WsSubscribeMessage(BaseModel):
    action: str = "subscribe"
    connector_long: str
    connector_short: str
    token: str


class WsMarketUpdate(BaseModel):
    type: str = "market_update"
    data: MarketSnapshot
