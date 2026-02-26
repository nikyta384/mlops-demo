import enum
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Enum,
    Numeric,
    String,
    func,
)

from app.db.session import Base


class TradeStatus(str, enum.Enum):
    OPEN = "open"
    CLOSED = "closed"
    PARTIALLY_CLOSED = "partially_closed"
    ERROR = "error"


class ArbitrageTrade(Base):
    __tablename__ = "arbitrage_trades"

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    # Timing
    opened_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    closed_at = Column(DateTime(timezone=True), nullable=True)

    # Exchange configuration
    connector_long = Column(String(64), nullable=False)
    connector_short = Column(String(64), nullable=False)
    token = Column(String(32), nullable=False)

    # Position size (in base asset)
    position_size = Column(Numeric(28, 10), nullable=False)

    # Entry prices
    entry_price_long = Column(Numeric(28, 10), nullable=True)
    entry_price_short = Column(Numeric(28, 10), nullable=True)

    # Exit prices
    exit_price_long = Column(Numeric(28, 10), nullable=True)
    exit_price_short = Column(Numeric(28, 10), nullable=True)

    # Funding at entry
    funding_rate_long = Column(Numeric(18, 10), nullable=True)
    funding_rate_short = Column(Numeric(18, 10), nullable=True)

    # Funding collected / paid while position was open
    funding_collected = Column(Numeric(28, 10), nullable=True, default=Decimal("0"))

    # Net funding at entry  (long_rate - short_rate expressed as fraction)
    net_funding = Column(Numeric(18, 10), nullable=True)

    # PnL
    realized_pnl = Column(Numeric(28, 10), nullable=True)

    # Status
    status = Column(
        Enum(TradeStatus, name="trade_status"),
        nullable=False,
        default=TradeStatus.OPEN,
    )

    # Exchange order ids for audit
    order_id_long = Column(String(128), nullable=True)
    order_id_short = Column(String(128), nullable=True)
