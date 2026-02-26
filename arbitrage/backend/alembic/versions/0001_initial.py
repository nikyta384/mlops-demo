"""initial

Revision ID: 0001_initial
Revises: 
Create Date: 2026-02-26

"""
from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE TYPE trade_status AS ENUM ('open', 'closed', 'partially_closed', 'error')")
    op.create_table(
        "arbitrage_trades",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("connector_long", sa.String(64), nullable=False),
        sa.Column("connector_short", sa.String(64), nullable=False),
        sa.Column("token", sa.String(32), nullable=False),
        sa.Column("position_size", sa.Numeric(28, 10), nullable=False),
        sa.Column("entry_price_long", sa.Numeric(28, 10), nullable=True),
        sa.Column("entry_price_short", sa.Numeric(28, 10), nullable=True),
        sa.Column("exit_price_long", sa.Numeric(28, 10), nullable=True),
        sa.Column("exit_price_short", sa.Numeric(28, 10), nullable=True),
        sa.Column("funding_rate_long", sa.Numeric(18, 10), nullable=True),
        sa.Column("funding_rate_short", sa.Numeric(18, 10), nullable=True),
        sa.Column("funding_collected", sa.Numeric(28, 10), nullable=True),
        sa.Column("net_funding", sa.Numeric(18, 10), nullable=True),
        sa.Column("realized_pnl", sa.Numeric(28, 10), nullable=True),
        sa.Column(
            "status",
            sa.Enum("open", "closed", "partially_closed", "error", name="trade_status"),
            nullable=False,
        ),
        sa.Column("order_id_long", sa.String(128), nullable=True),
        sa.Column("order_id_short", sa.String(128), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trades_status", "arbitrage_trades", ["status"])
    op.create_index("ix_trades_opened_at", "arbitrage_trades", ["opened_at"])


def downgrade() -> None:
    op.drop_index("ix_trades_opened_at", table_name="arbitrage_trades")
    op.drop_index("ix_trades_status", table_name="arbitrage_trades")
    op.drop_table("arbitrage_trades")
    op.execute("DROP TYPE trade_status")
