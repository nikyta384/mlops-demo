"""
Integration tests for the FastAPI app (no DB / exchange connections required).
Uses TestClient with mocked services.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.market import (
    FundingRate, MarketSnapshot, OrderBook, OrderBookLevel, SpreadSnapshot
)


def _make_snapshot(cl="binance", cs="bybit", token="BTC"):
    ob = OrderBook(
        connector=cl, symbol=f"{token}/USDT:USDT", timestamp=0,
        bids=[OrderBookLevel(price=29990, qty=1)],
        asks=[OrderBookLevel(price=30000, qty=1)],
    )
    ob2 = OrderBook(
        connector=cs, symbol=f"{token}/USDT:USDT", timestamp=0,
        bids=[OrderBookLevel(price=30050, qty=1)],
        asks=[OrderBookLevel(price=30060, qty=1)],
    )
    return MarketSnapshot(
        spread=SpreadSnapshot(
            connector_long=cl, connector_short=cs, symbol=f"{token}/USDT:USDT",
            timestamp=0, best_ask_long=30000, best_bid_short=30050,
            spread_abs=50, spread_pct=0.167,
        ),
        orderbook_long=ob,
        orderbook_short=ob2,
        funding_long=FundingRate(connector=cl, symbol=f"{token}/USDT:USDT", rate=0.0001),
        funding_short=FundingRate(connector=cs, symbol=f"{token}/USDT:USDT", rate=0.0003),
        net_funding=0.0002,
        expected_carry_8h=6.0,
    )


@pytest.fixture
def client():
    """Create TestClient with DB and market data services mocked out."""
    mock_conn = AsyncMock()
    mock_conn.run_sync = AsyncMock()
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_engine = MagicMock()
    mock_engine.begin = MagicMock(return_value=mock_ctx)
    mock_engine.dispose = AsyncMock()

    with patch("app.main.engine", mock_engine), patch("app.db.session.engine", mock_engine):
        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


def test_list_connectors(client):
    resp = client.get("/api/v1/market/connectors")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert "binance" in data


def test_list_tokens(client):
    resp = client.get("/api/v1/market/tokens")
    assert resp.status_code == 200
    data = resp.json()
    assert "BTC" in data


def test_orderbook_invalid_connector(client):
    resp = client.get("/api/v1/market/orderbook?connector=nonexistent&token=BTC")
    assert resp.status_code == 400


def test_orderbook_invalid_token(client):
    resp = client.get("/api/v1/market/orderbook?connector=binance&token=UNKNOWN")
    assert resp.status_code == 400


@patch(
    "app.api.market.market_data_service.fetch_orderbook",
    new_callable=AsyncMock,
)
def test_orderbook_ok(mock_ob, client):
    mock_ob.return_value = OrderBook(
        connector="binance", symbol="BTC/USDT:USDT", timestamp=0,
        bids=[OrderBookLevel(price=29990, qty=1)],
        asks=[OrderBookLevel(price=30000, qty=1)],
    )
    resp = client.get("/api/v1/market/orderbook?connector=binance&token=BTC")
    assert resp.status_code == 200
    data = resp.json()
    assert data["connector"] == "binance"
    assert len(data["asks"]) == 1


@patch(
    "app.api.market.market_data_service.fetch_funding_rate",
    new_callable=AsyncMock,
)
def test_funding_ok(mock_fr, client):
    mock_fr.return_value = FundingRate(
        connector="binance", symbol="BTC/USDT:USDT", rate=0.0001
    )
    resp = client.get("/api/v1/market/funding?connector=binance&token=BTC")
    assert resp.status_code == 200
    assert resp.json()["rate"] == pytest.approx(0.0001)


@patch(
    "app.api.market.market_data_service.get_snapshot",
    new_callable=AsyncMock,
)
def test_snapshot_ok(mock_snap, client):
    mock_snap.return_value = _make_snapshot()
    resp = client.get(
        "/api/v1/market/snapshot?connector_long=binance&connector_short=bybit&token=BTC"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["net_funding"] == pytest.approx(0.0002)
    assert data["spread"]["spread_abs"] == pytest.approx(50)

