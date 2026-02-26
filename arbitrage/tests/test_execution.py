"""
Tests for execution service: slippage guard, retry logic, emergency close.
"""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from app.services.execution import CcxtExecutor, ExecutionResult, ExecutionService


class TestSlippageGuard:
    def setup_method(self):
        self.svc = ExecutionService()

    def test_within_tolerance(self):
        assert self.svc._check_slippage(30000.0, 30100.0, tolerance=0.005) is True

    def test_exceeds_tolerance(self):
        assert self.svc._check_slippage(30000.0, 30200.0, tolerance=0.005) is False

    def test_zero_expected_price(self):
        # If expected price is 0 (unknown), always passes
        assert self.svc._check_slippage(0.0, 30000.0) is True

    def test_exact_boundary(self):
        assert self.svc._check_slippage(30000.0, 30150.0, tolerance=0.005) is True  # 0.5% exactly

    def test_slightly_over_boundary(self):
        assert self.svc._check_slippage(30000.0, 30151.0, tolerance=0.005) is False


@pytest.mark.asyncio
class TestRetryLogic:
    async def test_succeeds_on_first_attempt(self):
        mock_exec = AsyncMock(spec=CcxtExecutor)
        mock_exec.place_market_order.return_value = {
            "id": "order1",
            "filled": 0.01,
            "average": 30000.0,
        }
        svc = ExecutionService(executor=mock_exec)
        result = await svc._execute_leg("binance", "BTC", "buy", 0.01, 30000.0)
        assert result.success is True
        assert result.order_id == "order1"
        assert mock_exec.place_market_order.call_count == 1

    async def test_retries_on_failure_then_succeeds(self):
        mock_exec = AsyncMock(spec=CcxtExecutor)
        mock_exec.place_market_order.side_effect = [
            Exception("network error"),
            {"id": "order2", "filled": 0.01, "average": 30000.0},
        ]
        svc = ExecutionService(executor=mock_exec)
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await svc._execute_leg("binance", "BTC", "buy", 0.01, 30000.0, attempts=3)
        assert result.success is True
        assert mock_exec.place_market_order.call_count == 2

    async def test_fails_after_max_retries(self):
        mock_exec = AsyncMock(spec=CcxtExecutor)
        mock_exec.place_market_order.side_effect = Exception("persistent error")
        svc = ExecutionService(executor=mock_exec)
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await svc._execute_leg("binance", "BTC", "buy", 0.01, 30000.0, attempts=3)
        assert result.success is False
        assert "persistent error" in result.error
        assert mock_exec.place_market_order.call_count == 3

    async def test_slippage_causes_failure(self):
        mock_exec = AsyncMock(spec=CcxtExecutor)
        mock_exec.place_market_order.return_value = {
            "id": "order3",
            "filled": 0.01,
            "average": 31000.0,  # >5% slippage from 30000
        }
        svc = ExecutionService(executor=mock_exec)
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await svc._execute_leg("binance", "BTC", "buy", 0.01, 30000.0, attempts=1)
        assert result.success is False
        assert "Slippage" in result.error


@pytest.mark.asyncio
class TestAtomicOpen:
    async def test_both_legs_succeed(self):
        mock_exec = AsyncMock(spec=CcxtExecutor)
        mock_exec.place_market_order.return_value = {
            "id": "oid", "filled": 0.01, "average": 30000.0
        }
        svc = ExecutionService(executor=mock_exec)
        long_res, short_res = await svc.open_position(
            "binance", "bybit", "BTC", Decimal("0.01"), 30000.0, 30050.0
        )
        assert long_res.success is True
        assert short_res.success is True

    async def test_short_leg_fails_triggers_emergency_close(self):
        """If short leg fails, the long leg must be emergency-closed."""
        call_count = {"n": 0}
        close_called = {"v": False}

        async def fake_place(connector, token, side, amount):
            call_count["n"] += 1
            if connector == "bybit" and side == "sell":
                raise Exception("short leg error")
            return {"id": "oid", "filled": amount, "average": 30000.0}

        mock_exec = MagicMock(spec=CcxtExecutor)
        mock_exec.place_market_order = AsyncMock(side_effect=fake_place)

        async def fake_close(connector, token, side, amount):
            close_called["v"] = True
            return {"id": "close", "filled": amount, "average": 30000.0}

        mock_exec.close_position = AsyncMock(side_effect=fake_close)

        svc = ExecutionService(executor=mock_exec)
        with patch("asyncio.sleep", new_callable=AsyncMock):
            long_res, short_res = await svc.open_position(
                "binance", "bybit", "BTC", Decimal("0.01"), 30000.0, 30050.0
            )

        # Both results should be marked failed after emergency close
        assert long_res.success is False
        assert short_res.success is False
        # Emergency close was called to flatten long
        assert close_called["v"] is True

    async def test_long_leg_fails_triggers_emergency_close(self):
        """If long leg fails, the short leg must be emergency-closed."""
        close_called = {"v": False}

        async def fake_place(connector, token, side, amount):
            if connector == "binance" and side == "buy":
                raise Exception("long leg error")
            return {"id": "oid", "filled": amount, "average": 30050.0}

        mock_exec = MagicMock(spec=CcxtExecutor)
        mock_exec.place_market_order = AsyncMock(side_effect=fake_place)

        async def fake_close(connector, token, side, amount):
            close_called["v"] = True
            return {}

        mock_exec.close_position = AsyncMock(side_effect=fake_close)

        svc = ExecutionService(executor=mock_exec)
        with patch("asyncio.sleep", new_callable=AsyncMock):
            long_res, short_res = await svc.open_position(
                "binance", "bybit", "BTC", Decimal("0.01"), 30000.0, 30050.0
            )

        assert long_res.success is False
        assert short_res.success is False
        assert close_called["v"] is True
