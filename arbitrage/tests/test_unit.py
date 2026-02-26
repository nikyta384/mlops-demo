"""
Unit tests for pure-function modules (no I/O).
"""

import pytest

from app.services.funding import FundingCalculator
from app.services.market_data import MarketDataService
from app.schemas.market import OrderBook, OrderBookLevel


class TestFundingCalculator:
    def test_net_funding_positive(self):
        # short rate > long rate → carry positive (we receive)
        assert FundingCalculator.net_funding(0.0001, 0.0003) == pytest.approx(0.0002)

    def test_net_funding_negative(self):
        assert FundingCalculator.net_funding(0.0003, 0.0001) == pytest.approx(-0.0002)

    def test_net_funding_zero(self):
        assert FundingCalculator.net_funding(0.0002, 0.0002) == pytest.approx(0.0)

    def test_expected_carry_8h(self):
        carry = FundingCalculator.expected_carry_8h(0.0001, 1.0, 30000.0)
        assert carry == pytest.approx(3.0)

    def test_annualized_yield(self):
        ann = FundingCalculator.annualized_yield(0.0001)
        assert ann == pytest.approx(0.0001 * 1095 * 100)

    def test_breakeven_spread(self):
        be = FundingCalculator.breakeven_spread(0.0004, 0.0004)
        assert be == pytest.approx(0.0016)


class TestSpreadComputation:
    def _make_ob(self, connector: str, bids, asks):
        return OrderBook(
            connector=connector,
            symbol="BTC/USDT:USDT",
            timestamp=0.0,
            bids=[OrderBookLevel(price=b[0], qty=b[1]) for b in bids],
            asks=[OrderBookLevel(price=a[0], qty=a[1]) for a in asks],
        )

    def test_positive_spread(self):
        ob_long = self._make_ob("binance", bids=[(29990, 1)], asks=[(30000, 1)])
        ob_short = self._make_ob("bybit", bids=[(30050, 1)], asks=[(30060, 1)])
        snap = MarketDataService.compute_spread(ob_long, ob_short)
        assert snap.spread_abs == pytest.approx(30050 - 30000)
        assert snap.spread_pct == pytest.approx((30050 - 30000) / 30000 * 100)

    def test_negative_spread(self):
        ob_long = self._make_ob("binance", bids=[(29990, 1)], asks=[(30100, 1)])
        ob_short = self._make_ob("bybit", bids=[(30050, 1)], asks=[(30060, 1)])
        snap = MarketDataService.compute_spread(ob_long, ob_short)
        assert snap.spread_abs == pytest.approx(30050 - 30100)

    def test_empty_orderbook(self):
        ob_long = self._make_ob("binance", bids=[], asks=[])
        ob_short = self._make_ob("bybit", bids=[(30050, 1)], asks=[(30060, 1)])
        snap = MarketDataService.compute_spread(ob_long, ob_short)
        assert snap.spread_abs == 0.0
        assert snap.spread_pct == 0.0
