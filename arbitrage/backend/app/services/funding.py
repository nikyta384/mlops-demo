"""
Funding Calculator
──────────────────
Pure-function module for funding-related maths.

Funding rates on perp exchanges are typically quoted per 8 h.
Convention used here: rate > 0  → longs pay shorts,
                      rate < 0  → shorts pay longs.
"""

from __future__ import annotations


class FundingCalculator:

    @staticmethod
    def net_funding(rate_long: float, rate_short: float) -> float:
        """
        Net funding when you are:
          • LONG on connector_long  → you PAY  rate_long  (if rate_long > 0)
          • SHORT on connector_short→ you RECEIVE rate_short (if rate_short > 0)

        net = rate_short - rate_long

        Positive net_funding ⟹ the strategy receives funding (carry positive).
        """
        return rate_short - rate_long

    @staticmethod
    def expected_carry_8h(
        net_funding: float,
        position_size: float,
        mark_price: float,
    ) -> float:
        """
        Expected dollar carry per 8-hour funding period.

        carry = net_funding * position_size * mark_price
        """
        return net_funding * position_size * mark_price

    @staticmethod
    def annualized_yield(net_funding_8h: float) -> float:
        """
        Convert 8-hour funding rate to annualized percentage yield.
        3 funding periods/day × 365 days = 1095 periods/year.
        """
        return net_funding_8h * 1095 * 100  # percent

    @staticmethod
    def breakeven_spread(
        entry_fee_rate: float = 0.0004,  # 0.04% maker/taker typical
        exit_fee_rate: float = 0.0004,
    ) -> float:
        """
        Minimum spread (as fraction) required to break even on round-trip fees.
        Both legs pay entry + exit fees.
        """
        return 2 * (entry_fee_rate + exit_fee_rate)
