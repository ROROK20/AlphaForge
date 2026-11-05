"""Tests for the event-driven Portfolio bookkeeping (cash, positions, NAV): a short
sale credits cash, the buy-back debits it, and NAV falls as the shorted price rises.
"""

from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.backtest.portfolio import Portfolio, Position

CAP = 100_000.0


class TestLong:
    def test_open_mark_close_without_costs(self):
        p = Portfolio(CAP)
        p.open_position("A", 100, 50.0, direction=1, commission=0.0)
        assert p.cash == pytest.approx(95_000.0)
        assert p.nav({"A": 50.0}) == pytest.approx(CAP)
        assert p.nav({"A": 60.0}) == pytest.approx(101_000.0)  # long gains as price rises
        pnl = p.close_position("A", 60.0, commission=0.0)
        assert pnl == pytest.approx(1_000.0)
        assert p.cash == pytest.approx(101_000.0) and not p.positions

    def test_round_trip_pnl_equals_the_change_in_cash_with_costs(self):
        p = Portfolio(CAP)
        p.open_position("A", 100, 50.0, direction=1, commission=0.001)
        pnl = p.close_position("A", 60.0, commission=0.001)
        # price P&L 1000 - entry fee 5.00 - exit fee 6.00
        assert pnl == pytest.approx(989.0)
        assert p.cash - CAP == pytest.approx(pnl)

    def test_a_long_larger_than_cash_is_cut_to_what_cash_pays_for(self):
        p = Portfolio(1_000.0)
        p.open_position("A", 100, 50.0, direction=1, commission=0.01)
        assert p.cash == pytest.approx(0.0, abs=1e-9)
        assert p.positions["A"].quantity == pytest.approx(1_000.0 / (50.0 * 1.01))


class TestShort:
    def test_short_100_at_100_credits_cash_and_loses_as_price_rises(self):
        """Short 100 @ 100, no costs: NAV 90,000 at a price of 200, and closing at 50
        reports +5,000 with cash at 105,000."""
        p = Portfolio(CAP)  # public API only: open / snapshot / close / cash
        p.open_position("S", 100, 100.0, direction=-1, commission=0.0)
        assert p.cash == pytest.approx(110_000.0)            # sale proceeds credited
        nav = lambda px: p.snapshot(pd.Timestamp("2024-01-02"), {"S": px})["nav"]  # noqa: E731
        assert nav(100.0) == pytest.approx(CAP)
        assert nav(200.0) == pytest.approx(90_000.0)  # short loses as price rises
        pnl = p.close_position("S", 50.0, commission=0.0)
        assert pnl == pytest.approx(5_000.0)
        assert p.cash == pytest.approx(105_000.0)             # cash agrees with P&L
        assert p.cash - CAP == pytest.approx(pnl)

    def test_a_losing_short_round_trip_with_costs(self):
        p = Portfolio(CAP)
        p.open_position("S", 10, 100.0, direction=-1, commission=0.002)
        assert p.cash == pytest.approx(CAP + 1_000.0 - 2.0)
        pnl = p.close_position("S", 130.0, commission=0.002)
        # price P&L -300 - entry fee 2.00 - exit fee 2.60
        assert pnl == pytest.approx(-304.6)
        assert p.cash - CAP == pytest.approx(pnl)

    def test_equity_at_cost_basis_counts_a_short_as_a_liability(self):
        p = Portfolio(CAP)
        p.open_position("S", 10, 100.0, direction=-1, commission=0.0)
        assert p.positions["S"].market_value == pytest.approx(-1_000.0)
        assert p.equity == pytest.approx(CAP)


class TestPartialAndMixed:
    def test_partial_close_keeps_the_rest_at_its_entry_price(self):
        p = Portfolio(CAP)
        p.open_position("A", 100, 10.0, direction=1, commission=0.001)
        pnl = p.close_position("A", 12.0, commission=0.001, quantity=40)
        # 40 * 2 = 80 - 40% of the 1.00 entry fee - 0.48 exit fee
        assert pnl == pytest.approx(80.0 - 0.4 - 0.48)
        rest = p.positions["A"]
        assert rest.quantity == pytest.approx(60) and rest.avg_entry_price == 10.0
        assert rest.entry_commission == pytest.approx(0.6)
        total = pnl + p.close_position("A", 11.0, commission=0.001)
        assert p.cash - CAP == pytest.approx(total)

    def test_partial_close_of_a_short(self):
        p = Portfolio(CAP)
        p.open_position("S", 50, 20.0, direction=-1, commission=0.0)
        assert p.close_position("S", 18.0, commission=0.0, quantity=20) == pytest.approx(40.0)
        assert p.positions["S"].signed_quantity == pytest.approx(-30)
        assert p.nav({"S": 18.0}) == pytest.approx(CAP + 50 * 2.0)

    def test_long_and_short_together_mark_independently(self):
        p = Portfolio(CAP)
        p.open_position("L", 100, 10.0, direction=1, commission=0.0)
        p.open_position("S", 100, 10.0, direction=-1, commission=0.0)
        # A pair that moves together nets to zero.
        assert p.nav({"L": 15.0, "S": 15.0}) == pytest.approx(CAP)
        assert p.nav({"L": 15.0, "S": 10.0}) == pytest.approx(CAP + 500.0)

    def test_adding_to_a_side_averages_and_the_other_side_is_refused(self):
        p = Portfolio(CAP)
        p.open_position("S", 10, 100.0, direction=-1, commission=0.0)
        p.open_position("S", 10, 110.0, direction=-1, commission=0.0)
        assert p.positions["S"].avg_entry_price == pytest.approx(105.0)
        with pytest.raises(ValueError, match="close it before"):
            p.open_position("S", 5, 100.0, direction=1)

    @pytest.mark.parametrize("bad", [{"quantity": 0}, {"quantity": 11}])
    def test_close_quantity_must_be_held(self, bad):
        p = Portfolio(CAP)
        p.open_position("A", 10, 10.0, direction=1)
        with pytest.raises(ValueError):
            p.close_position("A", 10.0, **bad)

    def test_equity_curve_and_reset(self):
        p = Portfolio(CAP)
        p.open_position("S", 10, 100.0, direction=-1, commission=0.0)
        for day, px in (("2024-01-02", 100.0), ("2024-01-03", 90.0)):
            p.snapshot(pd.Timestamp(day), {"S": px})
        assert list(p.equity_curve()) == pytest.approx([CAP, CAP + 100.0])
        p.reset()
        assert p.cash == CAP and not p.positions and p.equity_curve().empty

    def test_position_value_helpers(self):
        pos = Position("S", 10, 100.0, -1)
        assert pos.value_at(120.0) == pytest.approx(-1_200.0)
        assert pos.unrealized_pnl(120.0) == pytest.approx(-200.0)



class TestNoNegativeOrZeroFills:
    """After a short loss drives cash below zero, a long must not open a NEGATIVE
    number of shares, nor zero cash a 0-share position."""

    def test_a_long_with_negative_cash_is_refused(self):
        p = Portfolio(10_000.0)
        p.open_position("S", 100, 100.0, direction=-1, commission=0.0)   # cash 20,000
        p.close_position("S", 300.0, commission=0.0)                     # cash -10,000
        assert p.cash == pytest.approx(-10_000.0)
        with pytest.raises(ValueError):
            p.open_position("A", 10, 5.0, direction=1, commission=0.0)
        assert "A" not in p.positions

    def test_a_long_with_zero_cash_is_refused(self):
        p = Portfolio(1_000.0)
        p.open_position("A", 10, 100.0, direction=1, commission=0.0)     # all cash spent
        assert p.cash == pytest.approx(0.0)
        with pytest.raises(ValueError, match="no cash"):
            p.open_position("B", 5, 10.0, direction=1, commission=0.0)
        assert "B" not in p.positions

    def test_nothing_opens_once_equity_is_gone(self):
        p = Portfolio(10_000.0)
        p.open_position("S", 100, 100.0, direction=-1, commission=0.0)
        p.close_position("S", 300.0, commission=0.0)                     # equity -10,000
        with pytest.raises(ValueError, match="equity"):
            p.open_position("T", 1, 10.0, direction=-1, commission=0.0)
