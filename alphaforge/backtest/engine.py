"""
Backtesting engine: shares held at entry, an exact trade log, and ruin.

A position is a number of SHARES, fixed when it is entered (equity x target
fraction / fill price) and held until the signal's target changes. Between
changes everything -- positions, equity, per-bar P&L -- is numpy array
arithmetic over the bars; the simulation steps only from one position change
to the next, and the trade log iterates once per trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from alphaforge.signals.base import Signal


def _shown(value: Optional[float], spec: str) -> str:
    """A metric as text: "n/a" when it has no value, never a 0 that reads as a result."""
    return "n/a" if value is None or value != value else format(value, spec)


@dataclass
class BacktestResult:
    """Container for all backtest output."""

    equity_curve: pd.Series
    # Net return of each bar (equity(t) / equity(t-1) - 1). The first element
    # is a structural zero (no position and no interval before the first
    # bar), so ``returns.iloc[1:]`` are the per-interval returns
    # ``annualized_return`` expects.
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)
    signal_series: pd.Series = field(default_factory=pd.Series)
    # The bar on which equity reached zero, if it did (see BacktestEngine,
    # "Ruin"). None for a run that stayed solvent.
    ruin_date: Optional[pd.Timestamp] = None

    @property
    def ruined(self) -> bool:
        return self.ruin_date is not None

    def __repr__(self) -> str:
        m = self.metrics
        ruin = f", RUINED {self.ruin_date:%Y-%m-%d}" if self.ruined else ""
        return (
            f"BacktestResult("
            f"cagr={_shown(m.get('cagr'), '.2%')}, "
            f"sharpe={_shown(m.get('sharpe'), '.2f')}, "
            f"max_drawdown={_shown(m.get('max_drawdown'), '.2%')}, "
            f"trades={len(self.trades)}{ruin})"
        )


# Bars between the bar a signal is computed on and the first bar its position
# earns: 1 = filled at that bar's close (market-on-close), 2 = a bar later.
FILLS = {"close": 1, "next_close": 2}

# The only columns a fill may be priced from, compared lower-case: a bar's
# close, adjusted or not. Everything else a bar carries (Open, High, Low, VWAP)
# is a price from BEFORE the close the signal may have read. See
# BacktestEngine, "Execution model".
_CLOSE_COLUMNS = frozenset({
    "close", "adj close", "adj_close", "adjclose", "adjusted close", "adjusted_close",
})

_TRADE_COLUMNS = [
    "entry_date", "exit_date", "direction", "shares", "entry_price", "exit_price",
    "pnl_pct", "pnl_abs", "commission_abs", "pnl_net_abs", "ruin_writeoff_abs",
]


@dataclass
class _Simulation:
    """Bar-by-bar state of a simulated book (n bars x m assets)."""

    equity: np.ndarray        # (n,) equity at each bar's close; floored at 0 after ruin
    shares: np.ndarray        # (n, m) signed shares held DURING each bar
    fees: np.ndarray          # (n, m) traded notional |d shares| x fill price, charged on each bar
    ruin: Optional[int]       # bar on which equity reached <= 0, if any
    ruin_scale: float         # k = -E(r-1) / dE_uncapped(r): the ruin bar's floor factor


class BacktestEngine:
    """
    Single- and multi-asset backtesting engine.

    Execution model
    ---------------
    A signal value computed on bar *t* may use bar *t*'s data, including its
    close. The position it asks for is filled at bar *t*'s close and held from
    bar *t + 1*, so no bar's return is ever earned by a signal that saw that
    bar -- the market-on-close assumption of vectorized backtests
    (``fill="close"``, the default). That is optimistic for a signal that needs
    the close in order to be computed: in practice the order goes in after the
    close and fills a bar later. ``fill="next_close"`` models exactly that --
    the signal computed on bar *t* fills at bar *t + 1*'s close.
    ``Open``/``High``/``Low`` never affect fills or P&L. (There is no next-open
    fill: the demo feed's Open is the previous Close, so it would equal
    market-on-close there.)

    ``price_col`` names the close to fill at, and may only be a close
    (``Close``, ``Adj Close`` and their spellings; anything else raises
    ``ValueError``). ``price_col="Open"`` would fill a signal computed at bar
    *t*'s close at bar *t*'s OPEN, earning the very bar the signal had read:
    ``sign(close change)`` on the demo feed's random walk, 2020-2024, would read
    Sharpe 17.03 and CAGR 634%, and MACD would go from Sharpe 0.02 to 2.10.

    Positions are shares, fixed at entry
    ------------------------------------
    A signal value is the TARGET POSITION as a fraction of equity, in
    ``[-1, 1]``: +1 fully long, -1 fully short, 0 flat, 0.5 half long. When the
    target changes, the engine buys or sells to
    ``shares = target x equity / fill price`` (equity marked at that close)
    and then HOLDS those shares until the target changes again: equity is
    ``cash + shares x price``, a long gains ``shares x (price - entry)``, a
    short loses as the price rises. Integer ``{-1, 0, +1}`` signals are the
    common case; fractional ones let a position sizer's output be backtested
    directly (``signal.generate(df) * kelly.size(capital)``); anything outside
    ``[-1, 1]``, or non-finite, raises ``ValueError`` -- leverage at entry is
    not modelled. NaN and bars missing from the signal's index mean flat.

    A target bounds the position when it is ENTERED, and nothing afterwards.
    Shares are held, not rebalanced, so exposure drifts with the price: a short's
    grows as the price rises, up to ruin. Cash pays for the rest: a full long
    holds cash of minus its entry costs, and cash earns and owes 0% either way,
    so negative cash is financed free.

    A constant fraction of equity re-applied every bar (``return = signal x
    price_return``) is the shortcut this avoids: an implicit daily rebalance that
    is never charged, under which a short is never ruined by a steady rise (100 ->
    200 in ten equal steps leaves 47.4% of capital instead of nothing), and a short
    trade's logged return disagrees with what the curve books.

    Costs
    -----
    ``commission`` and ``slippage`` are each charged per side as a fraction of
    the notional traded, ``|shares change| x fill price``: an entry or exit
    pays once, a long <-> short reversal pays on both legs, resizing pays on
    the change.

    Trade-log dates are fill dates
    ------------------------------
    ``entry_date`` and ``exit_date`` name the bar whose close IS the logged
    price: ``entry_price = close(entry_date) x (1 +/- slippage)`` and likewise
    ``exit_price``, whatever ``fill`` is. Under ``fill="close"`` a signal first
    set on bar *t* has ``entry_date`` *t*; under ``"next_close"``, *t + 1*. So
    ``entry_date`` is the bar BEFORE the first bar the trade earns,
    ``exit_date`` is the last bar it earns, and the bars between the two dates
    are the bars it held. Equity shows a fill's cost one bar after that fill's
    date: the cost is charged on the first bar the new position is held.

    Two rows have an exit that no order made: a position still open on the
    last bar is marked at that bar's close, and a ruined position is closed at
    the ruin bar's close. Their ``exit_price`` is that close with NO slippage
    (no exit cost is charged, on the row or on the curve), their ``exit_date``
    is that bar, and their ``entry_price`` is an ordinary fill.

    Ruin
    ----
    Equity cannot go below zero. The first bar on which it would reach zero
    or less -- a short through a rise of 100% or more over any number of
    bars, or a long through a collapse -- ruins the account: equity is floored
    at exactly 0 on that bar, the position is closed at that bar's close, and
    nothing trades afterwards. ``BacktestResult.ruin_date`` records the bar;
    metrics are computed on the path up to and including it (total return
    -100%). The loss beyond the account -- what a broker would have to absorb
    -- is not booked: it is written back on the trade that held the ruin bar,
    in that row's ``ruin_writeoff_abs``, so the trade log still reconciles
    exactly with the equity curve.

    Inputs it refuses
    -----------------
    ``ValueError``, rather than a number that looks like a result: a negative
    or non-finite cost (commission -0.05 would pay the strategy on every trade:
    +604.84%, Sharpe 1.52 on the README's AAPL SMA crossover),
    capital that is not finite and positive, bars out of time order (a newest-first CSV
    would run backwards), duplicate dates, a price that is NaN, infinite, zero
    or negative, a ``price_col`` that is not a close (see "Execution
    model"), and bars whose spacing does not say how many make a year, unless
    ``periods_per_year`` does. There is deliberately no upper bound on a cost.

    Parameters
    ----------
    initial_capital : float
        Starting portfolio value (default 100_000).
    commission : float
        Commission per side, as a fraction of traded notional (default
        0.001 = 10 bps).
    slippage : float
        Slippage per side as a fraction of traded notional (default 0.0005 =
        5 bps).
    long_only : bool
        If True, a negative target is held flat (no short selling). Default
        False. ``run_multi`` refuses a negative weight under it.
    fill : {"close", "next_close"}
        When a signal's position is filled: at the close of the bar it was
        computed on (``"close"``, the default) or one bar later
        (``"next_close"``). See "Execution model".
    periods_per_year : float | None
        Bars per year for the annualised metrics. None (the default) reads it
        from the spacing of the bars, and refuses a spacing it cannot name, such
        as intraday (see ``alphaforge.backtest.metrics``).

    Examples
    --------
    >>> engine = BacktestEngine(initial_capital=100_000)
    >>> result = engine.run(signal, df)
    >>> print(result.metrics)
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        long_only: bool = False,
        fill: str = "close",
        periods_per_year: float | None = None,
    ) -> None:
        if fill not in FILLS:
            raise ValueError(f"fill must be one of {sorted(FILLS)}, got {fill!r}")
        if not (np.isfinite(initial_capital) and initial_capital > 0):
            raise ValueError(f"BacktestEngine: initial_capital must be finite and positive, got {initial_capital!r}.")
        for name, cost in (("commission", commission), ("slippage", slippage)):
            if not (np.isfinite(cost) and cost >= 0):
                raise ValueError(
                    f"BacktestEngine: {name} is a cost per side and must be finite and "
                    f"non-negative, got {cost!r}."
                )
        if periods_per_year is not None and not (np.isfinite(periods_per_year) and periods_per_year > 0):
            raise ValueError(
                f"BacktestEngine: periods_per_year must be finite and positive, got {periods_per_year!r}."
            )
        self.periods_per_year = periods_per_year
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.long_only = long_only
        self.fill = fill

    # ------------------------------------------------------------------
    # public entry points
    # ------------------------------------------------------------------

    def run(
        self,
        signal: Signal | pd.Series,
        data: pd.DataFrame,
        price_col: str = "Close",
    ) -> BacktestResult:
        """
        Execute a single-asset backtest.

        Parameters
        ----------
        signal : Signal or pd.Series
            Either a Signal instance (``generate`` will be called) or a
            pre-computed series of target positions in ``[-1, 1]`` (see the
            class docstring, "Positions are shares, fixed at entry").
        data : pd.DataFrame
            OHLCV DataFrame with DatetimeIndex.
        price_col : str
            The close column to fill at (default 'Close'; 'Adj Close' for
            adjusted prices). Any other column raises ``ValueError``: it would
            fill a signal at a price from before the data it read.

        Returns
        -------
        BacktestResult
        """
        self._check_price_col(price_col)
        self._check_bar_order(data.index, "the data")
        prices = self._checked_prices(data[price_col], "the data")
        raw = signal.generate(data) if isinstance(signal, Signal) else signal
        raw_signals = self._targets(raw, prices.index)
        targets = self._held_targets(raw_signals)

        sim = self._simulate(prices.to_numpy()[:, None], targets.to_numpy()[:, None], np.ones(1))
        trades = self._build_trades(sim, 0, prices, sim.shares[:, 0])
        return self._result(sim, prices.index, trades, positions=pd.DataFrame({
            "signal": self._stopped(targets, sim.ruin),
            "shares": sim.shares[:, 0],
            "price": prices,
        }), signal_series=raw_signals)

    def run_multi(
        self,
        signals_map: dict[str, pd.Series],
        data_map: dict[str, pd.DataFrame],
        weights: dict[str, float] | None = None,
    ) -> BacktestResult:
        """
        Multi-asset backtest on one shared book.

        Parameters
        ----------
        signals_map : dict[str, pd.Series]
            Mapping of symbol -> signal series (target fraction of the asset's
            weight).
        data_map : dict[str, pd.DataFrame]
            Mapping of symbol -> OHLCV DataFrame.
        weights : dict[str, float] | None
            Asset weights. Defaults to equal weight. Normalised so that
            ``sum(|w|) == 1``, which bounds what each leg is sized to when it
            enters, and NOT the book (dividing by ``sum(w)`` instead would run
            {A: 3, B: -2} at 5x gross). See Notes. A negative
            weight inverts that asset's signal, which ``long_only`` forbids: it
            would go short.

        Returns
        -------
        BacktestResult with the combined equity curve and a trade log with a
        ``symbol`` column.

        Notes
        -----
        Shares are sized from the COMBINED book's equity when an asset's
        target changes: ``w[sym] x target x book equity / fill price``. Every
        trade's P&L is its own shares' price moves less its own costs, so the
        concatenated trade log sums exactly to the book's P&L. Only bars on
        which every asset has a price (the common calendar) are simulated.

        Gross exposure can exceed 1x and cash can go negative. A leg is sized
        from the whole book's equity while the legs already held have drifted
        and are not rebalanced: {A: 0.5, B: 0.5}, A enters, quadruples, then B
        enters at half of the $250,000 book, for 1.30x gross and cash of
        -$75,000. Negative cash is financed free.
        """
        symbols = list(signals_map.keys())
        n_sym = len(symbols)
        if self.long_only and weights is not None and any(weights[s] < 0 for s in symbols):
            shorted = [s for s in symbols if weights[s] < 0]
            raise ValueError(
                f"run_multi: long_only=True cannot hold a negative weight, which goes short: {shorted}."
            )
        if weights is not None:
            gross = sum(abs(weights[s]) for s in symbols)
            if gross <= 0:
                raise ValueError("run_multi: weights must not all be zero.")
            w = np.array([weights[s] / gross for s in symbols])
        else:
            w = np.full(n_sym, 1.0 / n_sym)

        for sym in symbols:
            self._check_bar_order(data_map[sym].index, f"the data for {sym}")
        common_idx = data_map[symbols[0]].index
        for sym in symbols[1:]:
            common_idx = common_idx.intersection(data_map[sym].index)
        if len(common_idx) == 0 and any(len(data_map[s]) for s in symbols):
            raise ValueError(f"run_multi: {symbols} share no dates, so there is nothing to simulate.")

        prices = pd.DataFrame({
            s: self._checked_prices(data_map[s].loc[common_idx, "Close"], f"the data for {s}") for s in symbols
        })
        def raw_for(sym: str):
            sig = signals_map[sym]
            if isinstance(sig, Signal):
                return sig.generate(data_map[sym].loc[common_idx])
            return sig  # _targets -> _aligned refuses one that does not line up

        targets = pd.DataFrame({
            s: self._held_targets(self._targets(raw_for(s), common_idx)) for s in symbols
        })

        sim = self._simulate(prices.to_numpy(), targets.to_numpy(), w)
        logs = []
        for j, sym in enumerate(symbols):
            t = self._build_trades(sim, j, prices[sym], sim.shares[:, j])
            if not t.empty:
                t["symbol"] = sym
                logs.append(t)
        trades = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(columns=_TRADE_COLUMNS + ["symbol"])
        return self._result(sim, common_idx, trades, positions=pd.DataFrame(sim.shares, index=common_idx,
                                                                               columns=symbols))

    # ------------------------------------------------------------------
    # input checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_price_col(price_col: Any) -> None:
        """A fill may only be priced at a close: see the class docstring."""
        if str(price_col).strip().lower() not in _CLOSE_COLUMNS:
            raise ValueError(
                f"BacktestEngine: price_col={price_col!r} is not a close. A signal computed "
                f"on bar t may have read bar t's close, so filling it at an earlier price from "
                f"that bar would earn the move it already saw. Use 'Close' or 'Adj Close'; "
                f"fill='next_close' fills a bar later."
            )

    @staticmethod
    def _check_bar_order(index: pd.Index, label: str) -> None:
        """Bars must run forward in time, one per date."""
        if index.has_duplicates:
            raise ValueError(
                f"BacktestEngine: {label} has more than one bar for {index[index.duplicated()][0]}; "
                f"keep one bar per date."
            )
        if not index.is_monotonic_increasing:
            raise ValueError(
                f"BacktestEngine: {label} is not in time order, oldest first (a newest-first "
                f"CSV?), so it would run backwards. Sort it with data.sort_index()."
            )

    @staticmethod
    def _checked_prices(prices: pd.Series, label: str) -> pd.Series:
        """*prices* as floats, every one finite and positive: a fill needs a price."""
        prices = prices.astype(float)
        values = prices.to_numpy()
        bad = ~np.isfinite(values) | (values <= 0)
        if bad.any():
            first = int(np.flatnonzero(bad)[0])
            raise ValueError(
                f"BacktestEngine: {label} has {int(bad.sum())} NaN, infinite or non-positive "
                f"price(s), the first {values[first]!r} at {prices.index[first]}. Drop or fill "
                f"those bars first."
            )
        return prices

    # ------------------------------------------------------------------
    # signal handling
    # ------------------------------------------------------------------

    def _targets(self, raw: pd.Series, index: pd.Index) -> pd.Series:
        """Validated target positions in [-1, 1] on *index* (long_only applied)."""
        pos = self._as_positions(raw, index)
        return pos.clip(lower=0.0) if self.long_only else pos

    def _held_targets(self, raw_signals: pd.Series) -> pd.Series:
        """The target in force DURING each bar: the signal ``FILLS[fill]`` bars back."""
        return raw_signals.shift(FILLS[self.fill]).fillna(0.0)

    @staticmethod
    def _stopped(targets: pd.Series, ruin: Optional[int]) -> pd.Series:
        """Targets as actually held: flat after a ruin bar."""
        if ruin is None:
            return targets
        held = targets.copy()
        held.iloc[ruin + 1:] = 0.0
        return held

    @staticmethod
    def _as_positions(raw: pd.Series, index: pd.Index) -> pd.Series:
        """Validate a signal as target positions in ``[-1, 1]``, as floats.

        A fraction is held as that fraction, and anything the engine cannot model
        is an error rather than a silently different backtest.
        """
        pos = pd.to_numeric(BacktestEngine._aligned(raw, index), errors="raise").astype(float)
        pos = pos.fillna(0.0)  # missing bars / NaN = no position
        values = pos.to_numpy()
        bad = ~np.isfinite(values) | (np.abs(values) > 1.0 + 1e-9)
        if bad.any():
            first = pos.index[bad][0]
            raise ValueError(
                "BacktestEngine: a signal is a target position as a fraction of "
                "capital in [-1, 1] (+1 fully long, -1 fully short); leverage is "
                f"not modelled. Got {pos.loc[first]!r} at {first} "
                f"({int(bad.sum())} out-of-range bar(s))."
            )
        # Absorb float noise like 1.0000000000000002 from a normalisation.
        return pos.clip(-1.0, 1.0)

    @staticmethod
    def _aligned(raw: Any, index: pd.Index) -> pd.Series:
        """*raw* as a Series on *index*, refusing a signal that does not line up.

        An array or list must have exactly one value per bar. A Series is
        matched by date; one that shares no dates with the prices (a
        tz-aware vs tz-naive index, or a ``reset_index`` copy with 0..n-1
        labels) or covers under half of the bars is an error: reindexed, it would
        turn to NaN -> flat and come back as a backtest with zero trades.
        """
        if not isinstance(raw, pd.Series):
            values = np.asarray(raw, dtype=float).reshape(-1)
            if len(values) != len(index):
                raise ValueError(
                    f"BacktestEngine: signal has {len(values)} values but the data has "
                    f"{len(index)} bars; pass one value per bar, or a Series indexed by date."
                )
            return pd.Series(values, index=index)
        if raw.index.equals(index):
            return raw
        try:
            covered = float(index.isin(raw.index).mean()) if len(index) else 1.0
        except TypeError:
            covered = 0.0
        if covered < 0.5:
            hint = ""
            if isinstance(raw.index, pd.DatetimeIndex) and isinstance(index, pd.DatetimeIndex) \
                    and (raw.index.tz is None) != (index.tz is None):
                hint = " (one index is timezone-aware, the other is not)"
            elif not isinstance(raw.index, pd.DatetimeIndex):
                hint = f" (the signal's index is a {type(raw.index).__name__}, not dates)"
            raise ValueError(
                f"BacktestEngine: the signal's index matches only {covered:.0%} of the "
                f"price bars{hint}. A signal is a position per bar: reindex it to the "
                f"data's index before passing it in."
            )
        return raw.reindex(index)

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------

    def _simulate(self, prices: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> _Simulation:
        """Step through the bars on which any asset's target changes.

        On such a bar *k* every asset whose target changed trades, at bar
        *k - 1*'s close, to ``weight x target x equity(k-1) / price(k-1)``
        shares (others keep theirs); the traded notional is charged
        ``commission + slippage`` on bar *k*. Between changes the shares are
        constant, so each segment's equity is ``equity(k-1) - costs +
        sum(shares x (price - price(k-1)))`` -- array arithmetic over the
        segment's bars. The first bar whose equity is <= 0 is the ruin bar.
        """
        n, m = prices.shape
        rate = self.commission + self.slippage
        equity = np.full(n, float(self.initial_capital))
        shares = np.zeros((n, m))
        fees = np.zeros((n, m))
        if n == 0:
            return _Simulation(equity, shares, fees, None, 1.0)

        changed = np.zeros(n, dtype=bool)
        changed[1:] = (targets[1:] != targets[:-1]).any(axis=1)
        changes = np.flatnonzero(changed)
        held = np.zeros(m)
        ruin: Optional[int] = None
        ruin_scale = 1.0
        for i, k in enumerate(changes):
            base = equity[k - 1]
            ref = prices[k - 1]
            moved = targets[k] != targets[k - 1]
            new = held.copy()
            new[moved] = weights[moved] * targets[k, moved] * base / ref[moved]
            fees[k] = np.abs(new - held) * ref
            held = new
            end = changes[i + 1] if i + 1 < len(changes) else n
            seg = equity[k:end] = base - fees[k].sum() * rate + (prices[k:end] - ref) @ held
            shares[k:end] = held
            blown = np.flatnonzero(seg <= 0.0)
            if blown.size:
                ruin = k + int(blown[0])
                prev = equity[ruin - 1]
                ruin_scale = -prev / (equity[ruin] - prev)
                equity[ruin:] = 0.0
                shares[ruin + 1:] = 0.0
                break
        return _Simulation(equity, shares, fees, ruin, ruin_scale)

    def _result(
        self,
        sim: _Simulation,
        index: pd.Index,
        trades: pd.DataFrame,
        positions: pd.DataFrame,
        signal_series: Optional[pd.Series] = None,
    ) -> BacktestResult:
        equity = pd.Series(sim.equity, index=index)
        prev = np.r_[self.initial_capital, sim.equity[:-1]]
        with np.errstate(divide="ignore", invalid="ignore"):
            rets = np.where(prev > 0, sim.equity / np.where(prev > 0, prev, 1.0) - 1.0, 0.0)
        returns = pd.Series(rets, index=index)

        # Metrics on the path through the ruin bar, if ruined; ratio
        # statistics skip the structural pre-position bars (no position can
        # exist before the first fill), which would otherwise enter every
        # Sharpe/Sortino as returns of exactly 0.
        from alphaforge.backtest.metrics import compute_metrics, infer_periods_per_year
        end = len(equity) if sim.ruin is None else sim.ruin + 1
        warm = min(FILLS[self.fill], end)
        # From the whole run's bars: a run ruined on bar 1 has the same spacing.
        periods = self.periods_per_year if self.periods_per_year is not None else infer_periods_per_year(index)
        metrics = compute_metrics(equity.iloc[:end], returns.iloc[warm:end], trades, periods_per_year=periods)
        return BacktestResult(
            equity_curve=equity,
            returns=returns,
            positions=positions,
            trades=trades,
            metrics=metrics,
            signal_series=signal_series if signal_series is not None else pd.Series(dtype=float),
            ruin_date=None if sim.ruin is None else index[sim.ruin],
        )

    # ------------------------------------------------------------------
    # trade log
    # ------------------------------------------------------------------

    def _build_trades(
        self,
        sim: _Simulation,
        j: int,
        prices: pd.Series,
        held: np.ndarray,
    ) -> pd.DataFrame:
        """Asset *j*'s trades: an EXACT decomposition of its part of the curve.

        A trade is a run of bars holding shares on the same side; a fractional
        target may resize it on the way. Each bar's P&L,
        ``shares(t) x (price(t) - price(t-1))``, belongs to the trade holding
        bar *t*; each fill's cost belongs to the trade it opens, resizes or
        closes (a reversal's two legs split between the trade it closes and
        the one it opens). So:

        - ``pnl_abs``: the trade's price P&L less its slippage (gross of
          commission); ``commission_abs``: its commission; ``pnl_net_abs =
          pnl_abs - commission_abs``. Summed over every trade of every asset,
          ``pnl_net_abs`` equals ``equity[-1] - initial_capital`` to float
          precision -- the reconciliation invariant the suite pins.
        - ``shares``: signed shares at entry. For a trade that was not resized,
          ``shares x (exit_close - entry_close)`` IS its price P&L.
        - ``entry_date`` / ``exit_date``: the bars whose closes were filled (see
          the class docstring, "Trade-log dates are fill dates"); the number
          of bars between them is the number of bars the trade held.
        - ``entry_price`` / ``exit_price``: the fills, slippage-adjusted
          against the trade (an open position's final mark and a ruined
          position's close are not fills: no slippage); ``pnl_pct`` is the
          return between them, signed by ``direction`` (the first fill's, for
          a resized trade).
        - ``ruin_writeoff_abs``: 0 except on the trade that held (or closed on)
          the ruin bar -- the loss beyond the account, which is not booked; it
          is included in ``pnl_abs``.
        """
        n = len(held)
        px = prices.to_numpy()
        side = np.sign(held)
        rows: list[dict[str, Any]] = []
        if n == 0 or not side.any():
            return pd.DataFrame(columns=_TRADE_COLUMNS)

        moves = np.zeros(n)
        moves[1:] = held[1:] * (px[1:] - px[:-1])
        fill_px = np.r_[px[0], px[:-1]]        # a change on bar t fills at close(t-1)
        slip_rate, comm_rate = self.slippage, self.commission
        bar = 1
        while bar < n:
            if side[bar] == 0:
                bar += 1
                continue
            start, d = bar, int(side[bar])
            while bar + 1 < n and side[bar + 1] == d:
                bar += 1
            last = bar                          # last bar held
            # Notional legs this trade owns: the entry leg (only this side's
            # part of a reversal), resizes inside the run, and the exit leg.
            legs = abs(held[start]) * fill_px[start]
            if last > start:
                legs += float(np.sum(np.abs(np.diff(held[start:last + 1])) * fill_px[start + 1:last + 1]))
            closed_by_change = last + 1 < n and not (sim.ruin is not None and last >= sim.ruin)
            if closed_by_change:
                legs += abs(held[last]) * px[last]
                exit_price = px[last] * (1 - d * slip_rate)
            else:                               # mark-to-market close, or the ruin bar
                exit_price = px[last]
            entry_price = fill_px[start] * (1 + d * slip_rate)
            pnl_abs = float(moves[start:last + 1].sum()) - slip_rate * legs
            commission_abs = comm_rate * legs
            rows.append({
                "entry_date": prices.index[start - 1],  # the fill bar: bar 0 holds nothing, so start >= 1
                "exit_date": prices.index[last],
                "direction": d,
                "shares": float(held[start]),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_pct": d * (exit_price - entry_price) / entry_price,
                "pnl_abs": pnl_abs,
                "commission_abs": commission_abs,
                "pnl_net_abs": pnl_abs - commission_abs,
                "ruin_writeoff_abs": 0.0,
                "_span": (start, last),
            })
            bar = last + 1

        if sim.ruin is not None:
            self._write_off_ruin(rows, sim, j, moves)
        for row in rows:
            del row["_span"]
        return pd.DataFrame(rows, columns=_TRADE_COLUMNS)

    def _write_off_ruin(self, rows: list[dict[str, Any]], sim: _Simulation, j: int, moves: np.ndarray) -> None:
        """Book this asset's share of the ruin bar's floor on the trade that owns it.

        On the ruin bar every P&L component is scaled by ``k`` (so the bar's
        parts sum to exactly -equity(ruin - 1)); this asset's write-off is
        ``(k - 1) x (its bar P&L - its bar costs)``. The owner is the trade
        holding the ruin bar, else the one whose closing leg was charged on it.
        """
        r = sim.ruin
        bar_part = moves[r] - (self.commission + self.slippage) * sim.fees[r, j]
        writeoff = (sim.ruin_scale - 1.0) * bar_part
        if writeoff == 0.0:
            return
        owner = next((row for row in rows if row["_span"][0] <= r <= row["_span"][1]), None)
        if owner is None:
            owner = next((row for row in rows if row["_span"][1] + 1 == r), None)
        if owner is None:
            return
        owner["ruin_writeoff_abs"] += writeoff
        owner["pnl_abs"] += writeoff
        owner["pnl_net_abs"] += writeoff
