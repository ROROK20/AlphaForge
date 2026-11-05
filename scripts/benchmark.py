"""Time the operations the docs quote, on synthetic data, and print a table.

    OMP_NUM_THREADS=1 python scripts/benchmark.py            # median of 5 runs each

Every figure in docs/architecture.md "Performance" comes from this script; the
docs say on which date, machine and Python it was run. Numbers depend on the
machine and its load, so treat them as orders of magnitude, and re-run before
quoting. Nothing here touches the network.
"""

from __future__ import annotations

import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from alphaforge.api.app import synthetic_ohlcv  # noqa: E402
from alphaforge.backtest.engine import BacktestEngine  # noqa: E402
from alphaforge.optimization.markowitz import MarkowitzOptimizer  # noqa: E402
from alphaforge.risk.var import monte_carlo_var  # noqa: E402
from alphaforge.signals.ml_signals import XGBoostSignal  # noqa: E402
from alphaforge.signals.technical import (  # noqa: E402
    BollingerBandSignal, MACDSignal, RSIMeanReversionSignal, SMACrossoverSignal,
)

REPEATS = 5


def timed(fn) -> float:
    """Median wall-clock seconds of REPEATS calls (after one warm-up call)."""
    fn()
    samples = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def main() -> int:
    five_years = synthetic_ohlcv("AAPL", "2019-01-01", "2023-12-31")    # 1,304 bars
    twenty_years = synthetic_ohlcv("AAPL", "2004-01-01", "2023-12-31")  # 5,217 bars
    engine = BacktestEngine()
    sma = SMACrossoverSignal(20, 50).generate(five_years)
    sma20 = SMACrossoverSignal(20, 50).generate(twenty_years)
    returns = five_years["Close"].pct_change().dropna()
    rng = np.random.default_rng(0)
    assets = pd.DataFrame(rng.normal(0.0004, 0.01, (1260, 50)), columns=[f"A{i}" for i in range(50)])
    xgb_frame = synthetic_ohlcv("MSFT", "2019-01-01", "2023-12-31").iloc[:1000]

    rows = [
        (f"BacktestEngine.run, precomputed SMA signal, {len(five_years):,} bars",
         lambda: engine.run(sma, five_years)),
        (f"BacktestEngine.run, precomputed SMA signal, {len(twenty_years):,} bars",
         lambda: engine.run(sma20, twenty_years)),
    ]
    for sig in (SMACrossoverSignal(20, 50), RSIMeanReversionSignal(14), MACDSignal(), BollingerBandSignal()):
        rows.append((f"{type(sig).__name__}.generate + run, {len(five_years):,} bars",
                     lambda s=sig: engine.run(s, five_years)))
    rows += [
        ("Monte Carlo VaR, 100,000 paths, 10-day horizon",
         lambda: monte_carlo_var(returns, n_simulations=100_000, horizon=10)),
        ("Markowitz max_sharpe, 50 assets x 1,260 days",
         lambda: MarkowitzOptimizer(assets).optimize("max_sharpe")),
        ("XGBoostSignal.fit, 1,000 bars (default 300 trees)",
         lambda: XGBoostSignal().fit(xgb_frame)),
    ]

    print(f"# {time.strftime('%Y-%m-%d')} | {platform.system()} {platform.machine()} | "
          f"Python {platform.python_version()} | median of {REPEATS}")
    print("| Operation | Time |\n|---|---|")
    for label, fn in rows:
        secs = timed(fn)
        shown = f"{secs * 1000:,.1f} ms" if secs < 1 else f"{secs:,.2f} s"
        print(f"| {label} | {shown} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
