"""Record the web demo's offline snapshot from the REAL engine.

    python scripts/record_engine_snapshot.py               # recordedOn = today (UTC)
    python scripts/record_engine_snapshot.py --date 2026-11-02

When the dashboard cannot reach the API it shows these files, and only for the
exact configuration they were recorded with, under a "Recorded snapshot" badge
that names the recording date. Nothing is generated in the browser.

Each file is one ``POST /backtest`` response, produced in-process through the
same FastAPI handler the live dashboard calls (``fastapi.testclient``), so the
equity curve, trade log and metrics are the engine's own output on the demo
API's synthetic feed (seeded GBM prices, NOT market data for the ticker), and
the trade log reconciles with the equity curve exactly as it does live.

``tests/test_engine_snapshot.py`` re-records every file in memory and fails if
the committed snapshot does not match the engine, so an engine change cannot
leave a stale offline demo behind: re-run this script and commit the result.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = ROOT / "frontend" / "public" / "engine-snapshot"

# The dashboard's default form (frontend/app/page.tsx) -- the configuration a
# visitor lands on. Commission is the per-side fraction the form sends (0.1%).
REQUEST = {
    "symbol": "AAPL",
    "startDate": "2020-01-01",
    "endDate": "2024-01-01",
    "initialCapital": 100000,
    "commission": 0.001,
    "slippage": 0.0005,
    "fill": "close",
}
STRATEGIES = ("sma_crossover", "rsi_meanrev", "macd", "ml_momentum")


def record(strategy: str) -> dict:
    """The engine's response for the default request with *strategy*."""
    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient

    from alphaforge.api.app import app

    resp = TestClient(app).post("/backtest", json={**REQUEST, "strategy": strategy})
    resp.raise_for_status()
    return resp.json()


def snapshot(strategy: str, recorded_on: str) -> dict:
    return {
        "kind": "alphaforge-engine-snapshot",
        "recordedOn": recorded_on,
        "recordedWith": "scripts/record_engine_snapshot.py (POST /backtest, in-process)",
        "request": {**REQUEST, "strategy": strategy},
        "response": record(strategy),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="recordedOn date written into every file (default: today, UTC)",
    )
    args = parser.parse_args(argv)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    for strategy in STRATEGIES:
        data = snapshot(strategy, args.date)
        path = SNAPSHOT_DIR / f"{strategy}.json"
        path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
        m = data["response"]["metrics"]
        print(f"{path.relative_to(ROOT)}: {m['nTrades']} trades, total return {m['totalReturn']:+.4%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
