"""The web demo's offline mode shows only recorded REAL engine output: committed
snapshots recorded through the real ``POST /backtest`` handler, which must reconcile,
stay current with the engine, and be the only offline data source. There is no
in-browser generator.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = ROOT / "frontend" / "public" / "engine-snapshot"
FRONTEND_SRC = [ROOT / "frontend" / d for d in ("app", "lib", "components")]


STRATEGIES = ("sma_crossover", "rsi_meanrev", "macd", "ml_momentum")


@functools.cache
def _recorder():
    spec = importlib.util.spec_from_file_location(
        "record_engine_snapshot", ROOT / "scripts" / "record_engine_snapshot.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load(strategy: str) -> dict:
    return json.loads((SNAPSHOT_DIR / f"{strategy}.json").read_text())


def test_no_in_browser_generator_ships():
    """The frontend must not be able to fabricate a backtest."""
    assert not (ROOT / "frontend" / "lib" / "mockBacktest.ts").exists()
    offenders = []
    for src in FRONTEND_SRC:
        for path in src.rglob("*.ts*"):
            text = path.read_text()
            for pattern in (r"Math\.random", r"mulberry32", r"runMockBacktest", r"signalStrength"):
                if re.search(pattern, text):
                    offenders.append(f"{path.relative_to(ROOT)}: {pattern}")
    assert not offenders, offenders


def test_there_is_one_snapshot_per_strategy_and_nothing_else():
    names = sorted(p.stem for p in SNAPSHOT_DIR.glob("*.json"))
    assert names == sorted(STRATEGIES) == sorted(_recorder().STRATEGIES)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_snapshot_says_what_it_is(strategy):
    snap = _load(strategy)
    assert snap["kind"] == "alphaforge-engine-snapshot"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", snap["recordedOn"])
    assert snap["request"] == {**_recorder().REQUEST, "strategy": strategy}
    resp = snap["response"]
    assert resp["source"] == "engine"
    assert resp["dataSource"] == "synthetic"
    assert resp["strategy"] == strategy


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_snapshot_trade_log_reconciles_with_its_equity_curve(strategy):
    """The reconciliation invariant on the offline data itself: sum of net trade P&L == equity P&L,
    up to the wire's 2-dp rounding of each figure."""
    resp = _load(strategy)["response"]
    capital = _recorder().REQUEST["initialCapital"]
    trades = resp["trades"]
    assert trades, "the recorded run must trade"
    gap = abs(sum(t["pnlNetAbs"] for t in trades) - (resp["equity"][-1]["equity"] - capital))
    assert gap <= 0.005 * (len(trades) + 1) + 1e-9, gap
    assert resp["metrics"]["nTrades"] == len(trades)
    final = resp["equity"][-1]["equity"]
    assert resp["metrics"]["totalReturn"] == pytest.approx(final / capital - 1, abs=1e-7)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_snapshot_is_current_with_the_engine(strategy):
    """Re-record in memory: a committed snapshot that does not match the
    engine is stale -- run scripts/record_engine_snapshot.py and commit."""
    committed = _load(strategy)["response"]
    fresh = _recorder().record(strategy)
    assert fresh["strategyExecuted"] == committed["strategyExecuted"]
    assert fresh["note"] == committed["note"]
    assert [(t["entryDate"], t["exitDate"], t["direction"]) for t in fresh["trades"]] == [
        (t["entryDate"], t["exitDate"], t["direction"]) for t in committed["trades"]
    ]
    for got, want in zip(fresh["trades"], committed["trades"]):
        for key in ("entryPrice", "exitPrice", "pnlAbs", "commissionAbs", "pnlNetAbs"):
            assert got[key] == pytest.approx(want[key], abs=0.011), key
    assert len(fresh["equity"]) == len(committed["equity"])
    for got, want in zip(fresh["equity"], committed["equity"]):
        assert got["date"] == want["date"]
        assert got["equity"] == pytest.approx(want["equity"], abs=0.011)
        assert got["benchmark"] == pytest.approx(want["benchmark"], abs=0.011)
    for key, want in committed["metrics"].items():
        got = fresh["metrics"][key]
        assert (got is None) == (want is None), key
        if want is not None:
            assert got == pytest.approx(want, rel=1e-9, abs=1e-12), key


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_snapshot_rows_show_the_close_of_the_date_beside_them(strategy):
    """The committed rows (what the offline dashboard shows, and the hero image's
    trade log): logged price == close(logged date) x (1 +/- slippage)."""
    from alphaforge.api.app import synthetic_ohlcv

    req, resp = _recorder().REQUEST, _load(strategy)["response"]
    close = synthetic_ohlcv(req["symbol"], req["startDate"], req["endDate"])["Close"]
    close.index = close.index.strftime("%Y-%m-%d")
    slip, last = req["slippage"], close.index[-1]
    for t in resp["trades"]:
        d = 1 if t["direction"] == "long" else -1
        assert t["entryPrice"] == pytest.approx(close[t["entryDate"]] * (1 + d * slip), abs=0.0051)
        factor = 1.0 if t["exitDate"] == last else 1 - d * slip
        assert t["exitPrice"] == pytest.approx(close[t["exitDate"]] * factor, abs=0.0051)


def test_vs_buy_hold_is_a_gap_in_points_not_a_ratio_of_final_equities():
    """The live number is checked by e2e/smoke.mjs; this keeps the ratio formula out in CI."""
    page = (ROOT / "frontend" / "app" / "page.tsx").read_text()
    assert "last.equity / last.benchmark" not in page
    assert "fmtPts((last.equity - last.benchmark) / result.params.initialCapital)" in page


def test_dashboard_default_form_is_the_recorded_configuration():
    """The offline snapshot is only useful if it matches what a visitor lands on."""
    page = (ROOT / "frontend" / "app" / "page.tsx").read_text()
    req = _recorder().REQUEST
    assert f'useState("{req["symbol"]}")' in page
    assert f'useState("{req["startDate"]}")' in page
    assert f'useState("{req["endDate"]}")' in page
    assert "useState(100_000)" in page and req["initialCapital"] == 100_000
    assert "useState(0.1)" in page and req["commission"] == 0.001  # 0.1% / 100
    assert 'useState<Fill>("close")' in page and req["fill"] == "close"
    assert "const SLIPPAGE = 0.0005;" in page and req["slippage"] == 0.0005


def test_the_social_card_is_the_state_a_frontend_only_deploy_shows():
    """The card must not unfurl "Live engine" for a page badged "Recorded snapshot".
    The README's hero image stays the live engine."""
    import struct

    smoke = (ROOT / "e2e" / "smoke.mjs").read_text()
    card = re.search(r'\{ path: join\(FRONTEND, "public", "og\.png"\).*\}', smoke).group(0)
    assert 'badge: "Recorded snapshot"' in card and "offline: true" in card
    hero = re.search(r'\{ path: join\(ROOT, "docs", "screenshot\.png"\).*\}', smoke).group(0)
    assert 'badge: "Live engine"' in hero and "offline: false" in hero

    layout = (ROOT / "frontend" / "app" / "layout.tsx").read_text()
    description = re.search(r'const OG_DESCRIPTION =\s*"([^"]+)"', layout).group(1)
    assert "recorded output" in description and "synthetic prices" in description
    assert "powered by" not in description

    png = (ROOT / "frontend" / "public" / "og.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", png[16:24]) == (2400, 1260)  # the declared 1200x630, at 2x

