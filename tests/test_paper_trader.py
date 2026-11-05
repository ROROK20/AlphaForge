"""PaperTrader must only ever talk to Alpaca's PAPER endpoint unless told otherwise.

An explicit ``base_url`` beats ``ALPACA_BASE_URL``, so ``PaperTrader(base_url=<paper
URL>)`` on a machine whose environment points at the live endpoint stays on paper, and
any other endpoint is refused. ``alpaca-trade-api`` is not installed in the test
environment, so no client is built and nothing touches the network.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from alphaforge.live import paper_trader
from alphaforge.live.paper_trader import PaperTrader

PAPER = "https://paper-api.alpaca.markets"
LIVE = "https://api.alpaca.markets"


@pytest.fixture(autouse=True)
def _no_client(monkeypatch, tmp_path):
    monkeypatch.setattr(PaperTrader, "_init_api", lambda self: None)
    # Away from any developer's real .env in the checkout.
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def dotenv_file(monkeypatch, tmp_path):
    for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / ".env"
    path.write_text("ALPACA_API_KEY=key-from-file\nALPACA_SECRET_KEY=secret-from-file\nUNRELATED=1\n")
    return path


def test_the_documented_env_file_supplies_the_credentials(dotenv_file):
    """`cp .env.example .env`, fill in the keys, `PaperTrader()`: the keys are read."""
    trader = PaperTrader()
    assert (trader.api_key, trader.secret_key) == ("key-from-file", "secret-from-file")
    assert trader.base_url == PAPER


def test_the_env_file_is_found_from_a_subdirectory(dotenv_file, monkeypatch):
    sub = dotenv_file.parent / "notebooks" / "deep"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    assert PaperTrader().api_key == "key-from-file"


def test_the_environment_and_arguments_beat_the_env_file(dotenv_file, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key-from-environment")
    assert PaperTrader().api_key == "key-from-environment"
    assert PaperTrader().secret_key == "secret-from-file"
    assert PaperTrader(api_key="key-from-argument").api_key == "key-from-argument"


def test_a_live_url_in_the_env_file_is_refused_as_any_other(dotenv_file):
    dotenv_file.write_text(f"ALPACA_BASE_URL={LIVE}\n")
    with pytest.raises(ValueError, match="allow_live"):
        PaperTrader()
    assert PaperTrader(base_url=PAPER).base_url == PAPER


def test_reading_the_env_file_exports_nothing_into_the_process(dotenv_file):
    import os

    PaperTrader()
    assert not {"ALPACA_API_KEY", "ALPACA_SECRET_KEY", "UNRELATED"} & set(os.environ)


def test_no_env_file_is_no_error(monkeypatch):
    for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    trader = PaperTrader()
    assert (trader.api_key, trader.secret_key, trader.base_url) == ("", "", PAPER)


def test_explicit_paper_url_beats_a_live_environment(monkeypatch):
    monkeypatch.setenv("ALPACA_BASE_URL", LIVE)
    assert PaperTrader(api_key="k", secret_key="s", base_url=PAPER).base_url == PAPER


def test_a_live_url_from_the_environment_is_refused(monkeypatch):
    monkeypatch.setenv("ALPACA_BASE_URL", LIVE)
    with pytest.raises(ValueError, match="paper"):
        PaperTrader(api_key="k", secret_key="s")


@pytest.mark.parametrize(
    "url",
    [LIVE, "http://paper-api.alpaca.markets", "https://paper-api.alpaca.markets.evil.example"],
)
def test_any_non_paper_url_is_refused(monkeypatch, url):
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="allow_live"):
        PaperTrader(base_url=url)


def test_defaults_to_the_paper_endpoint(monkeypatch):
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)
    assert PaperTrader().base_url == PAPER
    monkeypatch.setenv("ALPACA_BASE_URL", PAPER + "/")
    assert PaperTrader().base_url == PAPER


def test_live_endpoint_needs_an_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)
    assert PaperTrader(base_url=LIVE, allow_live=True).base_url == LIVE


def test_close_all_positions_counts_what_it_closed(monkeypatch):
    """The count is of the positions closed, not of those left AFTER closing."""
    class FakeApi:
        def __init__(self):
            self.open = [object(), object(), object()]
        def list_positions(self):
            return [type("P", (), {"symbol": "X", "qty": 1, "avg_entry_price": 1, "market_value": 1,
                                   "unrealized_pl": 0, "unrealized_plpc": 0})() for _ in self.open]
        def close_all_positions(self):
            self.open = []

    trader = PaperTrader()
    trader._api = FakeApi()
    assert trader.close_all_positions() == 3


class _FakeRest:
    """The parts of the SDK client that the documented snippet reaches."""

    def get_account(self):
        return SimpleNamespace(equity="1000", cash="500", buying_power="500", portfolio_value="1000", status="ACTIVE")

    def list_positions(self):
        return [SimpleNamespace(symbol="AAPL", qty="10", avg_entry_price="50", market_value="500",
                                unrealized_pl="0", unrealized_plpc="0")]

    def submit_order(self, **kwargs):
        return SimpleNamespace(status="accepted", id="1")


def test_the_usage_guide_snippet_runs(monkeypatch, capsys):
    """The usage guide's snippet runs as written, against a fake client."""
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)
    monkeypatch.setattr(PaperTrader, "_init_api", lambda self: _FakeRest())
    usage = (Path(__file__).resolve().parent.parent / "docs" / "usage.md").read_text()
    code = re.search(r"```python\n(from alphaforge\.live\.paper_trader.*?)```", usage, re.S).group(1)
    exec(code, {})
    printed = capsys.readouterr().out
    assert "'equity': 1000.0" in printed and "symbol='AAPL'" in printed


def test_the_module_docstring_only_names_methods_that_exist():
    names = re.findall(r"trader\.(\w+)\(", paper_trader.__doc__)
    assert names
    assert [n for n in names if not hasattr(PaperTrader, n)] == []


def test_the_docs_do_not_sell_it_as_live_trading():
    """The docs call this a paper-order client: no signal reaches an order."""
    import re
    from pathlib import Path

    root = Path(__file__).parent.parent
    for name in ("README.md", "docs/usage.md", "docs/architecture.md", "CONTRIBUTING.md", ".env.example",
                 "alphaforge/live/paper_trader.py"):
        assert not re.search(r"live[ /-]paper", (root / name).read_text(), re.I), name
    readme = " ".join((root / "README.md").read_text().split())
    assert "Alpaca Paper-Order Client" in readme
    assert "nothing connects a signal to an order" in readme


def test_nothing_in_the_library_connects_a_signal_to_an_order():
    """What the docs claim: only alphaforge/live itself names PaperTrader."""
    from pathlib import Path

    import alphaforge

    package = Path(alphaforge.__file__).parent
    users = [
        path for root in (package, package.parent / "strategies") for path in root.rglob("*.py")
        if "PaperTrader" in path.read_text() and path.parent.name != "live"
    ]
    assert users == []
