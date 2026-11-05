"""xgboost and torch in one process, in the environment a user has: no pin.

tests/conftest.py pins OpenMP to one thread for this test process, which would hide
the failure: unguarded, on macOS the README's ML flow as one script deadlocks
(XGBoost, then LSTM) or segfaults (LSTM, then XGBoost). Each case here runs in a child process
with that pin removed and a timeout, because a deadlock can only be seen from
outside.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import alphaforge


@pytest.fixture(autouse=True)
def _needs_both_libraries():
    pytest.importorskip("torch")
    pytest.importorskip("xgboost")


# A few seconds on an idle machine, over a minute on a heavily loaded one; a
# deadlock never ends.
TIMEOUT_SECONDS = 300

FLOW = """
    import warnings
    warnings.simplefilter("ignore")
    {first_import}
    from alphaforge.api.app import synthetic_ohlcv
    from alphaforge.signals.ml_signals import LSTMSignal, XGBoostSignal

    df = synthetic_ohlcv("AAPL", "2018-01-01", "2023-12-29")
    train = df.iloc[: int(len(df) * 0.7)]

    def xgb():
        XGBoostSignal(target_horizon=5, n_estimators=50).fit(train).generate(df)

    def lstm():
        LSTMSignal(hidden_size=16, seq_len=20, epochs=1, random_state=0).fit(train).generate(df)

    for step in ({steps}):
        step()
    print("finished")
"""


def run_flow(steps: str, first_import: str = "", **env_extra: str) -> subprocess.CompletedProcess:
    script = textwrap.dedent(FLOW).format(steps=steps, first_import=first_import)
    env = {k: v for k, v in os.environ.items() if k not in {"OMP_NUM_THREADS", "MKL_NUM_THREADS"}}
    env["PYTHONPATH"] = str(Path(alphaforge.__file__).resolve().parent.parent)
    env.update(env_extra)
    try:
        return subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, env=env, timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{steps}: no result in {TIMEOUT_SECONDS} s: deadlocked")


@pytest.mark.parametrize(
    "steps, first_import, env",
    [
        pytest.param("xgb, lstm", "", {}, id="xgboost-then-lstm"),
        pytest.param("lstm, xgb", "", {}, id="lstm-then-xgboost"),
        pytest.param("xgb, lstm, xgb, lstm", "import torch", {}, id="torch-imported-first"),
        pytest.param("lstm, xgb, lstm, xgb", "import xgboost", {}, id="xgboost-imported-first"),
        pytest.param("xgb, lstm", "", {"OMP_NUM_THREADS": "8"}, id="a-caller-asks-for-8-threads"),
    ],
)
def test_both_models_finish_in_one_process(steps, first_import, env):
    done = run_flow(steps, first_import, **env)
    assert done.returncode == 0, f"exit {done.returncode}: {done.stderr[-500:]}"
    assert "finished" in done.stdout


def test_the_callers_thread_counts_are_restored():
    """A fit limits itself, not the rest of the caller's process."""
    done = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import warnings
            warnings.simplefilter("ignore")
            import torch
            import xgboost
            from alphaforge.signals.ml_signals import _torch_threads, _xgboost_threads

            torch.set_num_threads(3)
            xgboost.set_config(nthread=3)
            with _torch_threads(), _xgboost_threads():
                pass
            print(torch.get_num_threads(), xgboost.get_config()["nthread"])
        """)],
        capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
        env={**os.environ, "PYTHONPATH": str(Path(alphaforge.__file__).resolve().parent.parent)},
    )
    assert done.returncode == 0, done.stderr[-500:]
    assert done.stdout.split() == ["3", "3"]
