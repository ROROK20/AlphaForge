"""Shared test configuration.

OpenMP single-threading: xgboost and torch each bundle their own OpenMP
runtime (libomp). On macOS, when both runtimes spin worker pools inside one
process, the LSTM training loop deadlocks (reproduced 2026-07-06: the full
suite hung forever at TestLSTMSignal::test_generate_runs_end_to_end, while
the same test alone passed in <2s). Pinning OpenMP to one thread BEFORE
either library loads sidesteps the duplicate-runtime contention and makes
the tiny test models deterministic-fast. This pin is test-process-only policy;
what the library itself does about the two runtimes (one thread each while it
fits and scores, on macOS, where both are installed) is in the docstring of
alphaforge/signals/ml_signals.py, and test_ml_openmp.py tests it with this pin
removed.
"""

import os

# Must happen before xgboost/torch are imported anywhere in the test run.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def pytest_configure(config):
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
