# Contributing to AlphaForge

Thanks for your interest in contributing! AlphaForge is a Python
backtesting framework, and contributions of all kinds are welcome —
bug reports, new signals, documentation improvements, tests, and features.

This guide explains how to set up the project, run the test suite, and submit a
high-quality pull request.

By participating in this project, you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

---

## Prerequisites

- **Python 3.11+** (CI runs against 3.11 and 3.12; the tested floors —
  pandas 3.0, numpy 2.4, scipy 1.17, scikit-learn 1.8 — require 3.11)
- **pip** and a virtual environment tool (`venv` is fine)
- **Node 18+** and **npm** — only required if you work on the web frontend
- **git**

---

## Setup

Clone the repository and install the package in editable mode with the dev
extras (pytest, pytest-cov, jupyter):

```bash
git clone https://github.com/ROROK20/AlphaForge.git
cd AlphaForge

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[torch]"      # PyTorch LSTM signal
pip install -e ".[live]"       # Alpaca paper-order client
```

If you use the Alpaca paper-order client, copy the example environment
file and fill in your keys (never commit real secrets):

```bash
cp .env.example .env
```

---

## Running the Test Suite

The full suite must pass before a PR can be merged.

```bash
# Run everything
pytest tests/

# Verbose
pytest tests/ -v

# With coverage report
pytest tests/ --cov=alphaforge --cov-report=term-missing

# A single module
pytest tests/test_backtest.py -v
```

---

## Linting

We use [Ruff](https://github.com/astral-sh/ruff) for linting (matching CI):

```bash
pip install ruff==0.17.3   # the exact version CI installs
ruff check alphaforge/ --ignore E501,E402
```

Please make sure `ruff check` is clean before opening a PR. Type hints are
encouraged throughout the codebase — keep public APIs annotated.

---

## Frontend (Web Demo)

The interactive demo lives in `frontend/` (Next.js 15.5.28, React 19, TypeScript, Tailwind).

```bash
cd frontend
npm install
npm run dev        # serves http://localhost:3001
```

Before pushing frontend changes, verify both the linter and the production
build succeed:

```bash
npm run lint
npm run build
```

---

## Code Style

- **Python:** follow PEP 8, keep functions small and vectorized where possible,
  and add type hints. New signals must subclass `alphaforge.signals.base.Signal`
  and return a `pd.Series` of `{-1, 0, +1}` aligned to the input index.
- **Tests:** every new feature or bug fix should come with tests under
  `tests/`. Aim to keep coverage from regressing.
- **Docs:** update the relevant docs in `docs/` and the `README.md` when you
  change public behavior.
- **Frontend:** follow the existing TypeScript / React conventions; run
  `npm run lint` before committing.

---

## Branch & Pull Request Workflow

1. **Fork** the repository and create a feature branch off `main`:

   ```bash
   git checkout -b feat/my-new-signal
   ```

2. Make your changes with focused, logical commits.
3. Ensure `pytest tests/` and `ruff check` pass (and, for frontend work,
   `npm run build`).
4. Push your branch and open a Pull Request against `main`.
5. Fill in the [pull request template](.github/PULL_REQUEST_TEMPLATE.md) and
   link any related issue.
6. A maintainer will review; please respond to feedback and keep the branch up
   to date.

For large or breaking changes, please
[open an issue](https://github.com/ROROK20/AlphaForge/issues) first to
discuss the design.

---

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/). Format:

```
<type>(<optional scope>): <short summary>
```

Common types:

| Type       | Use for                                            |
| ---------- | -------------------------------------------------- |
| `feat`     | A new feature                                      |
| `fix`      | A bug fix                                           |
| `docs`     | Documentation only changes                          |
| `test`     | Adding or fixing tests                              |
| `refactor` | Code change that neither fixes a bug nor adds a feat |
| `perf`     | A performance improvement                           |
| `chore`    | Build process, tooling, or dependency changes       |
| `ci`       | CI configuration changes                            |

Examples:

```
feat(signals): add momentum breakout signal
fix(backtest): correct commission applied on flat positions
docs(readme): clarify Alpaca env setup
```

---

## Reporting Bugs & Requesting Features

Use the GitHub issue forms:

- [Bug report](https://github.com/ROROK20/AlphaForge/issues/new?template=bug_report.yml)
- [Feature request](https://github.com/ROROK20/AlphaForge/issues/new?template=feature_request.yml)

For security vulnerabilities, **do not** open a public issue — see
[SECURITY.md](SECURITY.md).

Thank you for helping make AlphaForge better!
