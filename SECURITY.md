# Security Policy

## Supported Versions

| Version | Supported |
| --- | --- |
| `main` and the latest 0.1.x | Yes |
| Older snapshots and forks | No |

## Reporting a Vulnerability

Please do not report vulnerabilities through public issues, pull requests or
discussions. Email **[rohan.s.khiani.vora@gmail.com](mailto:rohan.s.khiani.vora@gmail.com)**
or use GitHub's
[private vulnerability reporting](https://github.com/ROROK20/AlphaForge/security/advisories/new).

Include what you can: the impact, steps to reproduce or a proof of concept, the
affected module and configuration, and any suggested fix. We aim to acknowledge
within **48 hours** and to confirm and assess severity within **7 days**. We
disclose after a fix is available and credit the reporter unless asked not to.
Good-faith research under this policy will not face legal action from the
maintainers.

AlphaForge is research and educational software for backtesting and paper orders.
It is not intended for trading real capital, and it ships with no warranty (see
[LICENSE](LICENSE)).

## Dependency Audits (run 2026-11-04)

Advisory databases move faster than this file. Re-run the commands rather than
trusting it.

**Frontend**, Next.js 15.5.28 and React 19, on the committed `package-lock.json`
(Node 23.9, npm 10.9):

```bash
cd frontend && npm audit              # 0 critical / 0 high / 6 moderate
cd frontend && npm audit --omit=dev   # 0 vulnerabilities
```

The 6 moderates are one dev-only advisory and the packages above it:
GHSA-g3hv-75wg-m249 (denial of service via glob parts amplification) in the
`minimatch` 10.2.5 under `@typescript-eslint/typescript-estree`, plus the five
`@typescript-eslint/*` packages `eslint-config-next` brings in. They run when
`npm run lint` reads this repo's own source; none is in a production install or the
shipped bundle. `npm audit fix` is not applied: by its own dry run it still leaves 6
moderates.

`frontend/package.json` overrides Next's own `postcss` to this repo's 8.5.32
(`"overrides": {"next": {"postcss": "$postcss"}}`, pinned by
`tests/test_packaging.py`), because the 8.4.31 Next 15 asks for carries high
advisories. Re-check this on every Next upgrade.

**Python base install**, against the exact pins (pip-audit 2.10.1):

```bash
pip-audit -r requirements.txt         # No known vulnerabilities found
```

`pip-audit` resolves a requirements file, so this result describes the shipped set
only because the pins are `==`. The `torch>=2.13.0` floor and the build backend's
`setuptools>=83` floor in `pyproject.toml` keep an install from resolving back to a
version with a known advisory.

**The `[live]` extra is outside that result.** `pip install "alphaforge[live]"` pulls
`alpaca-trade-api<4`, which pins legacy `urllib3 1.26.x` and `msgpack 1.0.x` carrying
known advisories (one CVSS 7.5). It is excluded from the base install and loaded
lazily, only for the paper-order client. No SDK version has run green here:
`tests/test_paper_trader.py` drives `PaperTrader` against a fake client. On Python
3.12 its `msgpack==1.0.3` pin has no binary wheel, so that install compiles from
source, untested. Install the extra only if you accept that.

## Deploying the API

- **Input bounds.** Every endpoint bounds array lengths, date ranges and simulation
  sizes, and checks values: money must be finite and positive, commission within
  `[0, 0.1]`, symbols 1–32 characters, supplied bars with positive prices and unique
  dates. Anything else is a `422`, and a `422` does not echo a huge or unencodable
  input back.
- **CORS is `*` and there is no authentication**, by design for a read-only demo that
  computes and stores nothing. For anything beyond that, restrict `allow_origins` to
  your frontend and add an API key and rate limiting.
- **Secrets.** `.env` is git-ignored and only `.env.example` placeholders are tracked.
  `PaperTrader` reads the `ALPACA_*` values from `.env` in the current directory or its
  nearest parent that has one, and exports nothing from that file into the process. It
  refuses any endpoint other than Alpaca's paper API unless constructed with
  `allow_live=True`.
- **Data-provider keys stay out of logs.** `requests` puts the full URL, query string
  included, into its exception messages. `DataFetcher` masks the Alpha Vantage key (raw
  and URL-encoded) and any `apikey=` value before logging, pinned by
  `tests/test_data.py::TestAlphaVantageFallback`.
