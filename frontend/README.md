# AlphaForge Frontend

Interactive backtest studio built with Next.js 15 + Tailwind + Recharts.

## Quick start

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3001

## Features

- Live backtest configuration (symbol / strategy / dates / capital)
- 12 metrics: CAGR, Sharpe, Sortino, Calmar, Max DD, Win Rate, Profit Factor, etc.
- Equity curve vs buy-and-hold benchmark
- Drawdown chart
- Trade-by-trade log with P&L

## Data source

Results come from the real AlphaForge engine over HTTP (`POST /backtest` at `NEXT_PUBLIC_API_URL`, inlined at build time; only `npm run dev` defaults to `http://localhost:8003`, and a production build without it calls no API). The engine's demo prices are synthetic GBM seeded per symbol — one path per ticker, shared by every strategy — not market data for the ticker named.

## Build for production

```bash
NEXT_PUBLIC_API_URL=http://localhost:8003 npm run build   # baked in at build time
npm start
npm test                   # unit tests (URL resolver)
npm run check:prerender    # after a build: no localhost URL shipped, honest prerender
```

Deploys to Vercel as-is — set `NEXT_PUBLIC_API_URL` to your API. Without a
reachable API the dashboard shows only `public/engine-snapshot/` (the real
engine's recorded output for the default configuration, badged "Recorded
snapshot" with its date) and no results for anything else; there is no
in-browser engine. Re-record with `python scripts/record_engine_snapshot.py`
from the repo root after any engine change (`tests/test_engine_snapshot.py`
fails until you do).
