"use client";

import { useState, useEffect, type ReactNode } from "react";
import type { BacktestParams, BacktestResult, Fill, Strategy } from "@/lib/types";
import { ApiRejected, loadRecordedSnapshot, runRealBacktest } from "@/lib/api";
import MetricCard from "@/components/MetricCard";
import EquityChart from "@/components/EquityChart";
import DrawdownChart from "@/components/DrawdownChart";
import TradesTable from "@/components/TradesTable";
import { Play, Github, TrendingUp, Wifi, WifiOff, Loader2, Archive } from "lucide-react";

/**
 * Where the numbers on screen come from. There is no in-browser engine.
 *
 * - loading:  connecting to the API; no results shown yet.
 * - live:     the response of a live POST /backtest.
 * - recorded: API unreachable, and this exact configuration has a recorded
 *             snapshot of real engine output (scripts/record_engine_snapshot.py).
 * - offline:  API unreachable and nothing recorded for this configuration;
 *             no results are shown.
 * - rejected: the API answered and refused the request (e.g. an end date in
 *             the future); its explanation is shown.
 */
type DataMode = "loading" | "live" | "recorded" | "offline" | "rejected";

interface ModeCopy {
  badge: string;
  icon: ReactNode;
  badgeClass: string;
  title: string;
  footer: (recordedOn?: string) => string;
}

// A TOTAL map over DataMode: adding a mode without its copy is a type error.
const MODE_COPY: Record<DataMode, ModeCopy> = {
  loading: {
    badge: "Connecting…",
    icon: <Loader2 className="w-3.5 h-3.5 animate-spin" />,
    badgeClass: "border-border text-gray-400",
    title: "Connecting to the AlphaForge engine API.",
    footer: () => "Connecting to the AlphaForge engine…",
  },
  live: {
    badge: "Live engine",
    icon: <Wifi className="w-3.5 h-3.5" />,
    badgeClass: "border-accent-green/40 bg-accent-green/10 text-accent-green",
    title: "Results computed just now by the AlphaForge engine API.",
    footer: () => "Live results from the AlphaForge engine",
  },
  recorded: {
    badge: "Recorded snapshot",
    icon: <Archive className="w-3.5 h-3.5" />,
    badgeClass: "border-yellow-500/40 bg-yellow-500/10 text-yellow-500",
    title: "Engine API unreachable — showing recorded output of the real engine for this exact configuration.",
    footer: (on) => `Recorded engine output (recorded ${on ?? "?"}) — the engine API is offline`,
  },
  rejected: {
    badge: "Request rejected",
    icon: <WifiOff className="w-3.5 h-3.5" />,
    badgeClass: "border-accent-red/40 bg-accent-red/10 text-accent-red",
    title: "The engine API answered and refused this request — see the message below.",
    footer: () => "The engine rejected this request — no results",
  },
  offline: {
    badge: "Engine offline",
    icon: <WifiOff className="w-3.5 h-3.5" />,
    badgeClass: "border-border bg-panel text-gray-400",
    title: "Engine API unreachable and no recording exists for this configuration — no results.",
    footer: () => "Engine offline — no results (nothing is simulated in the browser)",
  },
};

const STRATEGIES: { id: Strategy; name: string; description: string }[] = [
  { id: "sma_crossover", name: "SMA Crossover", description: "20 / 50 day moving averages" },
  { id: "rsi_meanrev", name: "RSI Mean Reversion", description: "Buy oversold, sell overbought" },
  { id: "macd", name: "MACD", description: "Long after a bullish signal-line cross, short after a bearish one (held between crosses)" },
  { id: "ml_momentum", name: "Momentum (SMA demo proxy)", description: "The XGBoost walk-forward takes seconds to minutes per run, so this public demo runs SMA 10/30" },
];

// The demo's fixed per-side slippage (fraction of price), sent with every
// request and shown next to commission: both are charged on traded notional.
const SLIPPAGE = 0.0005;

const SYMBOLS = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN", "META", "SPY"];

const FILLS: { id: Fill; name: string; caption: string }[] = [
  // Short names: the select is one grid column wide (the long form, which
  // truncated in the README screenshot, is the caption below the form).
  { id: "close", name: "Same close", caption: "at the signal bar's close (market-on-close)" },
  { id: "next_close", name: "Next close", caption: "one bar after the signal (next close)" },
];

export default function Home() {
  const [symbol, setSymbol] = useState("AAPL");
  const [strategy, setStrategy] = useState<Strategy>("sma_crossover");
  const [startDate, setStartDate] = useState("2020-01-01");
  const [endDate, setEndDate] = useState("2024-01-01");
  const [capital, setCapital] = useState(100_000);
  const [commission, setCommission] = useState(0.1);
  const [fill, setFill] = useState<Fill>("close");

  // No result until the engine (or its recorded snapshot) answers. The
  // prerendered HTML is therefore the "Connecting…" state with empty metrics,
  // never a number nobody computed.
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [mode, setMode] = useState<DataMode>("loading");
  const [rejection, setRejection] = useState<string | null>(null);

  useEffect(() => {
    const params: BacktestParams = {
      symbol,
      strategy,
      startDate,
      endDate,
      initialCapital: capital,
      commission: commission / 100,
      slippage: SLIPPAGE,
      fill,
    };

    let cancelled = false;
    setMode("loading");
    setResult(null);
    setRejection(null);

    runRealBacktest(params)
      .then((real) => {
        if (cancelled) return;
        setResult(real);
        setMode("live");
      })
      .catch(async (err: unknown) => {
        if (err instanceof ApiRejected) {
          // The engine is up and said no: say why, show nothing else.
          if (cancelled) return;
          setRejection(err.message);
          setMode("rejected");
          return;
        }
        // Engine unreachable: show its recorded output for this exact
        // configuration if one exists, otherwise nothing.
        const recorded = await loadRecordedSnapshot(params);
        if (cancelled) return;
        setResult(recorded);
        setMode(recorded ? "recorded" : "offline");
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol, strategy, startDate, endDate, capital, commission, fill]);

  const m = result?.metrics ?? null;
  // null = no result yet / offline, or an undefined ratio (zero denominator)
  // from the engine: shown as "—", never as a misleading 0.
  const fmt = (n: number | null | undefined) =>
    n == null ? "—" : n.toLocaleString("en-US", { maximumFractionDigits: 2 });
  const fmtPct = (n: number | null | undefined) => (n == null ? "—" : `${(n * 100).toFixed(2)}%`);
  // A difference of two returns: percentage POINTS. The ratio of final equities
  // minus 1 reads as a return gap and is not one (-45.60% for a gap of -89.42 points).
  const fmtPts = (n: number) => `${n > 0 ? "+" : ""}${(n * 100).toFixed(2)} pts`;
  const last = result?.equity[result.equity.length - 1];
  const copy = MODE_COPY[mode];

  return (
    <main className="min-h-screen">
      {/* Header */}
      <header className="border-b border-border bg-panel/30 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-gradient-to-br from-accent-blue to-accent-purple rounded-lg flex items-center justify-center glow">
              <TrendingUp className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">AlphaForge</h1>
              <p className="text-xs text-gray-500">Quantitative Trading Studio</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <ModeBadge mode={mode} />
            <a
              href="https://github.com/ROROK20/AlphaForge"
              className="flex items-center gap-2 px-3 py-1.5 text-xs border border-border hover:border-accent-blue rounded-md transition-colors"
            >
              <Github className="w-3.5 h-3.5" />
              View Source
            </a>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Controls */}
        <div className="bg-panel border border-border rounded-lg p-5 mb-6">
          <div className="flex items-center gap-2 mb-4">
            <Play className="w-4 h-4 text-accent-blue" />
            <h2 className="text-sm font-semibold">Backtest Configuration</h2>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-7 gap-4">
            <div>
              <label className="text-xs text-gray-500 mb-1 block">Symbol</label>
              <select
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                className="w-full bg-bg border border-border rounded px-2 py-1.5 text-sm focus:border-accent-blue outline-none"
              >
                {SYMBOLS.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>
            <div className="md:col-span-2">
              <label className="text-xs text-gray-500 mb-1 block">Strategy</label>
              <select
                value={strategy}
                onChange={(e) => setStrategy(e.target.value as Strategy)}
                className="w-full bg-bg border border-border rounded px-2 py-1.5 text-sm focus:border-accent-blue outline-none"
              >
                {STRATEGIES.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-xs text-gray-500 mb-1 block">Start</label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full bg-bg border border-border rounded px-2 py-1.5 text-sm focus:border-accent-blue outline-none"
              />
            </div>
            <div>
              <label className="text-xs text-gray-500 mb-1 block">End</label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full bg-bg border border-border rounded px-2 py-1.5 text-sm focus:border-accent-blue outline-none"
              />
            </div>
            <div>
              <label className="text-xs text-gray-500 mb-1 block">Fill</label>
              <select
                value={fill}
                onChange={(e) => setFill(e.target.value as Fill)}
                className="w-full bg-bg border border-border rounded px-2 py-1.5 text-sm focus:border-accent-blue outline-none"
              >
                {FILLS.map((f) => (
                  <option key={f.id} value={f.id}>
                    {f.name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-xs text-gray-500 mb-1 block">Capital ($)</label>
              <input
                type="number"
                value={capital}
                onChange={(e) => setCapital(Number(e.target.value))}
                className="w-full bg-bg border border-border rounded px-2 py-1.5 text-sm focus:border-accent-blue outline-none"
              />
            </div>
          </div>
          <p className="text-xs text-gray-500 mt-3">
            Strategy: {STRATEGIES.find((s) => s.id === strategy)?.description} ·{" "}
            Costs per side: {commission}% commission + {(SLIPPAGE * 100).toFixed(2)}% slippage · Fills{" "}
            {FILLS.find((f) => f.id === fill)?.caption} · Prices: synthetic (seeded per
            symbol), not market data
          </p>
          {result?.note && (
            <p className="text-xs text-yellow-500 mt-2" data-testid="strategy-note">
              ⚠ {result.note}
            </p>
          )}
        </div>

        {result?.ruined && (
          <div
            className="border border-accent-red/40 bg-accent-red/10 rounded-lg px-4 py-3 mb-6 text-xs text-accent-red"
            data-testid="ruin-notice"
          >
            Ruined on {result.ruinDate}: equity reached $0, the position was liquidated at that
            bar&apos;s close and nothing traded afterwards. Metrics cover the path through that
            bar.
          </div>
        )}

        {mode === "rejected" && (
          <div
            className="border border-accent-red/40 bg-accent-red/10 rounded-lg px-4 py-3 mb-6 text-xs text-accent-red"
            data-testid="rejected-notice"
          >
            The engine rejected this request: {rejection}
          </div>
        )}

        {(mode === "recorded" || mode === "offline") && (
          <div
            className="border border-yellow-500/30 bg-yellow-500/5 rounded-lg px-4 py-3 mb-6 text-xs text-gray-300"
            data-testid="offline-notice"
          >
            {mode === "recorded" ? (
              <>
                The engine API is unreachable (or this deployment has none configured). Below
                is a <strong>recorded snapshot</strong> of
                the real engine&apos;s output for exactly this configuration, recorded{" "}
                {result?.recordedOn} by <code>scripts/record_engine_snapshot.py</code> on the
                API&apos;s synthetic {symbol} feed. Change any setting and there is no
                recording — start the API (<code>make api</code>) for live runs.
              </>
            ) : (
              <>
                The engine API is unreachable (or not configured) and this configuration has no recorded
                snapshot, so there are no results — nothing is simulated in the browser.
                Start the API (<code>make api</code>) to run it.
              </>
            )}
          </div>
        )}

        {/* Metrics row */}
        <div
          className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 mb-6"
          data-testid="metrics"
        >
          <MetricCard
            label="Total Return"
            value={fmtPct(m?.totalReturn)}
            trend={m == null ? "neutral" : m.totalReturn >= 0 ? "up" : "down"}
          />
          <MetricCard
            label="CAGR"
            value={fmtPct(m?.cagr)}
            trend={m == null ? "neutral" : m.cagr >= 0 ? "up" : "down"}
          />
          <MetricCard
            label="Sharpe"
            value={fmt(m?.sharpe)}
            trend={m?.sharpe != null && m.sharpe >= 1 ? "up" : "neutral"}
          />
          <MetricCard label="Sortino" value={fmt(m?.sortino)} />
          <MetricCard
            label="Max DD"
            value={fmtPct(m?.maxDrawdown)}
            trend={m == null ? "neutral" : "down"}
          />
          <MetricCard label="Calmar" value={fmt(m?.calmar)} />
          <MetricCard label="Vol (ann)" value={fmtPct(m?.volatility)} />
          <MetricCard label="Win Rate" value={fmtPct(m?.winRate)} />
          <MetricCard
            label="Profit Factor"
            value={fmt(m?.profitFactor)}
            trend={m?.profitFactor == null ? "neutral" : m.profitFactor > 1 ? "up" : "down"}
          />
          <MetricCard label="# Trades" value={m == null ? "—" : m.nTrades.toString()} />
          <MetricCard
            label="Final Equity"
            value={last ? `$${last.equity.toLocaleString()}` : "—"}
          />
          <MetricCard
            label="vs Buy-Hold"
            value={last && result ? fmtPts((last.equity - last.benchmark) / result.params.initialCapital) : "—"}
          />
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
          <div className="lg:col-span-2">
            <EquityChart data={result?.equity ?? []} />
          </div>
          <DrawdownChart data={result?.equity ?? []} />
        </div>

        {/* Trades */}
        <TradesTable
          trades={result?.trades ?? []}
          initialCapital={result?.params.initialCapital}
          emptyMessage={
            mode === "loading"
              ? "Waiting for the engine…"
              : mode === "offline"
                ? "No results — the engine API is offline and this configuration was not recorded."
                : mode === "rejected"
                  ? "No results — the engine rejected this request."
                  : "This run made no trades."
          }
        />

        {/* Footer */}
        <footer className="mt-12 pt-6 border-t border-border flex flex-wrap items-center justify-between gap-2 text-xs text-gray-500">
          <p>
            AlphaForge · Open-source quant trading framework ·{" "}
            <a
              href="https://github.com/ROROK20/AlphaForge"
              className="underline hover:text-accent-blue transition-colors"
              target="_blank"
              rel="noreferrer"
            >
              GitHub
            </a>{" "}
            ·{" "}
            <a
              href="https://github.com/ROROK20/AlphaForge/blob/main/LICENSE"
              className="underline hover:text-accent-blue transition-colors"
              target="_blank"
              rel="noreferrer"
            >
              MIT License
            </a>{" "}
            ·{" "}
            {/* From the same total map as the badge (MODE_COPY). */}
            <span data-testid="mode-footer">{copy.footer(result?.recordedOn)}</span>
          </p>
          <p>v0.1.0</p>
        </footer>
      </div>
    </main>
  );
}

function ModeBadge({ mode }: { mode: DataMode }) {
  const copy = MODE_COPY[mode];
  return (
    <span
      className={`flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-md border ${copy.badgeClass}`}
      title={copy.title}
      data-testid="mode-badge"
    >
      {copy.icon}
      {copy.badge}
    </span>
  );
}
