"use client";

import { useMemo, useState, useEffect, type ReactNode } from "react";
import type { BacktestParams, BacktestResult, Fill, Strategy } from "@/lib/types";
import { API_URL, ApiRejected, loadRecordedSnapshot, runRealBacktest } from "@/lib/api";
import { type DataMode, modeTitle, recordedBecause } from "@/lib/modes";
import { benchmarkReturn, gapPhrase, maxDrawdownFromCurve, plainReturn } from "@/lib/reading";
import Entry from "@/components/Entry";
import StatsTable from "@/components/StatsTable";
import EquityChart from "@/components/EquityChart";
import DrawdownChart from "@/components/DrawdownChart";
import TradesTable from "@/components/TradesTable";

interface ModeCopy {
  badge: string;
  /** The small square beside the badge: solid only when an engine just answered. */
  markClass: string;
  badgeClass: string;
  footer: (recordedOn?: string) => ReactNode;
  /** The headline when there is no result to state, and any line under it. */
  noResult: string;
  noResultDetail?: string;
  /** Room held for the headline, so a run swapping for the next one does not jump the page. */
  heroSpace: string;
  /** What each figure says over its blank paper. */
  figureEmpty: string;
  tradesEmpty: string;
}

const RESULT_HEIGHT = "min-h-[12rem] md:min-h-[8.25rem]";

// A production build with no NEXT_PUBLIC_API_URL has no engine to call: the hosted
// demo. Its visitors cannot start one, so the offline notices tell them how to run
// the stack themselves instead of naming a command for a server they do not have,
// and the badge's tooltip and the footer (lib/modes.ts) do not call it unreachable.
const NO_ENGINE = API_URL === null;

// A TOTAL map over DataMode: adding a mode without its copy is a type error.
// Each badge's tooltip is modeTitle() in lib/modes.ts, the same kind of map.
const MODE_COPY: Record<DataMode, ModeCopy> = {
  loading: {
    badge: "Connecting…",
    markClass: "border-rule-strong",
    badgeClass: "text-ink-2",
    footer: () => "Connecting to the AlphaForge engine…",
    noResult: "Connecting to the AlphaForge engine.",
    noResultDetail: "No figures appear until the engine, or its recording of this exact configuration, answers.",
    heroSpace: RESULT_HEIGHT,
    figureEmpty: "Waiting for the engine…",
    tradesEmpty: "Waiting for the engine…",
  },
  live: {
    badge: "Live engine",
    markClass: "border-plot bg-plot",
    badgeClass: "text-ink",
    footer: () => "Live results from the AlphaForge engine",
    noResult: "The engine returned no result.",
    heroSpace: RESULT_HEIGHT,
    figureEmpty: "Nothing to plot.",
    tradesEmpty: "This run made no trades.",
  },
  recorded: {
    badge: "Recorded snapshot",
    markClass: "border-ink",
    badgeClass: "text-ink",
    footer: (on) => (
      <>
        Recorded engine output (recorded <span className="whitespace-nowrap">{on ?? "?"}</span>) —{" "}
        {recordedBecause(NO_ENGINE)}
      </>
    ),
    noResult: "The recording holds no result.",
    heroSpace: RESULT_HEIGHT,
    figureEmpty: "Nothing to plot.",
    tradesEmpty: "This run made no trades.",
  },
  rejected: {
    badge: "Request rejected",
    markClass: "border-pen bg-pen",
    badgeClass: "text-pen",
    footer: () => "The engine rejected this request — no results",
    noResult: "The engine refused this request.",
    heroSpace: "",
    figureEmpty: "No result to plot.",
    tradesEmpty: "No results — the engine rejected this request.",
  },
  offline: {
    badge: "Engine offline",
    markClass: "border-ink-3",
    badgeClass: "text-ink-2",
    footer: () => "Engine offline — no results (nothing is simulated in the browser)",
    noResult: "No result for this configuration.",
    heroSpace: "",
    figureEmpty: "No result to plot.",
    tradesEmpty: "No results — the engine API is offline and this configuration was not recorded.",
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
  // Short names: the select is one grid column wide. The long form is the
  // caption below the form and the sentence under the headline.
  { id: "close", name: "Same close", caption: "at the signal bar's close (market-on-close)" },
  { id: "next_close", name: "Next close", caption: "one bar after the signal (next close)" },
];

const SOURCE_URL = "https://github.com/ROROK20/AlphaForge";

const cents = (n: number) =>
  `$${n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

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
    n == null ? "—" : n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const fmtPct = (n: number | null | undefined) => (n == null ? "—" : `${(n * 100).toFixed(2)}%`);
  // A difference of two returns: percentage POINTS. The ratio of final equities
  // minus 1 reads as a return gap and is not one (-45.60% for a gap of -89.42 points).
  const fmtPts = (n: number) => `${n > 0 ? "+" : ""}${(n * 100).toFixed(2)} pts`;
  const last = result?.equity[result.equity.length - 1];
  const copy = MODE_COPY[mode];
  const strategyInfo = STRATEGIES.find((s) => s.id === strategy);
  const fillInfo = FILLS.find((f) => f.id === fill);
  const gapLabel = last && result ? fmtPts((last.equity - last.benchmark) / result.params.initialCapital) : "—";

  // The headline, from the same response the figures below plot. Figures come
  // from `result.params`, the request that produced them, not the live form.
  const held = last && result ? plainReturn(benchmarkReturn(last, result.params.initialCapital)) : null;
  const made = m ? plainReturn(m.totalReturn) : null;
  // The maximum drawdown's dates and values, from the curve both figures plot.
  const drop = useMemo(() => (result ? maxDrawdownFromCurve(result.equity) : null), [result]);

  const stats = [
    { label: "Total Return", value: fmtPct(m?.totalReturn) },
    { label: "CAGR", value: fmtPct(m?.cagr) },
    { label: "Final Equity", value: last ? `$${last.equity.toLocaleString("en-US")}` : "—" },
    { label: "vs Buy-Hold", value: gapLabel, marked: !!last },
    { label: "Max DD", value: fmtPct(m?.maxDrawdown), marked: !!drop },
    { label: "Vol (ann)", value: fmtPct(m?.volatility) },
    { label: "Sharpe", value: fmt(m?.sharpe) },
    { label: "Sortino", value: fmt(m?.sortino) },
    { label: "Calmar", value: fmt(m?.calmar) },
    { label: "Win Rate", value: fmtPct(m?.winRate) },
    { label: "Profit Factor", value: fmt(m?.profitFactor) },
    { label: "# Trades", value: m == null ? "—" : m.nTrades.toString() },
  ];

  return (
    <main className="column pb-10">
      <header className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 border-b border-rule-strong py-3">
        <p className="flex flex-wrap items-baseline gap-x-2.5">
          <span className="text-lg font-bold tracking-tight">AlphaForge</span>
          <span className="text-sm text-ink-2">Backtest notebook</span>
        </p>
        <div className="flex flex-wrap items-center gap-x-5 gap-y-1">
          <ModeBadge mode={mode} />
          <a href={SOURCE_URL} className="inline-block py-1 text-sm text-ink-2 underline hover:text-ink">
            View Source
          </a>
        </div>
      </header>

      <div className="indent">
        {/* The point first, in words. Only what the response just said. */}
        <section className={`mt-5 ${copy.heroSpace}`} aria-live="polite" data-testid="result">
          {made && held && last && result ? (
            <div className="settled">
              {/* Each sentence stays whole where it can: a phone breaks between them. */}
              <h1 className="text-[1.6875rem] leading-[1.18] font-semibold tracking-tight md:text-[2.75rem]">
                <span className="inline-block">
                  {made.verb === "made" ? "Made" : made.verb === "lost" ? "Lost" : "Returned"}{" "}
                  <span className="text-plot">{made.pct}</span>.
                </span>{" "}
                <span className="inline-block">
                  Buy and hold {held.verb} <span>{held.pct}</span>.
                </span>
              </h1>
              <p className="mt-2.5 max-w-[78ch] text-base leading-snug md:text-[1.0625rem]" data-testid="result-detail">
                {STRATEGIES.find((s) => s.id === result.params.strategy)?.name} on the synthetic {result.params.symbol}{" "}
                feed, ${result.params.initialCapital.toLocaleString("en-US")} from{" "}
                <span className="whitespace-nowrap">{result.params.startDate}</span> to{" "}
                <span className="whitespace-nowrap">{result.params.endDate}</span>:{" "}
                <span className="pen-mark">{gapPhrase(last, result.params.initialCapital)}</span>. After{" "}
                {+(result.params.commission * 100).toFixed(4)}% commission and{" "}
                {(result.params.slippage * 100).toFixed(2)}% slippage per side, with fills{" "}
                {FILLS.find((f) => f.id === result.params.fill)?.caption}.{" "}
                <span className="inline-block">Synthetic prices, not market data.</span>
              </p>
            </div>
          ) : (
            <>
              <h1 className="text-[1.6875rem] leading-[1.18] font-semibold tracking-tight md:text-[2.75rem]">
                {copy.noResult}
              </h1>
              {copy.noResultDetail && (
                <p className="mt-2.5 max-w-[78ch] text-base leading-snug md:text-[1.0625rem]">{copy.noResultDetail}</p>
              )}
            </>
          )}
        </section>

        {result?.note && (
          <p className="notice" data-testid="strategy-note">
            <strong className="font-semibold text-pen">Note.</strong> {result.note}
          </p>
        )}

        {result?.ruined && (
          <p className="notice" data-testid="ruin-notice">
            <strong className="font-semibold text-pen">Ruined on {result.ruinDate}:</strong> equity reached $0, the
            position was liquidated at that bar&apos;s close and nothing traded afterwards. Metrics cover the path
            through that bar.
          </p>
        )}

        {mode === "rejected" && (
          <p className="notice" data-testid="rejected-notice">
            <strong className="font-semibold text-pen">The engine rejected this request:</strong> {rejection}
          </p>
        )}

        {(mode === "recorded" || mode === "offline") && (
          <p className="notice" data-testid="offline-notice">
            {mode === "recorded" ? (
              <>
                {NO_ENGINE
                  ? "This hosted demo has no engine behind it. It shows a "
                  : "The engine API is unreachable. This page shows a "}
                <strong className="font-semibold">recorded snapshot</strong> of the real engine&apos;s output for
                exactly this configuration, recorded <span className="whitespace-nowrap">{result?.recordedOn}</span> by{" "}
                <code>scripts/record_engine_snapshot.py</code> on the API&apos;s synthetic {symbol} feed. Change any
                setting and there is no recording —{" "}
                {NO_ENGINE ? (
                  <>
                    to compute other runs, clone the repo and run <code>docker compose up --build</code>.
                  </>
                ) : (
                  <>
                    start the API (<code>make api</code>) for live runs.
                  </>
                )}
              </>
            ) : NO_ENGINE ? (
              <>
                This hosted demo has no engine behind it, only its recordings of the default settings, one per
                strategy, so this configuration has no result — nothing is simulated in the browser. To compute it,
                clone the repo and run <code>docker compose up --build</code>.
              </>
            ) : (
              <>
                The engine API is unreachable and this configuration has no recorded snapshot, so there are no
                results — nothing is simulated in the browser. Start the API (<code>make api</code>) to run it.
              </>
            )}
          </p>
        )}

        <Entry
          label="Fig. 1"
          title="Equity, strategy against buy and hold"
          caption={
            <p>
              Blue: the strategy, net of its costs. Thin line: buy and hold on the same prices from the same capital,
              charged no costs. Red:{" "}
              {drop && (
                <>
                  the largest fall from a peak,{" "}
                  <span className="num">{cents(drop.peak.equity)}</span> on{" "}
                  <span className="num">{drop.peak.date}</span> to{" "}
                  <span className="num">{cents(drop.trough.equity)}</span> on{" "}
                  <span className="num">{drop.trough.date}</span>, and{" "}
                </>
              )}
              the gap at the last bar.
            </p>
          }
        >
          <EquityChart data={result?.equity ?? []} mark={drop} gapLabel={gapLabel} emptyMessage={copy.figureEmpty} />
        </Entry>

        <Entry label="Setup" title="Backtest configuration">
          {/* The two date columns are wider than the rest: Firefox spaces its date
              text ("01 / 01 / 2020") and ran the year into its calendar icon. */}
          <form
            className="grid grid-cols-2 gap-x-3 gap-y-2.5 md:grid-cols-[2fr_1fr_1fr_1.2fr_1.2fr_1fr]"
            onSubmit={(e) => e.preventDefault()}
          >
            <div className="col-span-2 md:col-span-1">
              <label htmlFor="strategy" className="field-label">
                Strategy
              </label>
              <select
                id="strategy"
                value={strategy}
                onChange={(e) => setStrategy(e.target.value as Strategy)}
                className="field"
              >
                {STRATEGIES.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="symbol" className="field-label">
                Symbol
              </label>
              <select id="symbol" value={symbol} onChange={(e) => setSymbol(e.target.value)} className="field">
                {SYMBOLS.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="fill" className="field-label">
                Fill
              </label>
              <select id="fill" value={fill} onChange={(e) => setFill(e.target.value as Fill)} className="field">
                {FILLS.map((f) => (
                  <option key={f.id} value={f.id}>
                    {f.name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="start" className="field-label">
                Start
              </label>
              <input
                id="start"
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="field font-mono"
              />
            </div>
            <div>
              <label htmlFor="end" className="field-label">
                End
              </label>
              <input
                id="end"
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="field font-mono"
              />
            </div>
            <div>
              <label htmlFor="capital" className="field-label">
                Capital ($)
              </label>
              <input
                id="capital"
                type="number"
                value={capital}
                onChange={(e) => setCapital(Number(e.target.value))}
                className="field font-mono"
              />
            </div>
          </form>
          <ul className="caption mt-3 space-y-0.5">
            <li>
              Strategy: {strategyInfo?.description}
              {result?.executed && (
                <>
                  . Engine ran <code>{result.executed}</code>
                </>
              )}
            </li>
            <li>
              Costs per side: {commission}% commission + {(SLIPPAGE * 100).toFixed(2)}% slippage
            </li>
            <li>Fills {fillInfo?.caption}</li>
            <li>Prices: synthetic (seeded per symbol), not market data</li>
          </ul>
        </Entry>

        <Entry
          label="Fig. 2"
          title="Drawdown"
          caption={<p>Distance from rolling peak: how far equity sits below the highest value it has reached so far.</p>}
        >
          <DrawdownChart data={result?.equity ?? []} mark={drop} emptyMessage={copy.figureEmpty} />
        </Entry>

        <Entry
          label="Table 1"
          title="Statistics"
          caption={
            <p>
              Underlined in red: figures marked on Fig. 1. vs Buy-Hold is the gap between the two total returns, in
              percentage points. Each figure is rounded on its own, so a difference taken from rounded figures can be
              off by 0.01. A dash means undefined (a zero denominator) or no result yet.
            </p>
          }
        >
          <StatsTable stats={stats} />
        </Entry>

        <TradesTable
          label="Table 2"
          trades={result ? result.trades : null}
          equityPnl={last && result ? last.equity - result.params.initialCapital : undefined}
          emptyMessage={copy.tradesEmpty}
        />

        <Entry label="Limits" title="What the model leaves out">
          <p className="caption">
            Shorts pay no borrow cost, cash earns 0%, negative cash is financed free, and there are no margin calls and
            no market impact. This is research software: not financial advice, and not for trading real money.
          </p>
        </Entry>

        <footer className="mt-10 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 border-t border-rule-strong pt-3 text-[0.8125rem] leading-6 text-ink-2">
          <p>
            AlphaForge · Open-source quant trading framework ·{" "}
            <a href={SOURCE_URL} className="inline-block underline hover:text-ink" target="_blank" rel="noreferrer">
              GitHub
            </a>{" "}
            ·{" "}
            <a
              href={`${SOURCE_URL}/blob/main/LICENSE`}
              className="inline-block underline hover:text-ink"
              target="_blank"
              rel="noreferrer"
            >
              MIT License
            </a>{" "}
            · {/* From the same total map as the badge (MODE_COPY). */}
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
      className={`flex items-center gap-1.5 text-sm font-medium ${copy.badgeClass}`}
      title={modeTitle(mode, NO_ENGINE)}
      data-testid="mode-badge"
    >
      <span aria-hidden="true" className={`inline-block h-2.5 w-2.5 border-[1.5px] ${copy.markClass}`} />
      {copy.badge}
    </span>
  );
}
