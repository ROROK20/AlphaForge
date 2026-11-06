/**
 * Shapes shared by the dashboard and the engine API client.
 *
 * Every BacktestResult the dashboard renders is real engine output: either a
 * live `POST /backtest` response, or a recorded snapshot of one (see
 * `loadRecordedSnapshot` in lib/api.ts). Nothing is simulated in the browser.
 */

export type Strategy = "sma_crossover" | "rsi_meanrev" | "macd" | "ml_momentum";

/** "close": fill at the signal bar's close (market-on-close); "next_close": a bar later. */
export type Fill = "close" | "next_close";

export interface BacktestParams {
  symbol: string;
  strategy: Strategy;
  startDate: string;
  endDate: string;
  initialCapital: number;
  /** Per-side fraction of traded notional (0.001 = 0.1%). */
  commission: number;
  /** Per-side slippage, fraction of price (0.0005 = 0.05%). */
  slippage: number;
  fill: Fill;
}

export interface EquityPoint {
  date: string;
  equity: number;
  benchmark: number;
  drawdown: number;
}

export interface Trade {
  id: number;
  /** The date of the bar whose close is entryPrice (slippage included). */
  entryDate: string;
  /** The date of the bar whose close is exitPrice. Still open on the last bar, or ruined: that bar's bare close. */
  exitDate: string;
  direction: "long" | "short";
  entryPrice: number;
  exitPrice: number;
  /** Price return between the entry and exit fills, signed by direction. */
  pnlPct: number;
  /** Dollar P&L net of slippage, gross of commission. */
  pnlAbs: number;
  commissionAbs: number;
  /** Net dollar P&L. Summed over the log it equals final equity - capital. */
  pnlNetAbs: number;
}

/** null = undefined (zero denominator), as the engine API reports it. */
export interface Metrics {
  totalReturn: number;
  cagr: number;
  sharpe: number | null;
  sortino: number | null;
  calmar: number | null;
  maxDrawdown: number;
  volatility: number;
  winRate: number | null;
  profitFactor: number | null;
  nTrades: number;
}

export interface BacktestResult {
  params: BacktestParams;
  equity: EquityPoint[];
  trades: Trade[];
  metrics: Metrics;
  /** What the engine actually ran, with its parameters, e.g. "sma_crossover(20,50)". */
  executed?: string;
  /** Engine disclosure, e.g. when ml_momentum runs as an SMA proxy. */
  note?: string;
  /** Equity reached $0 on ruinDate: liquidated, no trading afterwards. */
  ruined?: boolean;
  ruinDate?: string | null;
  /** Set only for a recorded snapshot: the date it was recorded (YYYY-MM-DD). */
  recordedOn?: string;
}
