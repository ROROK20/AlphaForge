/**
 * Typed client for the real AlphaForge engine API.
 *
 * Reads NEXT_PUBLIC_API_URL at BUILD time (see lib/urls.ts: only `next dev`
 * falls back to http://localhost:8003; a production build without it has no
 * API and makes no network call) and calls the FastAPI backend. When the backend is unreachable the dashboard does NOT
 * simulate anything: it can only show a recorded snapshot of the engine's own
 * output for the exact configuration that was recorded (`loadRecordedSnapshot`),
 * and otherwise shows no results.
 */

import type { BacktestParams, BacktestResult } from "@/lib/types";
import { resolveBaseUrl } from "@/lib/urls";

// Literal process.env.* expressions, so Next inlines them at build time and a
// production build folds the dev-only localhost fallback away entirely.
export const API_URL = resolveBaseUrl(
  process.env.NEXT_PUBLIC_API_URL,
  process.env.NODE_ENV === "development" ? "http://localhost:8003" : null,
);

/** This build has no engine API (production build without NEXT_PUBLIC_API_URL). */
export class ApiNotConfigured extends Error {
  constructor() {
    super("no engine API is configured for this build (NEXT_PUBLIC_API_URL unset)");
    this.name = "ApiNotConfigured";
  }
}

// Shape returned by POST /backtest (already camelCase from the backend).
interface ApiBacktestResponse {
  source: "engine";
  // "synthetic" = seeded GBM bars, not market data for `symbol`. This client
  // never sends its own bars, so it always receives "synthetic".
  dataSource: "synthetic" | "supplied";
  symbol: string;
  strategy: string;
  strategyExecuted: string;
  fill: BacktestParams["fill"];
  commission: number;
  slippage: number;
  note: string | null;
  ruined?: boolean;
  ruinDate?: string | null;
  equity: BacktestResult["equity"];
  trades: BacktestResult["trades"];
  metrics: BacktestResult["metrics"];
}

/** A file under public/engine-snapshot/, written by scripts/record_engine_snapshot.py. */
interface EngineSnapshot {
  kind: "alphaforge-engine-snapshot";
  recordedOn: string;
  recordedWith: string;
  request: BacktestParams;
  response: ApiBacktestResponse;
}

const DEFAULT_TIMEOUT_MS = 8000;

/**
 * The engine answered and refused the request (HTTP 4xx/5xx), as opposed to
 * being unreachable. Carries the API's own explanation (FastAPI `detail`).
 */
export class ApiRejected extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiRejected";
  }
}

/** FastAPI error bodies: `{"detail": "text"}` or `{"detail": [{"msg": ...}, ...]}`. */
function describe(body: unknown, status: number): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => String((d as { msg?: unknown }).msg ?? "").replace(/^Value error, /, ""))
      .filter(Boolean)
      .join("; ");
  }
  return `HTTP ${status}`;
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  // No API in this build: fail BEFORE any request (the caller falls back to
  // the recorded snapshot), rather than calling the visitor's own localhost.
  if (API_URL === null) throw new ApiNotConfigured();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);
  try {
    const res = await fetch(`${API_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
      cache: "no-store",
    });
    if (!res.ok) {
      const body = await res.json().catch(() => null);
      throw new ApiRejected(res.status, describe(body, res.status));
    }
    return (await res.json()) as T;
  } finally {
    clearTimeout(timeout);
  }
}

function toResult(params: BacktestParams, data: ApiBacktestResponse): BacktestResult {
  return {
    params,
    equity: data.equity,
    trades: data.trades,
    metrics: data.metrics,
    note: data.note ?? undefined,
    ruined: data.ruined ?? false,
    ruinDate: data.ruinDate ?? null,
  };
}

/**
 * Call the real engine. Throws ApiRejected when the engine answers with an
 * error, and any other error when it cannot be reached (network, timeout).
 */
export async function runRealBacktest(params: BacktestParams): Promise<BacktestResult> {
  const data = await postJSON<ApiBacktestResponse>("/backtest", {
    symbol: params.symbol,
    strategy: params.strategy,
    startDate: params.startDate,
    endDate: params.endDate,
    initialCapital: params.initialCapital,
    commission: params.commission,
    slippage: params.slippage,
    fill: params.fill,
  });
  return toResult(params, data);
}

const sameMoney = (a: number, b: number) => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b));

/**
 * The recorded engine output for exactly these params, or null.
 *
 * Snapshots are static files served with the frontend (one per strategy, for
 * the dashboard's default symbol, dates, capital and commission). Any other
 * configuration has no recording, and the caller must show no results.
 */
export async function loadRecordedSnapshot(params: BacktestParams): Promise<BacktestResult | null> {
  try {
    const res = await fetch(`/engine-snapshot/${params.strategy}.json`, { cache: "force-cache" });
    if (!res.ok) return null;
    const snap = (await res.json()) as EngineSnapshot;
    const r = snap.request;
    const matches =
      snap.kind === "alphaforge-engine-snapshot" &&
      r.symbol === params.symbol &&
      r.strategy === params.strategy &&
      r.startDate === params.startDate &&
      r.endDate === params.endDate &&
      sameMoney(r.initialCapital, params.initialCapital) &&
      sameMoney(r.commission, params.commission) &&
      sameMoney(r.slippage ?? 0.0005, params.slippage) &&
      (r.fill ?? "close") === params.fill;
    if (!matches) return null;
    return { ...toResult(params, snap.response), recordedOn: snap.recordedOn };
  } catch {
    return null;
  }
}
