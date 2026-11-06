/**
 * What the page says about a result, derived in the browser from the fields the
 * engine returned. Nothing here produces a figure the response does not imply:
 * every value is a rounding, a difference or a lookup over the equity curve.
 */

import type { EquityPoint, Trade } from "@/lib/types";

/** A signed fraction as the words a person would use, rounded as it is displayed. */
export function plainReturn(r: number): { verb: "made" | "lost" | "returned"; pct: string } {
  const pct = `${Math.abs(r * 100).toFixed(2)}%`;
  if (pct === "0.00%") return { verb: "returned", pct };
  return { verb: r > 0 ? "made" : "lost", pct };
}

/** Total return of the buy-and-hold curve: last benchmark value over starting capital. */
export function benchmarkReturn(last: EquityPoint, capital: number): number {
  return last.benchmark / capital - 1;
}

/** The gap between the two total returns, in percentage points, as a phrase. */
export function gapPhrase(last: EquityPoint, capital: number): string {
  const gap = (last.equity - last.benchmark) / capital;
  const points = Math.abs(gap * 100).toFixed(2);
  if (points === "0.00") return "level with buy and hold";
  return `${points} percentage points ${gap > 0 ? "ahead of" : "behind"} buy and hold`;
}

export interface DrawdownMark {
  /** Negative fraction: trough equity over peak equity, minus 1. */
  depth: number;
  peak: EquityPoint;
  trough: EquityPoint;
}

/**
 * The largest fall from a running peak, read from the equity column itself.
 * The curve's `drawdown` column is rounded to four places, which can tie and
 * name the wrong date; the equity column cannot. Null when the curve never falls
 * far enough to show (a flat, no-trade run has nothing to mark).
 */
export function maxDrawdownFromCurve(curve: EquityPoint[]): DrawdownMark | null {
  if (curve.length === 0) return null;
  let peak = curve[0];
  let mark: DrawdownMark | null = null;
  for (const point of curve) {
    if (point.equity > peak.equity) peak = point;
    if (peak.equity <= 0) continue;
    const depth = point.equity / peak.equity - 1;
    if (depth < (mark?.depth ?? 0)) mark = { depth, peak, trough: point };
  }
  return mark && (mark.depth * 100).toFixed(2) !== "-0.00" ? mark : null;
}

/** The losing trade with the largest net loss, or null if no trade lost money. */
export function largestLoss(trades: Trade[]): Trade | null {
  let worst: Trade | null = null;
  for (const t of trades) {
    if (t.pnlNetAbs < 0 && (worst === null || t.pnlNetAbs < worst.pnlNetAbs)) worst = t;
  }
  return worst;
}

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

/**
 * Where to put a chart note: the candidate that sits on the fewest curve points.
 * A candidate that falls off the plot is slid back inside (a narrow phone plot
 * may have no room either side of a mark) and counts as a little worse than one
 * that fits as asked. Candidates come in order of preference, so a tie keeps it.
 */
export function placeNote(candidates: Box[], curves: { x: number; y: number }[], plot: Box): Box {
  const slide = (v: number, size: number, start: number, length: number) =>
    Math.max(start, Math.min(v, start + length - size));
  const covered = (b: Box) =>
    curves.reduce((n, p) => n + (p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h ? 1 : 0), 0);
  let best = candidates[0];
  let bestScore = Infinity;
  for (const candidate of candidates) {
    const box = {
      ...candidate,
      x: slide(candidate.x, candidate.w, plot.x, plot.w),
      y: slide(candidate.y, candidate.h, plot.y, plot.h),
    };
    const moved = box.x !== candidate.x || box.y !== candidate.y;
    const score = covered(box) + (moved ? 8 : 0);
    if (score < bestScore) {
      best = box;
      bestScore = score;
    }
  }
  return best;
}

/**
 * Vertical centres for a stack of right-margin labels, kept in order and at
 * least `gap` apart. A stack that has to spread spreads around its own middle,
 * so the labels stay as close as they can to what they name; [min, max] holds.
 */
export function spreadLabels(wanted: number[], gap: number, min: number, max: number): number[] {
  const y = [...wanted];
  for (let i = 1; i < y.length; i++) y[i] = Math.max(y[i], y[i - 1] + gap);
  const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;
  const shift = mean(wanted) - mean(y);
  const shifted = y.map((v) => v + shift);
  const under = min - shifted[0];
  const over = shifted[shifted.length - 1] - max;
  const nudge = under > 0 ? under : over > 0 ? -over : 0;
  return shifted.map((v) => v + nudge);
}
