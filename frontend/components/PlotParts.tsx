"use client";

/**
 * What both figures share, so their plots line up on the page: the same left
 * and right gutters put a date in Fig. 2 directly under the same date in Fig. 1.
 */

export const PLOT_MARGIN = { top: 14, right: 112, bottom: 2, left: 0 };
export const Y_AXIS_WIDTH = 52;
export const AXIS_TICK = { fontSize: 12, fontFamily: "var(--font-spline-mono), ui-monospace, monospace", fill: "var(--ink-3)" };
export const TOOLTIP_STYLE = {
  background: "var(--field)",
  border: "1px solid var(--rule-strong)",
  borderRadius: 2,
  fontSize: 12,
  fontFamily: "var(--font-spline-mono), ui-monospace, monospace",
  color: "var(--ink)",
};
export const SQUARE = 16;

/**
 * Axis dates: the first bar of each year, half, quarter or month, by the length
 * of the window, so ticks fall on calendar boundaries rather than at every nth
 * trading day. Very short windows leave it to Recharts.
 */
export function dateTicks(data: { date: string }[]): string[] | undefined {
  if (data.length < 2) return undefined;
  const month = (d: string) => Number(d.slice(0, 4)) * 12 + Number(d.slice(5, 7)) - 1;
  const span = month(data[data.length - 1].date) - month(data[0].date);
  const every = span >= 60 ? 12 : span >= 24 ? 6 : span >= 9 ? 3 : span >= 3 ? 1 : 0;
  if (every === 0) return undefined;
  const ticks: string[] = [];
  let previous = -1;
  for (const { date } of data) {
    const m = month(date);
    if (m % every === 0 && m !== previous) {
      ticks.push(date);
      previous = m;
    }
  }
  return ticks.length >= 2 ? ticks : undefined;
}

/** Axis date format: YYYY-MM, or the whole date when a short window would repeat a month. */
export function dateLabel(data: { date: string }[]): (d: string) => string {
  const short = data.length < 2 || data[0].date.slice(0, 7) === data[data.length - 1].date.slice(0, 7);
  const monthsApart = (a: string, b: string) =>
    (Number(b.slice(0, 4)) - Number(a.slice(0, 4))) * 12 + Number(b.slice(5, 7)) - Number(a.slice(5, 7));
  return short || monthsApart(data[0].date, data[data.length - 1].date) < 3 ? (d) => d : (d) => d.slice(0, 7);
}

/** Round ticks from $0 for an equity axis, so the gridlines can be read as round sums. */
export function moneyAxis(top: number): { domain: [number, number]; ticks: number[] } {
  const scale = 10 ** Math.floor(Math.log10(Math.max(top, 1)));
  const step = [0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5].map((f) => f * scale).find((s) => top / s <= 5) ?? scale * 5;
  const count = Math.max(Math.ceil(top / step), 1);
  return { domain: [0, count * step], ticks: Array.from({ length: count + 1 }, (_, i) => i * step) };
}

/** Round ticks for a drawdown axis, with room under the deepest point for its note. */
export function drawdownAxis(deepest: number): { domain: [number, number]; ticks: number[]; digits: number } {
  const room = Math.max(Math.abs(deepest), 0.001) * 1.1;
  const step = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25].find((s) => room / s <= 5) ?? 0.25;
  // An account cannot fall further than all of it.
  const count = Math.min(Math.ceil(room / step), Math.ceil(1 / step));
  const ticks = Array.from({ length: count + 1 }, (_, i) => (i === 0 ? 0 : -i * step));
  return { domain: [-count * step, 0], ticks, digits: step < 0.01 ? 1 : 0 };
}

/** Text with a paper-coloured halo, so a note stays legible where it crosses a curve. */
export const HALO = { paintOrder: "stroke" as const, stroke: "var(--paper)", strokeWidth: 4, strokeLinejoin: "round" as const };

interface PlotOffset {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** The part of Recharts' chart state its Customized children read: pixel scales and the plot rectangle. */
export interface PlotState {
  xAxisMap?: Record<string, { scale: (date: string) => number }>;
  yAxisMap?: Record<string, { scale: (value: number) => number }>;
  offset?: PlotOffset;
}

/** Graph paper under the plot only. Rendered by Recharts' Customized, first, so curves sit on top. */
export function Quadrille({ id, offset }: { id: string; offset?: PlotOffset }) {
  if (!offset) return null;
  return (
    <g aria-hidden="true">
      <defs>
        <pattern id={id} x={offset.left} y={offset.top} width={SQUARE} height={SQUARE} patternUnits="userSpaceOnUse">
          <path d={`M ${SQUARE} 0 H 0 V ${SQUARE}`} fill="none" stroke="var(--quad)" strokeWidth={1} />
        </pattern>
      </defs>
      <rect x={offset.left} y={offset.top} width={offset.width} height={offset.height} fill={`url(#${id})`} />
    </g>
  );
}

/** A figure with nothing to plot yet: the same blank paper, never a placeholder curve. */
export function EmptyPlot({ className, message }: { className: string; message: string }) {
  return (
    <div className={`graph-paper flex items-center justify-center px-4 text-center ${className}`}>
      <p className="caption bg-paper px-2 py-0.5">{message}</p>
    </div>
  );
}
