"use client";

import { Area, AreaChart, CartesianGrid, Customized, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { EquityPoint } from "@/lib/types";
import { placeNote, spreadLabels, type DrawdownMark } from "@/lib/reading";
import {
  AXIS_TICK,
  dateLabel,
  dateTicks,
  EmptyPlot,
  HALO,
  moneyAxis,
  PLOT_MARGIN,
  type PlotState,
  Quadrille,
  TOOLTIP_STYLE,
  Y_AXIS_WIDTH,
} from "@/components/PlotParts";

interface Props {
  data: EquityPoint[];
  /** The largest fall from a peak, computed from this same curve. */
  mark: DrawdownMark | null;
  /** The gap at the last bar, formatted by the page (the same string as Table 1). */
  gapLabel: string;
  emptyMessage: string;
}

const pct = (n: number) => `${(n * 100).toFixed(2)}%`;
const dollars = (n: number) => `$${Math.round(n).toLocaleString("en-US")}`;
const axisDollars = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  notation: "compact",
  maximumFractionDigits: 1,
});

interface MarksProps extends PlotState {
  data: EquityPoint[];
  mark: DrawdownMark | null;
  gapLabel: string;
}

/** The red pen: drawn in the plot's own pixel space, on the dates and values they describe. */
function EquityMarks({ data, mark, gapLabel, xAxisMap, yAxisMap, offset }: MarksProps) {
  const xAxis = xAxisMap && Object.values(xAxisMap)[0];
  const yAxis = yAxisMap && Object.values(yAxisMap)[0];
  if (!xAxis || !yAxis || !offset || data.length === 0) return null;
  const x = (d: string): number => xAxis.scale(d);
  const y = (v: number): number => yAxis.scale(v);

  const last = data[data.length - 1];
  const xEnd = x(last.date);
  const plot = { x: offset.left, y: offset.top, w: offset.width, h: offset.height };

  // Right gutter: who ended where, and the gap between them. The higher curve's
  // label goes on top; labels spread apart only as far as they need to.
  const strategyOnTop = last.equity >= last.benchmark;
  const yStrategy = y(last.equity);
  const yBenchmark = y(last.benchmark);
  const [yTop, yBottom] = strategyOnTop ? [yStrategy, yBenchmark] : [yBenchmark, yStrategy];
  const [labelTop, labelGap, labelBottom] = spreadLabels(
    [yTop, (yTop + yBottom) / 2, yBottom],
    36,
    offset.top + 8,
    // Not past the plot's foot: the axis' last date label sits just below it.
    offset.top + offset.height - 14,
  );
  const bracketX = xEnd + 8;
  const textX = bracketX + 7;
  const strategyEnd = { name: "Strategy", value: dollars(last.equity), fill: "var(--plot)" };
  const benchmarkEnd = { name: "Buy and hold", value: dollars(last.benchmark), fill: "var(--ink)" };
  const [topEnd, bottomEnd] = strategyOnTop ? [strategyEnd, benchmarkEnd] : [benchmarkEnd, strategyEnd];

  // The drawdown: flat from the peak, straight down to the trough, note beside it.
  const drop = mark && {
    xPeak: x(mark.peak.date),
    yPeak: y(mark.peak.equity),
    xTrough: x(mark.trough.date),
    yTrough: y(mark.trough.equity),
  };
  let note = null;
  if (mark && drop) {
    const lines = [`Max drawdown ${pct(mark.depth)}`, `trough ${mark.trough.date}`];
    const w = Math.max(...lines.map((l) => l.length)) * 7.1 + 6;
    const h = 34;
    const midY = (drop.yPeak + drop.yTrough) / 2 - h / 2;
    const below = drop.yTrough + 10;
    // Everything the note must not sit on: both curves and the drawdown's own lines.
    const curves = data.flatMap((p) => [
      { x: x(p.date), y: y(p.equity) },
      { x: x(p.date), y: y(p.benchmark) },
    ]);
    for (let px = drop.xPeak; px <= drop.xTrough; px += 4) curves.push({ x: px, y: drop.yPeak });
    for (let py = drop.yPeak; py <= drop.yTrough; py += 4) curves.push({ x: drop.xTrough, y: py });
    const box = placeNote(
      [
        { x: drop.xTrough - w - 8, y: below, w, h },
        { x: drop.xTrough + 8, y: below, w, h },
        { x: drop.xTrough - w - 8, y: midY, w, h },
        { x: drop.xTrough + 8, y: midY, w, h },
        { x: drop.xTrough - w - 8, y: drop.yPeak - h - 6, w, h },
        { x: drop.xTrough + 8, y: drop.yPeak - h - 6, w, h },
        { x: plot.x + 8, y: plot.y + plot.h - h - 6, w, h },
        { x: plot.x + plot.w - w - 8, y: plot.y + plot.h - h - 6, w, h },
      ],
      curves,
      plot,
    );
    note = (
      <text x={box.x + 3} y={box.y + 13} fontSize={13} fill="var(--pen)" style={HALO} data-testid="drawdown-note">
        <tspan fontWeight={600}>{lines[0]}</tspan>
        <tspan x={box.x + 3} dy={16} fontSize={12}>
          {lines[1]}
        </tspan>
      </text>
    );
  }

  return (
    <g>
      {drop && (
        <g aria-hidden="true" fill="none" stroke="var(--pen)" strokeWidth={1.25}>
          <path d={`M ${drop.xPeak} ${drop.yPeak} H ${drop.xTrough}`} strokeDasharray="2 3" />
          <path d={`M ${drop.xTrough} ${drop.yPeak} V ${drop.yTrough}`} />
          <path d={`M ${drop.xPeak} ${drop.yPeak - 4} V ${drop.yPeak + 4}`} />
          <circle cx={drop.xTrough} cy={drop.yTrough} r={4} fill="var(--paper)" strokeWidth={1.5} />
        </g>
      )}
      {note}

      {/* The gap at the end: a bracket between the two curves' last values. */}
      <g aria-hidden="true" fill="none" stroke="var(--pen)" strokeWidth={1.25}>
        <path d={`M ${xEnd + 3} ${yTop} H ${bracketX} V ${yBottom} H ${xEnd + 3}`} />
      </g>
      <g aria-hidden="true" fontSize={12}>
        <text x={textX} y={labelTop - 3} fill={topEnd.fill} fontWeight={600}>
          {topEnd.name}
        </text>
        <text x={textX} y={labelTop + 12} fill={topEnd.fill} fontFamily="var(--font-spline-mono)">
          {topEnd.value}
        </text>
        <text x={textX} y={labelGap - 3} fill="var(--pen)" fontWeight={600}>
          Gap at end
        </text>
        <text x={textX} y={labelGap + 12} fill="var(--pen)" fontFamily="var(--font-spline-mono)" data-testid="gap-note">
          {gapLabel}
        </text>
        <text x={textX} y={labelBottom - 3} fill={bottomEnd.fill} fontWeight={600}>
          {bottomEnd.name}
        </text>
        <text x={textX} y={labelBottom + 12} fill={bottomEnd.fill} fontFamily="var(--font-spline-mono)">
          {bottomEnd.value}
        </text>
      </g>
    </g>
  );
}

export default function EquityChart({ data, mark, gapLabel, emptyMessage }: Props) {
  if (data.length === 0) return <EmptyPlot className="plot-equity" message={emptyMessage} />;
  const last = data[data.length - 1];
  const summary =
    `Equity curve from ${data[0].date} to ${last.date}. The strategy ended at ${dollars(last.equity)}, ` +
    `buy and hold at ${dollars(last.benchmark)}, a gap of ${gapLabel}.` +
    (mark ? ` Largest fall from a peak: ${pct(mark.depth)}, reaching its low on ${mark.trough.date}.` : "");
  const axis = moneyAxis(data.reduce((top, p) => Math.max(top, p.equity, p.benchmark), 0));
  return (
    <div role="img" aria-label={summary} className="plot-equity">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={PLOT_MARGIN}>
          <Customized component={<Quadrille id="quadrille-equity" />} />
          <CartesianGrid stroke="var(--rule)" />
          {/* Ticks are dropped where they would collide (preserveStartEnd +
              minTickGap), so labels never overlap at any width; they read
              YYYY-MM, and the tooltip still shows the full date. */}
          <XAxis
            dataKey="date"
            stroke="var(--rule-strong)"
            tick={AXIS_TICK}
            ticks={dateTicks(data)}
            tickFormatter={dateLabel(data)}
            interval="preserveStartEnd"
            minTickGap={24}
          />
          <YAxis
            stroke="var(--rule-strong)"
            tick={AXIS_TICK}
            width={Y_AXIS_WIDTH}
            domain={axis.domain}
            ticks={axis.ticks}
            tickFormatter={(v: number) => axisDollars.format(v)}
          />
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            formatter={(value: number) => `$${value.toLocaleString("en-US")}`}
            isAnimationActive={false}
          />
          <Area
            type="linear"
            dataKey="benchmark"
            name="Buy and hold"
            stroke="var(--ink)"
            strokeWidth={1}
            fill="none"
            dot={false}
            isAnimationActive={false}
          />
          <Area
            type="linear"
            dataKey="equity"
            name="Strategy"
            stroke="var(--plot)"
            strokeWidth={1.75}
            fill="none"
            dot={false}
            isAnimationActive={false}
          />
          <Customized component={<EquityMarks data={data} mark={mark} gapLabel={gapLabel} />} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
