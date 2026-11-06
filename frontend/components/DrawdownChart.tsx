"use client";

import { Area, AreaChart, CartesianGrid, Customized, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { EquityPoint } from "@/lib/types";
import { placeNote, type DrawdownMark } from "@/lib/reading";
import {
  AXIS_TICK,
  dateLabel,
  dateTicks,
  drawdownAxis,
  EmptyPlot,
  HALO,
  PLOT_MARGIN,
  type PlotState,
  Quadrille,
  TOOLTIP_STYLE,
  Y_AXIS_WIDTH,
} from "@/components/PlotParts";

const pct = (n: number) => `${(n * 100).toFixed(2)}%`;

interface MarksProps extends PlotState {
  data: EquityPoint[];
  mark: DrawdownMark | null;
}

function DrawdownMarks({ data, mark, xAxisMap, yAxisMap, offset }: MarksProps) {
  const xAxis = xAxisMap && Object.values(xAxisMap)[0];
  const yAxis = yAxisMap && Object.values(yAxisMap)[0];
  if (!xAxis || !yAxis || !offset || data.length === 0) return null;
  const x = (d: string): number => xAxis.scale(d);
  const y = (v: number): number => yAxis.scale(v);
  const last = data[data.length - 1];
  const xEnd = x(last.date);
  const plot = { x: offset.left, y: offset.top, w: offset.width, h: offset.height };

  let note = null;
  if (mark) {
    const xT = x(mark.trough.date);
    // Measured on the equity column, so this y is the depth itself rather than
    // the rounded drawdown column's value for that bar.
    const yT = y(mark.depth);
    const lines = [`Max drawdown ${pct(mark.depth)}`, `on ${mark.trough.date}`];
    const w = Math.max(...lines.map((l) => l.length)) * 7.1 + 6;
    const h = 34;
    const curve = data.map((p) => ({ x: x(p.date), y: y(p.drawdown) }));
    const box = placeNote(
      [
        { x: xT - w - 10, y: yT - h / 2 + 4, w, h },
        { x: xT + 10, y: yT - h / 2 + 4, w, h },
        { x: xT - w - 10, y: plot.y + plot.h - h - 2, w, h },
        { x: xT + 10, y: plot.y + plot.h - h - 2, w, h },
      ],
      curve,
      plot,
    );
    note = (
      <g>
        <circle cx={xT} cy={yT} r={4} fill="var(--paper)" stroke="var(--pen)" strokeWidth={1.5} aria-hidden="true" />
        <text x={box.x + 3} y={box.y + 13} fontSize={13} fill="var(--pen)" style={HALO}>
          <tspan fontWeight={600}>{lines[0]}</tspan>
          <tspan x={box.x + 3} dy={16} fontSize={12}>
            {lines[1]}
          </tspan>
        </text>
      </g>
    );
  }

  // The last bar's distance from its peak, in the gutter. Still inside the plot
  // when the curve ends at 0%, where the label would otherwise leave the top.
  const yLast = y(last.drawdown);
  const labelY = Math.max(yLast - 3, plot.y + 12);
  return (
    <g>
      {note}
      <g aria-hidden="true" fontSize={12} fill="var(--ink)">
        <circle cx={xEnd} cy={yLast} r={2.5} fill="var(--plot)" />
        <text x={xEnd + 15} y={labelY} fontWeight={600}>
          Last bar
        </text>
        <text x={xEnd + 15} y={labelY + 15} fontFamily="var(--font-spline-mono)">
          {pct(last.drawdown)}
        </text>
      </g>
    </g>
  );
}

interface Props {
  data: EquityPoint[];
  /** The largest fall from a peak, computed from the equity column of this same curve. */
  mark: DrawdownMark | null;
  emptyMessage: string;
}

export default function DrawdownChart({ data, mark, emptyMessage }: Props) {
  if (data.length === 0) return <EmptyPlot className="plot-drawdown" message={emptyMessage} />;
  const last = data[data.length - 1];
  const summary =
    `Drawdown from the rolling peak, ${data[0].date} to ${last.date}.` +
    (mark ? ` Deepest point ${pct(mark.depth)} on ${mark.trough.date}.` : " The curve never fell below its peak.") +
    ` At the last bar: ${pct(last.drawdown)}.`;
  const axis = drawdownAxis(mark?.depth ?? 0);
  return (
    <div role="img" aria-label={summary} className="plot-drawdown">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={PLOT_MARGIN}>
          <Customized component={<Quadrille id="quadrille-drawdown" />} />
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
            tickFormatter={(v: number) => `${(v * 100).toFixed(axis.digits)}%`}
          />
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            formatter={(value: number) => `${(value * 100).toFixed(2)}%`}
            isAnimationActive={false}
          />
          <Area
            type="linear"
            dataKey="drawdown"
            name="Drawdown"
            stroke="var(--plot)"
            strokeWidth={1.5}
            fill="var(--plot-wash)"
            fillOpacity={1}
            dot={false}
            isAnimationActive={false}
          />
          <Customized component={<DrawdownMarks data={data} mark={mark} />} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
