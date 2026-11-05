"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EquityPoint } from "@/lib/types";

interface Props {
  data: EquityPoint[];
}

export default function EquityChart({ data }: Props) {
  return (
    <div className="bg-panel border border-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold text-gray-300">Equity Curve</h3>
          <p className="text-xs text-gray-500">Strategy vs buy-and-hold benchmark</p>
        </div>
        <div className="flex gap-4 text-xs">
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-accent-blue" />
            Strategy
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-gray-500" />
            Benchmark
          </span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data}>
          <defs>
            <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1c2540" />
          {/* Ticks are dropped where they would collide (preserveStartEnd +
              minTickGap), so labels never overlap at any panel width; they
              read YYYY-MM, and the tooltip still shows the full date. A fixed
              interval of n/8 overlapped in this narrow panel. */}
          <XAxis
            dataKey="date"
            stroke="#6b7280"
            tick={{ fontSize: 10 }}
            tickFormatter={(d: string) => d.slice(0, 7)}
            interval="preserveStartEnd"
            minTickGap={16}
          />
          <YAxis
            stroke="#6b7280"
            tick={{ fontSize: 10 }}
            tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
          />
          <Tooltip
            contentStyle={{
              background: "#0f1525",
              border: "1px solid #1c2540",
              borderRadius: "0.375rem",
              fontSize: "0.75rem",
            }}
            formatter={(value: number) => `$${value.toLocaleString()}`}
          />
          <Area
            type="monotone"
            dataKey="benchmark"
            stroke="#6b7280"
            strokeDasharray="4 4"
            fill="transparent"
            strokeWidth={1.5}
            dot={false}
          />
          <Area
            type="monotone"
            dataKey="equity"
            stroke="#3b82f6"
            fill="url(#equityGrad)"
            strokeWidth={2}
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
