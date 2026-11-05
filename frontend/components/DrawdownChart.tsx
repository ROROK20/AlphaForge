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

export default function DrawdownChart({ data }: { data: EquityPoint[] }) {
  return (
    <div className="bg-panel border border-border rounded-lg p-4">
      <h3 className="text-sm font-semibold text-gray-300 mb-1">Drawdown</h3>
      <p className="text-xs text-gray-500 mb-3">Distance from rolling peak</p>
      <ResponsiveContainer width="100%" height={180}>
        <AreaChart data={data}>
          <defs>
            <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.5} />
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
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
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          />
          <Tooltip
            contentStyle={{
              background: "#0f1525",
              border: "1px solid #1c2540",
              borderRadius: "0.375rem",
              fontSize: "0.75rem",
            }}
            formatter={(value: number) => `${(value * 100).toFixed(2)}%`}
          />
          <Area
            type="monotone"
            dataKey="drawdown"
            stroke="#ef4444"
            fill="url(#ddGrad)"
            strokeWidth={2}
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
