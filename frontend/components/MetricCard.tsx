"use client";

import clsx from "clsx";

interface Props {
  label: string;
  value: string;
  hint?: string;
  trend?: "up" | "down" | "neutral";
}

export default function MetricCard({ label, value, hint, trend = "neutral" }: Props) {
  return (
    <div className="metric-card">
      <div className="text-xs uppercase tracking-wider text-gray-500 mb-2">{label}</div>
      <div
        className={clsx(
          "text-2xl font-bold tabular-nums",
          trend === "up" && "text-accent-green",
          trend === "down" && "text-accent-red",
          trend === "neutral" && "text-white",
        )}
      >
        {value}
      </div>
      {hint && <div className="text-xs text-gray-500 mt-1">{hint}</div>}
    </div>
  );
}
