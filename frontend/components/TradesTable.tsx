"use client";

import type { Trade } from "@/lib/types";
import clsx from "clsx";

interface Props {
  trades: Trade[];
  /** When given, the header shows the trade log's net P&L total. */
  initialCapital?: number;
  emptyMessage?: string;
}

const money = (n: number) =>
  `${n > 0 ? "+" : n < 0 ? "-" : ""}$${Math.abs(n).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;

export default function TradesTable({ trades, initialCapital, emptyMessage }: Props) {
  // Net P&L (after commission and slippage). The engine guarantees the log's
  // total equals final equity minus initial capital (the reconciliation invariant),
  // so the header total can be checked against the Final Equity card, to within
  // the rounding of the rows it sums: each row arrives rounded to the cent.
  const netTotal = trades.reduce((s, t) => s + t.pnlNetAbs, 0);
  return (
    <div className="bg-panel border border-border rounded-lg overflow-hidden">
      <div className="px-4 py-3 border-b border-border flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300">Trade Log</h3>
        <span className="text-xs text-gray-500">
          {trades.length} trades
          {initialCapital !== undefined && (
            <>
              {" "}
              ·{" "}
              <span title="The sum of the rows below, each rounded to the cent, so it can differ from Final Equity minus capital by a few cents.">
                net P&L <span data-testid="trade-net-total">{money(netTotal)}</span>
              </span>
            </>
          )}
        </span>
      </div>
      <div className="overflow-auto max-h-96">
        <table className="w-full text-xs">
          <thead className="bg-bg sticky top-0">
            <tr className="text-gray-500 uppercase tracking-wider">
              <th className="px-4 py-2 text-left">#</th>
              <th
                className="px-4 py-2 text-left"
                title="The date of the close this trade was filled at. Entry $ is that close plus slippage against the trade."
              >
                Entry
              </th>
              <th
                className="px-4 py-2 text-left"
                title="The date of the close this trade was filled at. A position still open on the last bar, or ruined, is marked at that bar's close with no slippage."
              >
                Exit
              </th>
              <th className="px-4 py-2 text-left">Side</th>
              <th className="px-4 py-2 text-right">Entry $</th>
              <th className="px-4 py-2 text-right">Exit $</th>
              <th
                className="px-4 py-2 text-right"
                title="The trade's return between its entry and exit fills, signed by side: positive means the price moved in the trade's favour (before commission)."
              >
                Return %
              </th>
              <th className="px-4 py-2 text-right">Net P&L $</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((t) => (
              <tr key={t.id} className="border-t border-border hover:bg-bg/50">
                <td className="px-4 py-2 text-gray-500">{t.id}</td>
                <td className="px-4 py-2 tabular-nums">{t.entryDate}</td>
                <td className="px-4 py-2 tabular-nums">{t.exitDate}</td>
                <td className="px-4 py-2">
                  <span
                    className={clsx(
                      "px-1.5 py-0.5 rounded text-[10px] uppercase",
                      t.direction === "long"
                        ? "bg-accent-green/20 text-accent-green"
                        : "bg-accent-red/20 text-accent-red",
                    )}
                  >
                    {t.direction}
                  </span>
                </td>
                <td className="px-4 py-2 text-right tabular-nums">${t.entryPrice}</td>
                <td className="px-4 py-2 text-right tabular-nums">${t.exitPrice}</td>
                <td
                  className={clsx(
                    "px-4 py-2 text-right tabular-nums",
                    t.pnlPct > 0 ? "text-accent-green" : "text-accent-red",
                  )}
                >
                  {t.pnlPct > 0 ? "+" : ""}
                  {(t.pnlPct * 100).toFixed(2)}%
                </td>
                <td
                  className={clsx(
                    "px-4 py-2 text-right tabular-nums",
                    t.pnlNetAbs > 0 ? "text-accent-green" : "text-accent-red",
                  )}
                >
                  {money(t.pnlNetAbs)}
                </td>
              </tr>
            ))}
            {trades.length === 0 && (
              <tr>
                <td colSpan={8} className="px-4 py-8 text-center text-gray-500">
                  {emptyMessage ?? "No trades."}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
