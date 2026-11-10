"use client";

import { useEffect, useRef, useState } from "react";
import type { Trade } from "@/lib/types";
import { largestLoss } from "@/lib/reading";
import Entry from "@/components/Entry";

interface Props {
  label: string;
  /** null = nobody has answered yet: no count and no total, not "0 trades". */
  trades: Trade[] | null;
  /** Final equity minus capital: what the log's net P&L column has to sum to. */
  equityPnl?: number;
  emptyMessage?: string;
}

const money = (n: number) =>
  `${n > 0 ? "+" : n < 0 ? "-" : ""}$${Math.abs(n).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;

/**
 * A price exactly as the engine sent it, kept to its digits, with the missing
 * trailing zero held as invisible space so the decimal points still line up.
 */
function Price({ value }: { value: number }) {
  const text = String(value);
  const decimals = text.includes(".") ? text.length - text.indexOf(".") - 1 : -1;
  const padding = decimals >= 2 ? "" : decimals === 1 ? "0" : ".00";
  return (
    <>
      ${text}
      {padding && <span className="invisible">{padding}</span>}
    </>
  );
}

export default function TradesTable({ label, trades, equityPnl, emptyMessage }: Props) {
  const rows = trades ?? [];
  // Net P&L (after commission and slippage). The engine guarantees the log's
  // total equals final equity minus initial capital (the reconciliation invariant),
  // so the total can be checked against Final Equity, to within the rounding of
  // the rows it sums: each row arrives rounded to the cent.
  const netTotal = rows.reduce((s, t) => s + t.pnlNetAbs, 0);
  const worst = largestLoss(rows);
  const difference = equityPnl === undefined ? null : Math.abs(netTotal - equityPnl);

  // The log scrolls sideways when its columns are wider than its box (on any
  // phone held upright). The hint says so only while that is measured to be
  // true: the box and the table are both watched, since rows can widen it.
  const scrollRef = useRef<HTMLDivElement>(null);
  const [scrolls, setScrolls] = useState(false);
  useEffect(() => {
    const box = scrollRef.current;
    if (!box) return;
    const measure = () => setScrolls(box.scrollWidth > box.clientWidth);
    const watch = new ResizeObserver(measure);
    watch.observe(box);
    if (box.firstElementChild) watch.observe(box.firstElementChild);
    measure();
    return () => watch.disconnect();
  }, []);

  return (
    <Entry
      label={label}
      title="Trade log"
      aside={
        trades && (
          <span className="caption font-mono">
            {trades.length} {trades.length === 1 ? "trade" : "trades"}
            {equityPnl !== undefined && (
              <>
                {" "}
                ·{" "}
                <span title="The sum of the rows below, each rounded to the cent, so it can differ from Final Equity minus capital by a few cents.">
                  net P&L <span data-testid="trade-net-total">{money(netTotal)}</span>
                </span>
              </>
            )}
          </span>
        )
      }
      caption={
        rows.length > 0 &&
        equityPnl !== undefined &&
        difference !== null && (
          <>
            <p data-testid="reconciliation">
              The column sums to <span className="num">{money(netTotal)}</span>. Final Equity minus capital is{" "}
              <span className="num">{money(equityPnl)}</span>.{" "}
              {difference.toFixed(2) === "0.00"
                ? "They agree to the cent."
                : `The difference, $${difference.toFixed(2)}, is each row's rounding to the cent.`}
            </p>
            {worst && (
              <p>Underlined: the largest net loss, trade {worst.id}.</p>
            )}
          </>
        )
      }
    >
      {scrolls && (
        <p data-scroll-hint className="caption mb-1.5">
          Scroll the log sideways for every column →
        </p>
      )}
      <div
        ref={scrollRef}
        className="table-scroll"
        role="region"
        aria-label="Trade log, scrolls sideways on a narrow screen"
        tabIndex={0}
      >
        <table className="trades">
          <thead>
            <tr>
              <th className="text-right">#</th>
              <th title="The date of the close this trade was filled at. Entry $ is that close plus slippage against the trade.">
                Entry
              </th>
              <th title="The date of the close this trade was filled at. A position still open on the last bar, or ruined, is marked at that bar's close with no slippage.">
                Exit
              </th>
              <th>Side</th>
              <th className="text-right">Entry $</th>
              <th className="text-right">Exit $</th>
              <th
                className="text-right"
                title="The trade's return between its entry and exit fills, signed by side: positive means the price moved in the trade's favour (before commission)."
              >
                Return %
              </th>
              <th className="text-right">Net P&L $</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((t) => (
              <tr key={t.id}>
                <td className="text-right text-ink-3">{t.id}</td>
                <td>{t.entryDate}</td>
                <td>{t.exitDate}</td>
                <td>{t.direction}</td>
                <td className="text-right">
                  <Price value={t.entryPrice} />
                </td>
                <td className="text-right">
                  <Price value={t.exitPrice} />
                </td>
                <td className="text-right">
                  {t.pnlPct > 0 ? "+" : ""}
                  {(t.pnlPct * 100).toFixed(2)}%
                </td>
                <td className="text-right">
                  <span className={worst?.id === t.id ? "pen-mark" : undefined}>{money(t.pnlNetAbs)}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {/* Outside the scrolling table, so a phone reads it without scrolling sideways. */}
      {rows.length === 0 && <p className="caption mx-auto py-6 text-center">{emptyMessage ?? "No trades."}</p>}
    </Entry>
  );
}
