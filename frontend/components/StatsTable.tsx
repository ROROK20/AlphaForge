export interface Stat {
  label: string;
  value: string;
  /** Underlined in red pen: this figure is also marked on a chart. */
  marked?: boolean;
}

/**
 * The run's statistics, ruled like a table in a paper. ARIA table roles rather
 * than <table>: its rows reflow from three columns to one, which a table's cannot.
 */
export default function StatsTable({ stats }: { stats: Stat[] }) {
  return (
    <div role="table" aria-label="Run statistics" className="stats" data-testid="metrics">
      {stats.map((s) => (
        <div key={s.label} role="row" className="stat-row">
          <div role="rowheader" className="text-ink-2">
            {s.label}
          </div>
          <div role="cell" className={s.marked ? "stat-value pen-mark" : "stat-value"}>
            {s.value}
          </div>
        </div>
      ))}
    </div>
  );
}
