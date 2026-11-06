import type { ReactNode } from "react";

interface Props {
  /** The notebook's running reference: "Fig. 1", "Table 2", "Setup". */
  label: string;
  title: ReactNode;
  /** Anything that belongs beside the title (a count, a total). */
  aside?: ReactNode;
  caption?: ReactNode;
  children: ReactNode;
}

/** One numbered entry in the reading column: reference in the margin, then content. */
export default function Entry({ label, title, aside, caption, children }: Props) {
  return (
    <section className="entry">
      <header className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-0.5 mb-3">
        <h2 className="text-base font-semibold text-ink">
          <span className="entry-label">{label}</span>
          {title}
        </h2>
        {aside}
      </header>
      {children}
      {caption && <div className="caption mt-3">{caption}</div>}
    </section>
  );
}
