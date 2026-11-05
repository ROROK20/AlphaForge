import type { Metadata } from "next";
import { resolveBaseUrl } from "@/lib/urls";
import "./globals.css";

// Absolute Open Graph / Twitter image links need the public site's URL. A
// production build without NEXT_PUBLIC_SITE_URL emits no metadataBase and no
// share image rather than a link to the visitor's localhost.
const SITE_URL = resolveBaseUrl(
  process.env.NEXT_PUBLIC_SITE_URL,
  process.env.NODE_ENV === "development" ? "http://localhost:3001" : null,
);
// The card is a capture (e2e/smoke.mjs --capture) of the state a frontend-only
// deploy shows: the engine's recorded output. Both lines stay true on a deploy with
// a backend too: the card pictures a recording either way.
const OG_IMAGE = {
  url: "/og.png",
  width: 1200,
  height: 630,
  alt: "AlphaForge dashboard: the engine's recorded output for an SMA crossover on synthetic prices",
};

const OG_TITLE = "AlphaForge — Quant Trading Studio";
const OG_DESCRIPTION =
  "Open-source Python framework for backtesting, ML signals and portfolio optimization. Pictured: the engine's recorded output on synthetic prices, not market data.";

export const metadata: Metadata = {
  ...(SITE_URL ? { metadataBase: new URL(SITE_URL) } : {}),
  title: {
    default: OG_TITLE,
    template: "%s · AlphaForge",
  },
  description:
    "Python backtesting framework that keeps itself honest, with ML signals, portfolio optimization and a web dashboard.",
  keywords: [
    "quantitative trading",
    "backtesting",
    "algorithmic trading",
    "portfolio optimization",
    "machine learning",
    "finance",
  ],
  authors: [{ name: "Rohan Khiani", url: "https://github.com/ROROK20" }],
  creator: "@ROROK20",
  openGraph: {
    title: OG_TITLE,
    description: OG_DESCRIPTION,
    type: "website",
    url: "https://github.com/ROROK20/AlphaForge",
    siteName: "AlphaForge",
    ...(SITE_URL ? { images: [OG_IMAGE] } : {}),
  },
  twitter: {
    card: SITE_URL ? "summary_large_image" : "summary",
    title: OG_TITLE,
    description: OG_DESCRIPTION,
    creator: "@ROROK20",
    ...(SITE_URL ? { images: [OG_IMAGE.url] } : {}),
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="font-mono antialiased">{children}</body>
    </html>
  );
}
