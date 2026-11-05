/**
 * Where the dashboard's API and public site live -- resolved from build-time
 * environment (NEXT_PUBLIC_* values are inlined into the bundle when it is
 * built; setting them on a running server changes nothing).
 *
 * Only a development build (`next dev`) falls back to localhost, and the
 * fallback is passed in by the caller inside a literal
 * `process.env.NODE_ENV === "development" ? ... : null` so a production build
 * folds it away and ships no localhost URL at all: in production, localhost is
 * the visitor's own machine.
 */

/** A configured base URL (trailing slashes trimmed), else the dev fallback, else null. */
export function resolveBaseUrl(configured: string | undefined, devFallback: string | null): string | null {
  const v = configured?.trim().replace(/\/+$/, "");
  return v ? v : devFallback;
}
