/**
 * Where the numbers on screen come from. There is no in-browser engine.
 *
 * - loading:  connecting to the API; no results shown yet.
 * - live:     the response of a live POST /backtest.
 * - recorded: API unreachable, and this exact configuration has a recorded
 *             snapshot of real engine output (scripts/record_engine_snapshot.py).
 * - offline:  API unreachable and nothing recorded for this configuration;
 *             no results are shown.
 * - rejected: the API answered and refused the request (e.g. an end date in
 *             the future); its explanation is shown.
 *
 * `hosted` below is a production build with no NEXT_PUBLIC_API_URL: the hosted
 * demo, which has no engine behind it by design and serves the recordings in
 * public/engine-snapshot/ with the page. "Unreachable" there would read as a
 * fault, so it says what it is. A build with an engine API configured keeps
 * "unreachable" for when that API does not answer.
 */
export type DataMode = "loading" | "live" | "recorded" | "offline" | "rejected";

const HOSTED = "Hosted demo on bundled data: no engine runs behind this page";

/** The mode badge's tooltip. */
export function modeTitle(mode: DataMode, hosted: boolean): string {
  switch (mode) {
    case "loading":
      return "Connecting to the AlphaForge engine API.";
    case "live":
      return "Results computed just now by the AlphaForge engine API.";
    case "recorded":
      return hosted
        ? `${HOSTED}; this is the real engine's recorded output for this exact configuration.`
        : "Engine API unreachable — showing recorded output of the real engine for this exact configuration.";
    case "rejected":
      return "The engine API answered and refused this request — see the message below.";
    case "offline":
      return hosted
        ? `${HOSTED}, and no bundled recording matches this configuration — no results.`
        : "Engine API unreachable and no recording exists for this configuration — no results.";
  }
}

/** Why the footer's recorded output is shown instead of a live run. */
export function recordedBecause(hosted: boolean): string {
  return hosted ? "the hosted demo has no engine behind it" : "the engine API is offline";
}
