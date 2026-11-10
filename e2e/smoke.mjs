/**
 * AlphaForge scripted live-fire smoke driver.
 *
 * Proves in ONE free, repeatable command that AlphaForge's reconciliation
 * invariant — "equity curve <-> trade-log P&L reconcile within 0.1% on every
 * shipped strategy" (README, "Testing") — holds over HTTP against the real Python
 * engine, that the engine's seeded output is reproducible, and that the
 * dashboard renders those same live numbers in a real browser with zero
 * console errors. With the API blocked in the browser it also proves the
 * offline mode never fabricates: the recorded snapshot it falls back to is the
 * live engine's own output (same figures, trade log reconciles), and any
 * configuration that was not recorded shows no numbers at all.
 *
 * It self-provisions: spawns the FastAPI engine (:8003) and the Next.js
 * dashboard (:3001), waits for readiness, asserts, and tears BOTH down in a
 * `finally` (also on failure). Nothing is left running.
 *
 *   node e2e/smoke.mjs            # from the repo root; exit 0 = all pass, 1 = any fail
 *   node e2e/smoke.mjs --capture  # ...and, only if the live checks pass, retake
 *                                 # docs/screenshot.png (the live engine) and
 *                                 # frontend/public/og.png (the recorded snapshot
 *                                 # a frontend-only deploy shows)
 *
 * This drives the Playwright *library* from a plain .mjs — NOT
 * @playwright/test, whose TS spec loader breaks under Node 23. Playwright is a
 * devDependency of frontend/ (this repo's only npm package), so it is
 * resolved from there even though this file lives at the repo root.
 *
 * The API process is spawned with OpenMP pinned to one thread. xgboost and
 * torch each bundle their own libomp and deadlock when both spin pools in one
 * process; tests/conftest.py pins this for the TEST process only and does not
 * cover a spawned server. Importing alphaforge.api.app pulls in neither library,
 * so this is defence in depth: the pin is free and the ml_momentum path is one
 * import away from needing it.
 */
import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import { setTimeout as sleep } from "node:timers/promises";
import { connect } from "node:net";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");
const FRONTEND = join(ROOT, "frontend");
const require = createRequire(join(FRONTEND, "package.json"));
const { chromium } = require("playwright");

const CAPTURE = process.argv.includes("--capture");
const API_PORT = 8003;
const UI_PORT = 3001;
const API = `http://127.0.0.1:${API_PORT}`;
const UI = `http://127.0.0.1:${UI_PORT}`;

// The dashboard's own default form state (frontend/app/page.tsx) — driving the
// API with exactly these means the UI section can compare against live values.
const INITIAL_CAPITAL = 100_000;
const REQUEST = {
  symbol: "AAPL",
  startDate: "2020-01-01",
  endDate: "2024-01-01",
  initialCapital: INITIAL_CAPITAL,
  commission: 0.001,
};

// The reconciliation invariant's hard bound: 0.1% of initial capital.
const RECONCILE_BOUND = 0.001 * INITIAL_CAPITAL;

/**
 * Engine output for REQUEST, re-derived 2026-11-03. The synthetic OHLCV feed is
 * seeded from the SYMBOL alone — one price path per ticker, shared by every
 * strategy — so these reproduce exactly: the "backtests reproducible from seed"
 * half of what this proves. If an engine change moves them, re-derive them; never
 * tune REQUEST to recover a number.
 */
const EXPECTED = {
  sma_crossover: { exec: "sma_crossover(20,50)", nTrades: 23, totalReturn: 0.066732, finalEquity: 106673.21, note: false },
  rsi_meanrev: { exec: "rsi_meanrev(14)", nTrades: 16, totalReturn: -0.051064, finalEquity: 94893.62, note: false },
  macd: { exec: "macd(12,26,9)", nTrades: 80, totalReturn: 0.02652, finalEquity: 102652.0, note: false },
  // The API does not spend an XGBoost walk-forward per unauthenticated request,
  // so it runs an SMA(10/30) proxy and MUST disclose that (an honest label is
  // part of the product).
  ml_momentum: { exec: "sma_crossover(10,30)", nTrades: 49, totalReturn: -0.489951, finalEquity: 51004.92, note: true },
};
// Buy-and-hold of the one AAPL path every strategy above shares.
const BENCHMARK_FINAL = 196097.22;

const fmtPct = (n) => `${(n * 100).toFixed(2)}%`; // mirrors frontend page.tsx
const near = (a, b, tol) => Math.abs(a - b) <= tol;
const dollars = (text) => Number(String(text).replace(/[^0-9.-]/g, ""));
// Every string a settled (non-loading) mode can show; none may be prerendered.
const SETTLED_COPY = ["Live engine", "Recorded snapshot", "Engine offline", "Request rejected", "Demo data", "backend offline"];

/** Trade-log net P&L (sum of the "Net P&L $" column) vs Final Equity - capital. */
async function uiReconciliation(page) {
  const cells = await page.locator("table tbody tr td:last-child").allTextContents();
  const rows = cells.filter((t) => /\$/.test(t));
  const tradeSum = rows.reduce((s, t) => s + dollars(t), 0);
  const finalText = await page.locator(".stat-row", { hasText: "Final Equity" }).locator("div").nth(1).textContent();
  const equityPnl = dollars(finalText) - INITIAL_CAPITAL;
  return { n: rows.length, tradeSum, equityPnl, gap: Math.abs(tradeSum - equityPnl), bound: 0.005 * (rows.length + 1) + 1e-9 };
}

// ---- pass/fail bookkeeping ------------------------------------------------
let pass = 0,
  fail = 0;
const ok = (n) => {
  pass++;
  console.log(`  ✓ ${n}`);
};
const bad = (n, e) => {
  fail++;
  console.log(`  ✗ ${n}${e ? " — " + e : ""}`);
};
async function check(name, fn) {
  try {
    await fn();
    ok(name);
  } catch (e) {
    bad(name, (e && e.message ? e.message : String(e)).split("\n")[0].slice(0, 180));
  }
}

// ---- server lifecycle -----------------------------------------------------
const children = [];
function launch(label, cmd, args, cwd, env) {
  // detached:true -> each server leads its own process group so teardown can
  // SIGKILL the whole tree (uvicorn, `next dev` compiler children).
  const child = spawn(cmd, args, {
    cwd,
    detached: true,
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env, ...(env || {}) },
  });
  child.label = label;
  child.errtail = [];
  const keep = (d) => {
    for (const line of String(d).split("\n")) {
      if (line.trim()) {
        child.errtail.push(line);
        if (child.errtail.length > 30) child.errtail.shift();
      }
    }
  };
  child.stderr.on("data", keep);
  child.stdout.on("data", keep);
  child.on("error", (e) => keep(`spawn error: ${e.message}`));
  children.push(child);
  return child;
}
function killAll() {
  for (const c of children) {
    if (c.pid == null || c.exitCode !== null) continue;
    try {
      process.kill(-c.pid, "SIGKILL");
    } catch {
      try {
        c.kill("SIGKILL");
      } catch {}
    }
  }
}
for (const sig of ["SIGINT", "SIGTERM"]) {
  process.on(sig, () => {
    killAll();
    process.exit(1);
  });
}

function portListening(port) {
  return new Promise((resolve) => {
    const s = connect({ host: "127.0.0.1", port }, () => {
      s.destroy();
      resolve(true);
    });
    s.on("error", () => resolve(false));
    s.setTimeout(1000, () => {
      s.destroy();
      resolve(false);
    });
  });
}
async function waitUntil(what, fn, timeoutMs, child) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (child && child.exitCode !== null) {
      throw new Error(`${child.label} exited early (code ${child.exitCode}): ${child.errtail.slice(-6).join(" | ")}`);
    }
    try {
      if (await fn()) return;
    } catch {}
    await sleep(400);
  }
  throw new Error(`timeout waiting for ${what} after ${timeoutMs}ms${child ? ": " + child.errtail.slice(-6).join(" | ") : ""}`);
}

async function postBacktest(strategy) {
  const res = await fetch(`${API}/backtest`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ ...REQUEST, strategy }),
    signal: AbortSignal.timeout(30000),
  });
  if (res.status !== 200) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

let browser;
const t0 = Date.now();
try {
  // Preflight: refuse to run against a server already on our ports (that would be
  // a false pass on whatever it is running). 8003/3001 are alphaforge's own ports; nothing
  // else is ever probed or touched.
  for (const [port, name] of [[API_PORT, "API"], [UI_PORT, "frontend"]]) {
    if (await portListening(port)) {
      throw new Error(`port ${port} (${name}) already in use — stop the stale process and re-run`);
    }
  }

  console.log("Starting servers:");
  // Pin OpenMP in the SERVER's own environment (see the header comment). torch honours
  // OMP_NUM_THREADS for its intra-op thread count, so this is the torch
  // single-thread pin too; conftest.py's pin covers only the pytest process.
  const apiProc = launch(
    "api",
    "python3",
    ["-m", "uvicorn", "alphaforge.api.app:app", "--host", "127.0.0.1", "--port", String(API_PORT)],
    ROOT,
    { OMP_NUM_THREADS: "1", MKL_NUM_THREADS: "1", OPENBLAS_NUM_THREADS: "1", NUMEXPR_NUM_THREADS: "1" },
  );
  await check("API up (/health ok)", () =>
    waitUntil(
      "API /health",
      async () => {
        const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(2000) });
        return r.ok && (await r.json()).status === "ok";
      },
      45000,
      apiProc,
    ),
  );

  const uiProc = launch("frontend", "npm", ["run", "dev"], FRONTEND);
  await check("frontend listening (:3001)", () => waitUntil("frontend port", () => portListening(UI_PORT), 90000, uiProc));

  // ---- API live-fire: the reconciliation reproduces over HTTP ---------------
  console.log("API POST /backtest — reconciliation (equity curve ↔ trade log):");
  const live = {};
  for (const [strategy, want] of Object.entries(EXPECTED)) {
    let data;
    await check(`POST /backtest {${strategy}} -> 200`, async () => {
      data = await postBacktest(strategy);
      live[strategy] = data;
    });
    if (!data) continue;

    await check(`${strategy}: engine ran ${want.exec}, ${want.nTrades} trades (seed-reproducible)`, () => {
      if (data.strategyExecuted !== want.exec) throw new Error(`strategyExecuted ${data.strategyExecuted} ≠ ${want.exec}`);
      if (data.trades.length !== want.nTrades) throw new Error(`${data.trades.length} trades ≠ ${want.nTrades}`);
      if (data.metrics.nTrades !== want.nTrades) throw new Error(`metrics.nTrades ${data.metrics.nTrades} ≠ ${want.nTrades}`);
      if (!near(data.metrics.totalReturn, want.totalReturn, 5e-6)) throw new Error(`totalReturn ${data.metrics.totalReturn} ≠ ${want.totalReturn}`);
      const finalEquity = data.equity[data.equity.length - 1].equity;
      if (!near(finalEquity, want.finalEquity, 0.01)) throw new Error(`final equity ${finalEquity} ≠ ${want.finalEquity}`);
    });

    await check(`${strategy}: Σ pnlNetAbs reconciles with the equity curve`, () => {
      const tradePnl = data.trades.reduce((s, t) => s + t.pnlNetAbs, 0);
      const equityPnl = data.equity[data.equity.length - 1].equity - INITIAL_CAPITAL;
      const gap = Math.abs(tradePnl - equityPnl);
      // The wire rounds every pnlNetAbs and the equity to 2dp, so the smallest
      // honest bound over HTTP is half a cent per rounded figure. In-process the
      // reconciliation is exact to ~1e-10 (pinned by tests/test_backtest.py).
      const wireBound = 0.005 * (data.trades.length + 1) + 1e-9;
      if (gap > wireBound) {
        throw new Error(`gap $${gap.toFixed(4)} exceeds wire-rounding bound $${wireBound.toFixed(4)} (${((gap / INITIAL_CAPITAL) * 100).toFixed(4)}% of initial)`);
      }
      if (gap >= RECONCILE_BOUND) throw new Error(`gap $${gap.toFixed(2)} breaches the 0.1% reconciliation bound`);
    });

    await check(`${strategy}: ${want.note ? "discloses the SMA proxy" : "no disclosure note (real strategy)"}`, () => {
      const hasNote = data.note != null && data.note !== "";
      if (hasNote !== want.note) throw new Error(`note ${hasNote ? "present" : "absent"}, expected ${want.note ? "present" : "absent"}`);
      if (want.note && !data.note.includes("SMA(10/30)")) throw new Error("note does not name the SMA(10/30) proxy");
    });
  }

  // One price path per symbol: every strategy's buy-and-hold benchmark is the
  // same series.
  await check(`one AAPL price path for all strategies (benchmark $${BENCHMARK_FINAL.toLocaleString("en-US")})`, () => {
    const runs = Object.entries(live);
    if (runs.length !== Object.keys(EXPECTED).length) throw new Error(`only ${runs.length} live runs`);
    const ref = JSON.stringify(runs[0][1].equity.map((p) => p.benchmark));
    for (const [strategy, data] of runs) {
      if (JSON.stringify(data.equity.map((p) => p.benchmark)) !== ref) throw new Error(`${strategy} ran on a different price path`);
    }
    const final = runs[0][1].equity[runs[0][1].equity.length - 1].benchmark;
    if (!near(final, BENCHMARK_FINAL, 0.01)) throw new Error(`benchmark final ${final} ≠ ${BENCHMARK_FINAL}`);
  });

  // Trade-log dates are fill dates, on prices we know: every logged price is the
  // close of the date beside it, +/- slippage (the last row's mark is the bare close).
  for (const fill of ["close", "next_close"]) {
    await check(`trade log (fill=${fill}): each price is the close of the date beside it`, async () => {
      const bars = Array.from({ length: 200 }, (_, i) => {
        const date = new Date(Date.UTC(2021, 0, 4 + i)).toISOString().slice(0, 10);
        const c = Number((100 + 15 * Math.sin(i / 6) + 0.05 * i).toFixed(4));
        return { date, open: c, high: c, low: c, close: c };
      });
      const slip = 0.01;
      const res = await fetch(`${API}/backtest`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          symbol: "TEST", strategy: "macd", startDate: bars[0].date, endDate: bars[199].date,
          initialCapital: INITIAL_CAPITAL, commission: 0.001, slippage: slip, fill, ohlcv: bars,
        }),
        signal: AbortSignal.timeout(30000),
      });
      if (res.status !== 200) throw new Error(`HTTP ${res.status}`);
      const { trades } = await res.json();
      if (trades.length < 3) throw new Error(`only ${trades.length} trades`);
      const close = new Map(bars.map((b) => [b.date, b.close]));
      for (const t of trades) {
        const d = t.direction === "long" ? 1 : -1;
        const wantEntry = close.get(t.entryDate) * (1 + d * slip);
        const wantExit = close.get(t.exitDate) * (t.exitDate === bars[199].date ? 1 : 1 - d * slip);
        if (!near(t.entryPrice, wantEntry, 0.0051)) throw new Error(`#${t.id} entry ${t.entryPrice} on ${t.entryDate}: want ${wantEntry}`);
        if (!near(t.exitPrice, wantExit, 0.0051)) throw new Error(`#${t.id} exit ${t.exitPrice} on ${t.exitDate}: want ${wantExit}`);
      }
    });
  }

  // ---- the prerendered HTML claims no settled mode and shows no numbers -----
  // (a page that renders "offline" before it has even tried the API would
  // make a false claim on a perfectly healthy engine)
  await check("prerendered HTML is the honest Connecting state (no mode claim, no figures)", async () => {
    const html = await (await fetch(UI, { signal: AbortSignal.timeout(90000) })).text();
    if (!html.includes("Connecting to the AlphaForge engine")) throw new Error("no Connecting footer in SSR HTML");
    const claimed = SETTLED_COPY.filter((c) => html.includes(c));
    if (claimed.length) throw new Error(`SSR HTML claims: ${claimed.join(", ")}`);
    const sma = EXPECTED.sma_crossover;
    for (const fig of [fmtPct(sma.totalReturn), sma.finalEquity.toLocaleString("en-US")]) {
      if (html.includes(fig)) throw new Error(`SSR HTML already shows ${fig}`);
    }
  });

  // ---- UI live-fire: real browser, real backend, 0 console errors ----------
  console.log("UI (headless Chromium @ :3001):");
  browser = await chromium.launch();
  const page = await (await browser.newContext()).newPage();
  const consoleErrors = [];
  page.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text());
  });
  page.on("pageerror", (e) => consoleErrors.push(`pageerror: ${e.message}`));
  await page.goto(UI, { waitUntil: "domcontentloaded", timeout: 90000 });

  await check('dashboard shows "Live engine" (real backend, not demo)', async () => {
    // exact:true so this cannot substring-match other copy, and waitFor
    // blocks through the "Connecting…" / demo-fallback flash on cold start
    // until the badge settles on the real-engine state.
    await page.getByText("Live engine", { exact: true }).first().waitFor({ timeout: 45000 });
    const body = await page.innerText("body");
    const other = SETTLED_COPY.filter((c) => c !== "Live engine" && body.includes(c));
    if (other.length) throw new Error(`live page also claims: ${other.join(", ")}`);
    if ((await page.getByTestId("mode-footer").innerText()) !== "Live results from the AlphaForge engine") {
      throw new Error("footer does not say live");
    }
  });

  // The dashboard's default form state is exactly REQUEST + sma_crossover, so
  // the rendered figures must equal the live API's. (The exact "Live engine"
  // badge above is what proves the mode: the offline fallback is the recorded
  // snapshot of these same engine figures, checked further down.)
  const sma = live.sma_crossover;
  const uiText = await page.innerText("body");
  await check(`UI Total Return ${fmtPct(EXPECTED.sma_crossover.totalReturn)} matches the live engine`, () => {
    if (!sma) throw new Error("no live API response to compare against");
    const want = fmtPct(sma.metrics.totalReturn);
    if (want !== fmtPct(EXPECTED.sma_crossover.totalReturn)) throw new Error(`API drifted from the documented value: ${want}`);
    if (!uiText.includes(want)) throw new Error(`"${want}" not rendered`);
  });
  await check("UI states both per-side cost legs (commission + slippage)", () => {
    if (!uiText.includes("Costs per side: 0.1% commission + 0.05% slippage")) throw new Error("cost caption missing");
    if (!sma || sma.commission !== 0.001 || sma.slippage !== 0.0005) throw new Error("API did not echo the costs");
  });
  await check(`UI trade log shows ${EXPECTED.sma_crossover.nTrades} trades (live engine count)`, () => {
    if (!uiText.includes(`${EXPECTED.sma_crossover.nTrades} trades`)) throw new Error(`"${EXPECTED.sma_crossover.nTrades} trades" not rendered`);
  });
  await check(`UI Final Equity $${EXPECTED.sma_crossover.finalEquity.toLocaleString("en-US", { minimumFractionDigits: 2 })} matches the reconciled curve`, () => {
    const want = `$${EXPECTED.sma_crossover.finalEquity.toLocaleString("en-US")}`;
    if (!uiText.includes(want)) throw new Error(`"${want}" not rendered`);
  });
  await check('UI trade log labels the signed trade return "Return %" (not a price move)', async () => {
    const headers = await page.locator("table thead th").allInnerTexts();
    if (!headers.map((h) => h.trim().toUpperCase()).includes("RETURN %")) throw new Error(`headers: ${headers.join(" | ")}`);
    if (headers.some((h) => /price move/i.test(h))) throw new Error("mislabelled column still present");
  });
  await check("UI trade log's net P&L reconciles with Final Equity - capital (live)", async () => {
    const r = await uiReconciliation(page);
    if (r.n !== EXPECTED.sma_crossover.nTrades) throw new Error(`${r.n} trade rows`);
    if (r.gap > r.bound) throw new Error(`Σ net $${r.tradeSum.toFixed(2)} vs equity $${r.equityPnl.toFixed(2)}`);
  });

  await check('UI "vs Buy-Hold" is the gap between the two total returns, in points', async () => {
    if (!sma) throw new Error("no live API response to compare against");
    const end = sma.equity[sma.equity.length - 1];
    const gap = (end.equity - end.benchmark) / INITIAL_CAPITAL;
    const want = `${gap > 0 ? "+" : ""}${(gap * 100).toFixed(2)} pts`;
    const shown = await page.locator(".stat-row", { hasText: "vs Buy-Hold" }).locator("div").nth(1).textContent();
    if (shown !== want) throw new Error(`"${shown}" != "${want}"`);
  });
  await check("UI trade log row 1 shows the live engine's fill dates and prices", async () => {
    if (!sma) throw new Error("no live API response to compare against");
    const cells = (await page.locator("table tbody tr").first().locator("td").allInnerTexts()).map((c) => c.trim());
    const t = sma.trades[0];
    const want = { 1: t.entryDate, 2: t.exitDate, 4: `$${t.entryPrice}`, 5: `$${t.exitPrice}` };
    for (const [i, v] of Object.entries(want)) {
      if (cells[i] !== v) throw new Error(`cell ${i} "${cells[i]}" != "${v}"`);
    }
  });

  if (CAPTURE) {
    // Only reached with the live page verified above. Each shot names the one
    // state it may show and refuses any other. The README's hero image is the
    // live engine. The social card is what a frontend-only deploy serves at the
    // same URL: with no engine, the recorded snapshot.
    await check('capture docs/screenshot.png ("Live engine") + frontend/public/og.png ("Recorded snapshot")', async () => {
      const shots = [
        { path: join(ROOT, "docs", "screenshot.png"), viewport: { width: 1920, height: 1080 }, scale: 1.5, fullPage: true, badge: "Live engine", offline: false },
        { path: join(FRONTEND, "public", "og.png"), viewport: { width: 1200, height: 630 }, scale: 2, fullPage: false, badge: "Recorded snapshot", offline: true },
      ];
      for (const shot of shots) {
        const ctx = await browser.newContext({ viewport: shot.viewport, deviceScaleFactor: shot.scale });
        if (shot.offline) await ctx.route((url) => new URL(url).port === String(API_PORT), (route) => route.abort());
        const p = await ctx.newPage();
        await p.goto(UI, { waitUntil: "domcontentloaded", timeout: 90000 });
        await p.getByText(shot.badge, { exact: true }).first().waitFor({ timeout: 45000 });
        for (const other of SETTLED_COPY.filter((c) => c !== shot.badge)) {
          if (await p.getByText(other, { exact: true }).count()) throw new Error(`${shot.path} would show "${other}" under "${shot.badge}"`);
        }
        await p.getByText(fmtPct(EXPECTED.sma_crossover.totalReturn), { exact: true }).first().waitFor({ timeout: 30000 });
        // Hide `next dev`'s own on-page indicator: it is a dev-server overlay,
        // not part of the app, and would otherwise sit on top of the chart.
        await p.addStyleTag({ content: "nextjs-portal { display: none !important; }" });
        if (shot.fullPage) {
          // Size the viewport to the whole page FIRST: a fullPage capture
          // otherwise resizes it mid-shot, Recharts' ResponsiveContainer
          // re-renders, and its entry animation restarts from an empty chart.
          const height = await p.evaluate(() => document.documentElement.scrollHeight);
          await p.setViewportSize({ width: shot.viewport.width, height });
        }
        // Let the charts finish their entry animation at the final size, and
        // require the equity line to be drawn before shooting.
        await p.waitForTimeout(3000);
        await p.waitForFunction(() => {
          const path = document.querySelector(".recharts-area-curve");
          return path && (path.getAttribute("d") || "").length > 1000;
        }, null, { timeout: 30000 });
        await p.screenshot({ path: shot.path, fullPage: false });
        await ctx.close();
        console.log(`    wrote ${shot.path}`);
      }
    });
  }

  // The ml_momentum proxy disclosure must survive into the UI.
  await check("UI discloses the ml_momentum SMA proxy when that strategy is picked", async () => {
    // The selects carry no id/for pair, so target the one that actually owns
    // the option rather than relying on DOM order.
    await page
      .locator('select:has(option[value="ml_momentum"])')
      .selectOption("ml_momentum");
    const note = page.getByTestId("strategy-note");
    await note.waitFor({ timeout: 30000 });
    const text = await note.innerText();
    if (!text.includes("SMA(10/30)")) throw new Error(`note does not name the proxy: ${text.slice(0, 80)}`);
    await page.getByText("Live engine", { exact: true }).first().waitFor({ timeout: 30000 });
  });

  // Chart date axes must stay legible at every width the README/OG images and
  // a phone use: no two x-axis tick labels may overlap.
  for (const width of [390, 1200, 1920]) {
    await check(`chart x-axis tick labels never overlap at ${width}px`, async () => {
      const ctx = await browser.newContext({ viewport: { width, height: 900 } });
      const p = await ctx.newPage();
      try {
        await p.goto(UI, { waitUntil: "domcontentloaded", timeout: 90000 });
        await p.getByText("Live engine", { exact: true }).first().waitFor({ timeout: 45000 });
        await p.waitForFunction(() => document.querySelectorAll(".recharts-xAxis .recharts-cartesian-axis-tick-value").length > 0, null, { timeout: 30000 });
        await p.waitForTimeout(1500);
        const overlaps = await p.evaluate(() =>
          [...document.querySelectorAll(".recharts-xAxis")].flatMap((axis, k) => {
            const boxes = [...axis.querySelectorAll(".recharts-cartesian-axis-tick-value")]
              .map((t) => t.getBoundingClientRect())
              .sort((a, b) => a.left - b.left);
            const bad = [];
            for (let i = 1; i < boxes.length; i++) {
              if (boxes[i].left < boxes[i - 1].right) bad.push(`chart ${k}: tick ${i} overlaps by ${(boxes[i - 1].right - boxes[i].left).toFixed(0)}px`);
            }
            return bad;
          }),
        );
        if (overlaps.length) throw new Error(`${overlaps.length} overlaps: ${overlaps.slice(0, 3).join("; ")}`);
      } finally {
        await ctx.close();
      }
    });
  }

  // On a phone every form field is at least 16px, since iOS Safari zooms the page
  // into a focused field set smaller, and the selects stand as tall as the inputs.
  await check("form fields at 390px: 16px text, every field the same height", async () => {
    const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
    const p = await ctx.newPage();
    try {
      await p.goto(UI, { waitUntil: "domcontentloaded", timeout: 90000 });
      await p.getByText("Live engine", { exact: true }).first().waitFor({ timeout: 45000 });
      const fields = await p.evaluate(() =>
        [...document.querySelectorAll(".field")].map((el) => ({
          id: el.id,
          px: parseFloat(getComputedStyle(el).fontSize),
          h: Math.round(el.getBoundingClientRect().height),
        })),
      );
      if (fields.length < 6) throw new Error(`found ${fields.length} fields`);
      const small = fields.filter((f) => f.px < 16);
      if (small.length) throw new Error(`under 16px: ${small.map((f) => `${f.id} ${f.px}px`).join(", ")}`);
      if (new Set(fields.map((f) => f.h)).size !== 1) {
        throw new Error(`heights differ: ${fields.map((f) => `${f.id} ${f.h}px`).join(", ")}`);
      }
    } finally {
      await ctx.close();
    }
  });

  // The trade log is wider than a phone and scrolls sideways inside its box. A line
  // above it says so, and is there only while the log is measured to overflow.
  for (const [width, phone] of [[375, true], [1440, false]]) {
    await check(`trade log at ${width}px: the scroll hint shows ${phone ? "because the log overflows" : "nowhere, as the log fits"}`, async () => {
      const ctx = await browser.newContext({ viewport: { width, height: 900 }, ...(phone ? { isMobile: true, hasTouch: true } : {}) });
      const p = await ctx.newPage();
      const state = () =>
        p.evaluate(() => {
          const box = document.querySelector(".table-scroll");
          return { overflows: box.scrollWidth > box.clientWidth, hint: document.querySelector("[data-scroll-hint]") !== null };
        });
      try {
        await p.goto(UI, { waitUntil: "domcontentloaded", timeout: 90000 });
        await p.getByText("Live engine", { exact: true }).first().waitFor({ timeout: 45000 });
        // The hint follows a measurement made after the page settles.
        await p
          .waitForFunction(
            () => {
              const box = document.querySelector(".table-scroll");
              return (document.querySelector("[data-scroll-hint]") !== null) === box.scrollWidth > box.clientWidth;
            },
            null,
            { timeout: 10000 },
          )
          .catch(() => {});
        const s = await state();
        if (s.overflows !== phone) throw new Error(`the log ${s.overflows ? "overflows" : "fits"} at ${width}px`);
        if (s.hint !== s.overflows) throw new Error(`the hint is ${s.hint ? "shown" : "missing"} while the log ${s.overflows ? "overflows" : "fits"}`);
      } finally {
        await ctx.close();
      }
    });
  }

  await check("0 console errors", () => {
    if (consoleErrors.length) throw new Error(`${consoleErrors.length}: ${consoleErrors.slice(0, 3).join(" | ")}`);
  });

  // ---- UI when the engine answers "no": its reason, not "unreachable" ------
  // (own context: Chromium logs the 422 response as a console error, which the
  // zero-console-errors check above must not see)
  {
    const ctx = await browser.newContext();
    const p = await ctx.newPage();
    await check("a rejected request shows the API's reason (end date in the future), not \"offline\"", async () => {
      await p.goto(UI, { waitUntil: "domcontentloaded", timeout: 90000 });
      await p.getByText("Live engine", { exact: true }).first().waitFor({ timeout: 45000 });
      await p.locator('input[type="date"]').nth(1).fill("2099-01-01");
      await p.getByText("Request rejected", { exact: true }).first().waitFor({ timeout: 30000 });
      const notice = await p.getByTestId("rejected-notice").innerText();
      if (!notice.includes("future")) throw new Error(`notice lacks the API's reason: ${notice.slice(0, 120)}`);
      const body = await p.innerText("body");
      if (body.includes("Engine offline") || body.includes("Recorded snapshot")) throw new Error("rejection reported as offline");
    });
    await ctx.close();
  }

  // ---- UI with the engine unreachable: recorded snapshot or nothing --------
  console.log("UI with the API blocked in the browser (offline mode):");
  const offline = await browser.newContext();
  await offline.route((url) => new URL(url).port === String(API_PORT), (route) => route.abort());
  const off = await offline.newPage();
  await off.goto(UI, { waitUntil: "domcontentloaded", timeout: 90000 });
  await check('offline default config shows the "Recorded snapshot" of the real engine', async () => {
    await off.getByText("Recorded snapshot", { exact: true }).first().waitFor({ timeout: 45000 });
    const footer = await off.getByTestId("mode-footer").innerText();
    if (!/^Recorded engine output \(recorded \d{4}-\d{2}-\d{2}\)/.test(footer)) throw new Error(`footer: ${footer}`);
    const body = await off.innerText("body");
    if (body.includes("Live engine")) throw new Error("offline page claims Live engine");
  });
  await check("recorded snapshot = the live engine's figures (Total Return, Final Equity, trades)", async () => {
    const body = await off.innerText("body");
    const sma = live.sma_crossover;
    if (!sma) throw new Error("no live API response to compare against");
    const want = [fmtPct(sma.metrics.totalReturn), `$${sma.equity[sma.equity.length - 1].equity.toLocaleString("en-US")}`, `${sma.trades.length} trades`];
    const missing = want.filter((w) => !body.includes(w));
    if (missing.length) throw new Error(`offline page lacks live figures: ${missing.join(", ")}`);
  });
  await check('the social card\'s state: at 1200x630 the "Recorded snapshot" badge is above the fold', async () => {
    const card = await browser.newContext({ viewport: { width: 1200, height: 630 } });
    await card.route((url) => new URL(url).port === String(API_PORT), (route) => route.abort());
    const p = await card.newPage();
    try {
      await p.goto(UI, { waitUntil: "domcontentloaded", timeout: 90000 });
      const badge = p.getByText("Recorded snapshot", { exact: true }).first();
      await badge.waitFor({ timeout: 45000 });
      const box = await badge.boundingBox();
      if (!box || box.y + box.height > 630) throw new Error(`badge outside the card: ${JSON.stringify(box)}`);
      if (await p.getByText("Live engine", { exact: true }).count()) throw new Error('the card state claims "Live engine"');
    } finally {
      await card.close();
    }
  });
  // A phone, not a narrow desktop window: the hyphen break shows with mobile layout.
  await check("on a 390px phone the footer's recorded date stays on one line", async () => {
    const phone = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
    await phone.route((url) => new URL(url).port === String(API_PORT), (route) => route.abort());
    const p = await phone.newPage();
    try {
      await p.goto(UI, { waitUntil: "domcontentloaded", timeout: 90000 });
      await p.getByText("Recorded snapshot", { exact: true }).first().waitFor({ timeout: 45000 });
      const lines = await p.evaluate(() => {
        const footer = document.querySelector('[data-testid="mode-footer"]');
        const walker = document.createTreeWalker(footer, NodeFilter.SHOW_TEXT);
        for (let node = walker.nextNode(); node; node = walker.nextNode()) {
          const i = node.textContent.search(/\d{4}-\d{2}-\d{2}/);
          if (i < 0) continue;
          const range = document.createRange();
          range.setStart(node, i);
          range.setEnd(node, i + 10);
          return new Set([...range.getClientRects()].map((r) => Math.round(r.top))).size;
        }
        return 0;
      });
      if (lines !== 1) throw new Error(`the recorded date spans ${lines} lines`);
    } finally {
      await phone.close();
    }
  });
  await check("offline trade log's net P&L reconciles with Final Equity - capital", async () => {
    const r = await uiReconciliation(off);
    if (r.n === 0) throw new Error("no trade rows");
    if (r.gap > r.bound) throw new Error(`Σ trade P&L $${r.tradeSum.toFixed(2)} vs equity P&L $${r.equityPnl.toFixed(2)} — the offline data does not reconcile`);
  });
  await check('offline + unrecorded config (MSFT) shows "Engine offline" and no figures', async () => {
    await off.locator('select:has(option[value="MSFT"])').selectOption("MSFT");
    await off.getByText("Engine offline", { exact: true }).first().waitFor({ timeout: 45000 });
    const grid = await off.getByTestId("metrics").innerText();
    if (/\d/.test(grid)) throw new Error(`metrics grid shows numbers offline: ${grid.replace(/\s+/g, " ").slice(0, 120)}`);
    const rows = await off.locator("table tbody tr td:last-child").allTextContents();
    if (rows.some((t) => /\$/.test(t))) throw new Error("trade rows shown offline");
  });
  await offline.close();
} catch (e) {
  fail++;
  console.log(`\n✗ FATAL — ${e && e.message ? e.message : String(e)}`);
} finally {
  if (browser) {
    try {
      await browser.close();
    } catch {}
  }
  killAll();
}

console.log(`\n${pass} passed, ${fail} failed  (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
process.exit(fail === 0 ? 0 : 1);
