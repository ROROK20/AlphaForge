// Unit tests for lib/modes.ts (run: `npm test`), transpiled and evaluated in
// place with the project's own `typescript`, like lib/urls.test.mjs.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { test } from "node:test";

const require = createRequire(import.meta.url);
const ts = require("typescript");
const source = readFileSync(new URL("./modes.ts", import.meta.url), "utf8");
const js = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 },
}).outputText;
const { modeTitle, recordedBecause } = await import(`data:text/javascript;base64,${Buffer.from(js).toString("base64")}`);

const MODES = ["loading", "live", "recorded", "offline", "rejected"];
const FAULT = /unreachable|offline|down\b/i;

test("the hosted demo (no engine API in the build) says so, not that an engine is unreachable", () => {
  for (const mode of ["recorded", "offline"]) {
    const title = modeTitle(mode, true);
    assert.match(title, /^Hosted demo on bundled data: no engine runs behind this page/, mode);
    assert.doesNotMatch(title, FAULT, mode);
  }
  assert.match(modeTitle("recorded", true), /recorded output for this exact configuration/);
  assert.match(modeTitle("offline", true), /no results\.$/);
  assert.doesNotMatch(recordedBecause(true), FAULT);
});

test("a build with an engine API that does not answer keeps the unreachable wording", () => {
  assert.equal(
    modeTitle("recorded", false),
    "Engine API unreachable — showing recorded output of the real engine for this exact configuration.",
  );
  assert.equal(
    modeTitle("offline", false),
    "Engine API unreachable and no recording exists for this configuration — no results.",
  );
  assert.equal(recordedBecause(false), "the engine API is offline");
});

test("modes where an engine answered, or nothing has yet, read the same in both builds", () => {
  for (const mode of ["loading", "live", "rejected"]) assert.equal(modeTitle(mode, true), modeTitle(mode, false), mode);
});

test("every mode has a tooltip in both builds", () => {
  for (const mode of MODES) for (const hosted of [true, false]) assert.ok(modeTitle(mode, hosted)?.length > 20, mode);
});
