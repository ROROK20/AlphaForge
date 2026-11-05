// Unit tests for lib/urls.ts (run: `npm test`). The module is TypeScript, so
// it is transpiled with the project's own `typescript` devDependency and
// evaluated in place -- no extra test runner. `npm run check:prerender` covers
// the production wiring end to end (no localhost URL in the built output).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { test } from "node:test";

const require = createRequire(import.meta.url);
const ts = require("typescript");
const source = readFileSync(new URL("./urls.ts", import.meta.url), "utf8");
const js = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 },
}).outputText;
const { resolveBaseUrl } = await import(`data:text/javascript;base64,${Buffer.from(js).toString("base64")}`);

test("an explicit URL wins over any fallback (trailing slashes trimmed)", () => {
  assert.equal(resolveBaseUrl("https://api.example.com/", "http://localhost:8003"), "https://api.example.com");
  assert.equal(resolveBaseUrl("https://api.example.com//", null), "https://api.example.com");
});

test("unset or blank uses the dev fallback when one is given", () => {
  assert.equal(resolveBaseUrl(undefined, "http://localhost:8003"), "http://localhost:8003");
  assert.equal(resolveBaseUrl("  ", "http://localhost:3001"), "http://localhost:3001");
});

test("unset with no fallback (a production build) is null -- never a localhost guess", () => {
  for (const configured of [undefined, "", "   "]) assert.equal(resolveBaseUrl(configured, null), null);
});
