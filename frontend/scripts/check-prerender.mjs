// Checks a PRODUCTION build (`npm run build` first):  npm run check:prerender
//
// 1. The prerendered home page is the honest "Connecting" state: it claims no
//    settled data mode and shows no figures computed by nobody.
// 2. Built without NEXT_PUBLIC_API_URL / NEXT_PUBLIC_SITE_URL, nothing the
//    browser receives points at localhost (the visitor's own machine): not
//    the prerendered HTML (metadata / Open Graph links) and not the client JS.
//    With those variables set, the check confirms they were baked in instead.
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

const root = new URL("..", import.meta.url).pathname;
const html = readFileSync(join(root, ".next/server/app/index.html"), "utf8");
const failures = [];

if (!html.includes("Connecting to the AlphaForge engine")) failures.push("prerendered page is not the Connecting state");
for (const claim of ["Live engine", "Recorded snapshot", "Engine offline", "Request rejected", "Demo data"]) {
  if (html.includes(claim)) failures.push(`prerendered page claims "${claim}"`);
}

function files(dir) {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    return statSync(path).isDirectory() ? files(path) : [path];
  });
}
const chunks = files(join(root, ".next/static")).filter((f) => f.endsWith(".js"));
// A localhost URL (not the bare word: Next's URL polyfill compares hosts to
// "localhost", which is harmless).
const LOCALHOST_URL = /https?:\/\/localhost[:/]/;
const apiUrl = process.env.NEXT_PUBLIC_API_URL;
const siteUrl = process.env.NEXT_PUBLIC_SITE_URL;
if (!siteUrl && LOCALHOST_URL.test(html)) failures.push("prerendered HTML links to localhost");
if (!apiUrl) {
  const leaking = chunks.filter((f) => LOCALHOST_URL.test(readFileSync(f, "utf8")));
  if (leaking.length) failures.push(`client JS calls localhost: ${leaking.map((f) => f.slice(root.length)).join(", ")}`);
} else if (!chunks.some((f) => readFileSync(f, "utf8").includes(apiUrl.replace(/\/+$/, "")))) {
  failures.push(`NEXT_PUBLIC_API_URL=${apiUrl} is not in the client bundle -- was it set at BUILD time?`);
}

if (failures.length) {
  console.error("check-prerender FAILED:\n  - " + failures.join("\n  - "));
  process.exit(1);
}
console.log(`check-prerender OK (${chunks.length} client chunks; API URL ${apiUrl ? "baked in" : "unset -> no network calls"})`);
