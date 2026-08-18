#!/usr/bin/env node
/**
 * vendor.mjs — copy TypeGPU into the harness, and do the one thing a bundler
 * would have done for it.
 *
 *   node vendor.mjs
 *
 * WHY A SCRIPT AND NOT `cp -R`. TypeGPU's published ESM is not loadable by a
 * browser as shipped. `shared/env.js` reads `process.env.NODE_ENV` at module top
 * level, and its own comment says why:
 *
 *     Even though the value of this constant uses Node.js specific APIs, pretty
 *     much every bundler replaces the expression below with either
 *     `development` or `production`
 *
 * Under Node the file evaluates fine, which is what makes this easy to miss —
 * `import("typegpu")` from a test script succeeds. In a browser `process` is not
 * defined, the module throws a ReferenceError, and the entire 166-module graph
 * fails to evaluate. The harness is a plain `python3 -m http.server` with no
 * bundler, so the substitution has to happen here.
 *
 * DEV/TEST are set to the production branch, which is what a bundler emits for a
 * non-development build.
 *
 * Every step asserts. A silent no-op here would mean the adapter is quietly
 * skipped on the next reload, which is exactly the failure this replaced.
 */

import { cpSync, rmSync, readFileSync, writeFileSync, existsSync, readdirSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

process.chdir(dirname(fileURLToPath(import.meta.url)));

const SRC = "../../node_modules/typegpu";
const DST = "vendor/typegpu";
/**
 * `--check` verifies an already-vendored copy without recreating it. That is
 * what makes the two guards below testable — they can be pointed at a
 * deliberately broken tree — and it is what `npm test` runs, so a vendored copy
 * that stopped being browser-loadable fails there instead of in a browser
 * console nobody is reading.
 */
const CHECK_ONLY = process.argv.includes("--check");

if (CHECK_ONLY && !existsSync(DST)) {
  console.log("  typegpu not vendored — run `npm run vendor`. Skipping.");
  process.exit(0);
}
if (!CHECK_ONLY) {
  if (!existsSync(SRC)) {
    console.error(`  ${SRC} not found — run \`npm install\` first.`);
    process.exit(1);
  }
  rmSync("vendor", { recursive: true, force: true });
  cpSync(SRC, DST, { recursive: true });
}

// ---- the substitution a bundler would make ---------------------------------
const envFile = join(DST, "shared/env.js");
const env = readFileSync(envFile, "utf8");
const patched = CHECK_ONLY ? env : env
  .replace("export const DEV = process.env.NODE_ENV === 'development';",
           "export const DEV = false;   // vendor.mjs: bundler substitution, production branch")
  .replace("export const TEST = process.env.NODE_ENV === 'test';",
           "export const TEST = false;  // vendor.mjs: bundler substitution, production branch");
if (!CHECK_ONLY && patched === env) {
  console.error("  shared/env.js no longer matches the expected `process.env.NODE_ENV` form.\n" +
                "  TypeGPU changed it. Re-read the file before assuming this is still needed;\n" +
                "  leaving it unpatched would make the adapter fail silently in the browser.");
  process.exit(1);
}
if (!CHECK_ONLY) writeFileSync(envFile, patched);

// ---- nothing else may need a bundler ---------------------------------------
// Not a one-time observation: re-checked on every vendor, so a TypeGPU upgrade
// that introduces another host-specific reference fails here rather than in a
// browser console nobody is reading.
const js = [];
(function walk(dir) {
  for (const e of readdirSync(dir)) {
    const p = join(dir, e);
    if (statSync(p).isDirectory()) walk(p);
    else if (p.endsWith(".js")) js.push(p);
  }
})(DST);

const HOST_ONLY = [
  [/\bprocess\s*\./, "process.*"],
  [/\brequire\s*\(/, "require()"],
  [/from\s*["']node:/, "node: import"],
  [/\b__dirname\b|\b__filename\b/, "__dirname/__filename"],
];
const offenders = [];
for (const f of js) {
  const src = readFileSync(f, "utf8");
  for (const [re, what] of HOST_ONLY) {
    for (const [i, line] of src.split("\n").entries()) {
      if (line.trimStart().startsWith("*") || line.trimStart().startsWith("//")) continue;
      if (re.test(line)) offenders.push(`${f}:${i + 1}  ${what}  ${line.trim().slice(0, 80)}`);
    }
  }
}
if (offenders.length) {
  console.error("  vendored TypeGPU still needs a bundler:\n    " + offenders.join("\n    "));
  process.exit(1);
}

// ---- and the graph a browser will fetch must be complete --------------------
const seen = new Set(), missing = [];
(function walk(f) {
  if (seen.has(f)) return;
  seen.add(f);
  if (!existsSync(f)) return void missing.push(f);
  const src = readFileSync(f, "utf8"), dir = dirname(f);
  const specs = [
    ...src.matchAll(/(?:^|\n)\s*(?:import|export)[^;\n]*?from\s*["'](\.[^"']+)["']/g),
    ...src.matchAll(/import\s*\(\s*["'](\.[^"']+)["']\s*\)/g),
    ...src.matchAll(/(?:^|\n)\s*import\s*["'](\.[^"']+)["']/g),
  ];
  for (const m of specs) walk(resolve(dir, m[1]));
})(resolve("typegpu-runner.js"));

if (missing.length) {
  console.error("  incomplete module graph — the browser would 404 on:\n    " + missing.join("\n    "));
  process.exit(1);
}

console.log(CHECK_ONLY
  ? `vendored typegpu is browser-loadable: ${js.length} files scanned, ${seen.size} modules reachable, no host-only references`
  : `-> typegpu vendored: ${js.length} files, ${seen.size} modules reachable from the adapter,\n` +
    `   env.js substituted (DEV=false, TEST=false), no host-only references remain`);
