#!/usr/bin/env node
/**
 * vendor.mjs — make TypeGPU loadable by a browser with no bundler.
 *
 *   node vendor.mjs            copy, rewrite, patch, verify
 *   node vendor.mjs --check    verify an existing tree (this is what npm test runs)
 *
 * The harness is a plain `python3 -m http.server` rooted at this directory, so it
 * cannot see the repo root's node_modules and there is no build step. TypeGPU's
 * published ESM does not survive that, in two independent ways:
 *
 *   1. `shared/env.js` reads `process.env.NODE_ENV` at module top level. Its own
 *      comment says "pretty much every bundler replaces the expression below",
 *      which is true and means the package as published is not loadable by the
 *      runtime it targets. In a browser this is a ReferenceError that takes the
 *      whole module graph down.
 *
 *   2. It imports three runtime dependencies by bare specifier — tsover-runtime,
 *      typed-binary, tinyest — which a browser cannot resolve without an import
 *      map. They are rewritten to relative paths here rather than pushed into an
 *      import map, so that typegpu-runner.js stays importable from any page and
 *      from Node without either knowing about this.
 *
 * Both are invisible from Node: `await import("typegpu")` succeeds there, because
 * Node has `process` and resolves bare specifiers. So neither is caught by
 * anything short of actually loading the tree the way a browser will.
 *
 * EVERY CHECK HERE HAS A POSITIVE CONTROL. An earlier version of this reported
 * "0 bare imports" from a grep that only matched double quotes, while TypeGPU
 * writes `from 'tsover-runtime'` in single quotes. A check that returns zero is
 * worthless unless it has been shown capable of returning non-zero, so the
 * scanner is required to find the problems BEFORE the rewrite and none after.
 */

import {
  cpSync, rmSync, readFileSync, writeFileSync, existsSync, readdirSync, statSync,
} from "node:fs";
import { dirname, join, resolve, relative } from "node:path";
import { fileURLToPath } from "node:url";

process.chdir(dirname(fileURLToPath(import.meta.url)));

const NM = "../../node_modules";
const VENDOR = "vendor";
const ROOT_PKG = "typegpu";
const CHECK_ONLY = process.argv.includes("--check");

const die = (msg) => { console.error("  " + msg.replace(/\n/g, "\n  ")); process.exit(1); };

// ---------------------------------------------------------------------------
// Which packages have to come along
// ---------------------------------------------------------------------------
/** Walk `dependencies` from typegpu so a new runtime dep is picked up, not missed. */
function transitiveDeps(name, seen = new Set()) {
  if (seen.has(name)) return seen;
  seen.add(name);
  const pj = join(NM, name, "package.json");
  if (!existsSync(pj)) die(`${name} is not installed — run \`npm install\`.`);
  for (const dep of Object.keys(JSON.parse(readFileSync(pj, "utf8")).dependencies ?? {})) {
    transitiveDeps(dep, seen);
  }
  return seen;
}
const PKGS = [...transitiveDeps(ROOT_PKG)];

// ---------------------------------------------------------------------------
// Resolving a bare specifier the way a browser cannot
// ---------------------------------------------------------------------------
/**
 * The ESM entry for `pkg` (and optional subpath), from its `exports` map, taking
 * the `import`/`module` condition — never `require`/`default`, which point at
 * .cjs for typed-binary and tinyest and would be a syntax error in a browser.
 */
function esmEntry(pkg, subpath) {
  const pj = JSON.parse(readFileSync(join(NM, pkg, "package.json"), "utf8"));
  const pick = (node) => {
    if (typeof node === "string") return node;
    if (!node || typeof node !== "object") return null;
    for (const cond of ["import", "module", "default"]) {
      const v = pick(node[cond]);
      if (v && v.endsWith(".js")) return v;
    }
    return null;
  };
  const exp = pj.exports;
  const key = subpath ? `./${subpath}` : ".";
  let target = exp ? pick(exp[key]) : null;
  if (!target && exp?.["./*"]) target = pick(exp["./*"])?.replace("*", subpath);
  if (!target && !subpath) target = pj.module ?? pj.main;
  if (!target) die(`cannot resolve "${pkg}${subpath ? "/" + subpath : ""}" to an ESM entry`);
  return join(pkg, target.replace(/^\.\//, ""));
}

// ---------------------------------------------------------------------------
// Scanning
// ---------------------------------------------------------------------------
const allJs = (dir, out = []) => {
  for (const e of readdirSync(dir)) {
    const p = join(dir, e);
    if (statSync(p).isDirectory()) allJs(p, out);
    else if (p.endsWith(".js")) out.push(p);
  }
  return out;
};

/**
 * Import/export statements carrying a non-relative specifier.
 *
 * Anchored to the start of a line so that JSDoc examples (`* import { common }
 * from 'typegpu';`) and WGSL built inside template literals (`from '${name}'`)
 * are not mistaken for imports. Both quote styles, because that is exactly the
 * bug this file exists to not repeat.
 */
const BARE = /^[ \t]*(?:import|export)\b[^\n]*?from\s*(['"])([^.'"][^'"]*)\1|^[ \t]*import\s*(['"])([^.'"][^'"]*)\3/gm;
function bareSpecs(src) {
  const out = [];
  for (const m of src.matchAll(BARE)) out.push({ text: m[0], spec: m[2] ?? m[4], quote: m[1] ?? m[3] });
  return out;
}

const HOST_ONLY = [
  [/\bprocess\s*\./, "process.*"],
  [/\brequire\s*\(/, "require()"],
  [/from\s*['"]node:/, "node: import"],
  [/\b__dirname\b|\b__filename\b/, "__dirname/__filename"],
];

// ---------------------------------------------------------------------------
// Copy + rewrite
// ---------------------------------------------------------------------------
if (CHECK_ONLY && !existsSync(join(VENDOR, ROOT_PKG))) {
  console.log("  typegpu not vendored — run `npm run vendor`. Skipping.");
  process.exit(0);
}

if (!CHECK_ONLY) {
  rmSync(VENDOR, { recursive: true, force: true });
  for (const p of PKGS) cpSync(join(NM, p), join(VENDOR, p), { recursive: true });

  // --- positive control: the scanner must find the problems it will later
  //     certify absent. If this is 0 the scanner is broken, not the tree.
  const files = allJs(VENDOR);
  const before = files.flatMap((f) => bareSpecs(readFileSync(f, "utf8")).map((b) => b.spec));
  if (!before.length) {
    die("the bare-specifier scanner found nothing in an UNMODIFIED tree.\n" +
        "That is not good news — typegpu imports tsover-runtime, typed-binary and\n" +
        "tinyest by bare specifier. The scanner is broken; fix it before trusting it.");
  }
  const envBefore = readFileSync(join(VENDOR, ROOT_PKG, "shared/env.js"), "utf8");
  if (!/process\.env\.NODE_ENV/.test(envBefore)) {
    die("shared/env.js no longer reads process.env.NODE_ENV. TypeGPU changed it —\n" +
        "re-read the file and decide whether the substitution below is still needed.");
  }

  // --- rewrite bare specifiers to relative paths
  let rewrites = 0;
  for (const f of files) {
    const src = readFileSync(f, "utf8");
    const specs = bareSpecs(src);
    if (!specs.length) continue;
    let out = src;
    for (const { text, spec, quote } of specs) {
      const [pkg, ...rest] = spec.startsWith("@") ? [spec.split("/").slice(0, 2).join("/"), ...spec.split("/").slice(2)]
                                                  : spec.split("/");
      if (!PKGS.includes(pkg)) {
        die(`${f}: imports "${spec}", which is not a dependency of ${ROOT_PKG}.\n` +
            `Vendoring cannot resolve it. Add it to the copy list or find out why it is there.`);
      }
      const target = resolve(VENDOR, esmEntry(pkg, rest.join("/") || undefined));
      if (!existsSync(target)) die(`${f}: "${spec}" resolves to ${target}, which does not exist`);
      let rel = relative(dirname(f), target).split("\\").join("/");
      if (!rel.startsWith(".")) rel = "./" + rel;
      out = out.replace(text, text.replace(`${quote}${spec}${quote}`, `${quote}${rel}${quote}`));
      rewrites++;
    }
    writeFileSync(f, out);
  }

  // --- the substitution a bundler would make
  writeFileSync(join(VENDOR, ROOT_PKG, "shared/env.js"), envBefore
    .replace("export const DEV = process.env.NODE_ENV === 'development';",
             "export const DEV = false;   // vendor.mjs: bundler substitution, production branch")
    .replace("export const TEST = process.env.NODE_ENV === 'test';",
             "export const TEST = false;  // vendor.mjs: bundler substitution, production branch"));

  console.log(`   ${PKGS.length} packages copied (${PKGS.join(", ")}), ` +
              `${rewrites} bare specifiers rewritten, env.js substituted`);
}

// ---------------------------------------------------------------------------
// Verify — this half also runs under --check
// ---------------------------------------------------------------------------
const files = allJs(VENDOR);
const problems = [];

for (const f of files) {
  const src = readFileSync(f, "utf8");
  for (const { spec } of bareSpecs(src)) {
    problems.push(`${f}  bare specifier "${spec}" — a browser cannot resolve it`);
  }
  for (const [re, what] of HOST_ONLY) {
    for (const [i, line] of src.split("\n").entries()) {
      const t = line.trimStart();
      if (t.startsWith("*") || t.startsWith("//")) continue;
      if (re.test(line)) problems.push(`${f}:${i + 1}  ${what}  ${t.slice(0, 78)}`);
    }
  }
}
if (problems.length) die(`the vendored tree would not load in a browser:\n${problems.join("\n")}`);

// Every module the browser will fetch, starting where the harness starts.
const seen = new Set(), missing = [];
(function walk(f) {
  if (seen.has(f)) return;
  seen.add(f);
  if (!existsSync(f)) return void missing.push(f);
  const src = readFileSync(f, "utf8"), dir = dirname(f);
  for (const m of [
    ...src.matchAll(/(?:^|\n)\s*(?:import|export)[^;\n]*?from\s*["'](\.[^"']+)["']/g),
    ...src.matchAll(/import\s*\(\s*["'](\.[^"']+)["']\s*\)/g),
    ...src.matchAll(/(?:^|\n)\s*import\s*["'](\.[^"']+)["']/g),
  ]) walk(resolve(dir, m[1]));
})(resolve("typegpu-runner.js"));

if (missing.length) die(`incomplete module graph — the browser would 404 on:\n${missing.join("\n")}`);

console.log(CHECK_ONLY
  ? `vendored typegpu is browser-loadable: ${files.length} files scanned, ` +
    `${seen.size} modules reachable, no bare specifiers, no host-only references`
  : `-> ${files.length} files, ${seen.size} modules reachable from the adapter, all specifiers relative`);
