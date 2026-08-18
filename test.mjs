#!/usr/bin/env node
/**
 * test.mjs — the walking skeleton's regression suite.
 *
 *   node test.mjs
 *
 * Three things are checked, in the order they can fail:
 *
 *   1. The canonical kernel compiles, and its MLIR is BYTE-IDENTICAL to
 *      spike/mlir-spirv/l3/matmul.mlir — the artifact that was verified end to
 *      end (spirv-val, naga, and bit-exact against the CPU oracle in Chrome).
 *      Any drift in the emitter shows up here as a diff rather than as a wrong
 *      number three stages later.
 *
 *   2. Every kernel in examples/negative/ is REJECTED, with the expected rule
 *      number. A default-reject admission gate is only as good as its coverage,
 *      and an earlier version of it silently compiled a body containing
 *      `while (false) {}` — so these are not decoration.
 *
 *   3. The compiler's own source type-checks under the pinned tsc.
 */

import { execFileSync } from "node:child_process";
import { readFileSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const out = mkdtempSync(join(tmpdir(), "tessera-test-"));
let failures = 0;

const ok = (m) => console.log(`  \x1b[32m✓\x1b[0m ${m}`);
const bad = (m) => { console.log(`  \x1b[31m✗\x1b[0m ${m}`); failures++; };

function build(entry) {
  try {
    return { ok: true, out: execFileSync("npx", ["tsx", "src/cli.ts", "build", entry, "-o", out],
      { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }) };
  } catch (e) {
    return { ok: false, out: `${e.stdout ?? ""}${e.stderr ?? ""}` };
  }
}

// ---- 1. the canonical kernel, against the verified reference ---------------
console.log("\ncanonical kernel");
const built = build("examples/matmul.kernel.ts");
if (!built.ok) {
  bad(`examples/matmul.kernel.ts failed to build\n${built.out}`);
} else {
  ok("builds: tsc -> IR -> MLIR -> spirv-val -> naga");

  // The reference used @matmul; tessera uses the spec's own name. That one
  // label is normalised; everything else must match exactly.
  const norm = (s) => s.replace(/@matmul(_relu_f32)?\(/, "@K(");
  const ref = norm(readFileSync("spike/mlir-spirv/l3/matmul.mlir", "utf8"));
  const got = norm(readFileSync(join(out, "matmul_relu_f32.mlir"), "utf8"));

  // The WGSL entry point is not necessarily spec.name — naga renames anything
  // ending in a reserved word. A host that trusts spec.name fails at pipeline
  // creation with "Entry point ... doesn't exist", which is what happened once.
  const man = JSON.parse(readFileSync(join(out, "matmul_relu_f32.json"), "utf8"));
  const wgsl = readFileSync(join(out, "matmul_relu_f32.wgsl"), "utf8");
  const declared = wgsl.match(/@compute[^\n]*\n\s*fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/)?.[1];
  if (declared && man.entryPoint === declared) {
    ok(`manifest entryPoint "${man.entryPoint}" matches the WGSL` +
       (man.entryPoint !== man.name ? ` (naga renamed it from "${man.name}")` : ""));
  } else {
    bad(`manifest says entryPoint "${man.entryPoint}" but the WGSL declares "${declared}"`);
  }

  if (ref === got) {
    ok(`MLIR byte-identical to the verified reference (${got.split("\n").length} lines)`);
  } else {
    const r = ref.split("\n"), g = got.split("\n");
    const i = r.findIndex((l, n) => l !== g[n]);
    bad(`MLIR drifted from spike/mlir-spirv/l3/matmul.mlir at line ${i + 1}\n` +
        `      reference: ${r[i]}\n      tessera:   ${g[i]}`);
  }
}

// ---- 2. every negative must be rejected, by the right rule -----------------
console.log("\nadmission gate");
const NEGATIVES = [
  ["ragged.kernel.ts",    "TSA0301", "extent the block does not divide"],
  ["body.kernel.ts",      "TSA0104", "a statement form the emitter does not read"],
  ["closure.kernel.ts",   "TSA0104", "an expression form outside the subset"],
  // Caught by GATE 1 — the surface types reject it before the front end ever
  // runs. That is the better outcome, and worth asserting as such.
  ["gridblock.kernel.ts", "does not type-check", "grid axis blocked at 32 while tile.bm is 64"],
  // Ragged. All three are caught by GATE 1 — the surface types reject them
  // before the front end runs, with the diagnostics the design specified.
  ["nopad.kernel.ts",   "this block is ragged", "ragged load with no identity named"],
  ["badpad.kernel.ts",  "does not type-check",  "a non-annihilating pad reaching a reduction"],
  ["deadpad.kernel.ts", "does not type-check",  ".pad() on an axis that is not ragged"],
];

for (const [file, rule, why] of NEGATIVES) {
  const r = build(`examples/negative/${file}`);
  if (r.ok) bad(`${file} COMPILED — it must be rejected (${why})`);
  else if (!r.out.includes(rule)) {
    bad(`${file} rejected, but not with ${rule}:\n      ${r.out.trim().split("\n").pop()}`);
  } else ok(`${file} → ${rule}  (${why})`);
}

// ---- 2b. the ragged kernel builds, and only it has masks -------------------
console.log("\nragged axes");
const rag = build("examples/matmul-ragged.kernel.ts");
if (!rag.ok) {
  bad(`examples/matmul-ragged.kernel.ts failed to build\n${rag.out}`);
} else {
  ok("builds: 1000 x 750 x 500, no axis divides its block");
  const rw = readFileSync(join(out, "matmul_relu_ragged.wgsl"), "utf8");
  const aw = readFileSync(join(out, "matmul_relu_f32.wgsl"), "utf8");
  const bounds = (t) => (t.match(/< \d+u\)/g) ?? []).length;
  // The claim under test is not "masks exist" but "masks exist EXACTLY where
  // raggedness is". An aligned kernel that grew a bound check would mean the
  // emitter is hedging, which is the cost the whole static design pays to avoid.
  if (bounds(rw) > 0 && bounds(aw) === 0) {
    ok(`${bounds(rw)} synthesised bound checks in the ragged kernel, 0 in the aligned one`);
  } else {
    bad(`ragged has ${bounds(rw)} bound checks, aligned has ${bounds(aw)} — expected >0 and 0`);
  }
  const rm = JSON.parse(readFileSync(join(out, "matmul_relu_ragged.json"), "utf8"));
  const expected = ["a:m", "a:k", "b:k", "b:n", "c:m", "c:n"];
  if (JSON.stringify(rm.maskedLoads) === JSON.stringify(expected) && rm.pad === 0) {
    ok(`manifest reports the masked pairs: ${rm.maskedLoads.join(" ")}`);
  } else {
    bad(`manifest maskedLoads = ${JSON.stringify(rm.maskedLoads)}, pad = ${rm.pad}`);
  }
}

// ---- 3. our own source ----------------------------------------------------
console.log("\ncompiler source");
try {
  execFileSync("./node_modules/.bin/tsc", ["--noEmit", "-p", "tsconfig.json"], { stdio: "pipe" });
  ok("type-checks under the pinned tsc");
} catch (e) {
  bad(`tsc:\n${e.stdout?.toString() ?? e.message}`);
}

rmSync(out, { recursive: true, force: true });
console.log(failures ? `\n${failures} failure(s)\n` : "\nall green\n");
process.exit(failures ? 1 : 0);
