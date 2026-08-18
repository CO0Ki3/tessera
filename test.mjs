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

function build(entry, ...flags) {
  try {
    return { ok: true, out: execFileSync("npx",
      ["tsx", "src/cli.ts", "build", entry, "-o", out, ...flags],
      { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }) };
  } catch (e) {
    return { ok: false, out: `${e.stdout ?? ""}${e.stderr ?? ""}` };
  }
}

// ---- 1. the canonical kernel, against the verified reference ---------------
console.log("\ncanonical kernel");
const built = build("examples/matmul.kernel.ts", "--backend=mlir");
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
const rag = build("examples/matmul-ragged.kernel.ts", "--backend=mlir");
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
  // The dispatch must CEIL, or the tail blocks are never launched. A host that
  // recomputed this as N/64 asked for 11.71875 workgroups and silently dropped
  // the last block in each dimension — the ragged kernel then looked like it had
  // a mask bug when the masks were correct.
  const want = [Math.ceil(750 / 64), Math.ceil(1000 / 64), 1];
  if (JSON.stringify(rm.dispatch) === JSON.stringify(want)) {
    ok(`dispatch ceils to [${want.join(", ")}] — tail blocks are launched`);
  } else {
    bad(`dispatch is ${JSON.stringify(rm.dispatch)}, expected ${JSON.stringify(want)}`);
  }
  const expected = ["a:m", "a:k", "b:k", "b:n", "c:m", "c:n"];
  if (JSON.stringify(rm.maskedLoads) === JSON.stringify(expected) && rm.pad === "zero") {
    ok(`manifest reports the masked pairs: ${rm.maskedLoads.join(" ")}`);
  } else {
    bad(`manifest maskedLoads = ${JSON.stringify(rm.maskedLoads)}, pad = ${rm.pad}`);
  }
}

// ---- 2c. the direct WGSL backend -------------------------------------------
// Same IR, no MLIR. Checked here for the two properties that make the A/B in
// docs/002 §5 meaningful: it must be valid WGSL, and it must make the same
// masking decisions as the MLIR backend — otherwise the measurement compares
// two different kernels rather than two emission paths.
console.log("\ndirect WGSL backend (the default)");
for (const [entry, label, wantMasks] of [
  ["examples/matmul.kernel.ts", "aligned", false],
  ["examples/matmul-ragged.kernel.ts", "ragged", true],
  ["examples/softmax.kernel.ts", "softmax", true],
  ["examples/layernorm.kernel.ts", "layernorm", true],
]) {
  let r;
  try {
    r = execFileSync("npx", ["tsx", "src/cli.ts", "build", entry, "-o", out, "--backend=wgsl"],
      { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
  } catch (e) { bad(`${label}: ${(e.stdout ?? "") + (e.stderr ?? "")}`); continue; }

  const name = label === "ragged" ? "matmul_relu_ragged"
             : label === "softmax" ? "softmax_f32"
             : label === "layernorm" ? "layernorm_f32" : "matmul_relu_f32";
  const wgsl = readFileSync(join(out, `${name}.wgsl`), "utf8");
  const bounds = (wgsl.match(/< \d+u\)/g) ?? []).length;

  try {
    execFileSync("naga", [join(out, `${name}.wgsl`)], { stdio: "pipe" });
    ok(`${label}: valid WGSL (${wgsl.split("\n").length} lines, ${bounds} bound checks)`);
  } catch (e) {
    bad(`${label}: naga rejected the direct output\n${e.stdout ?? e.message}`);
  }
  // naga is not sufficient on its own: it accepted 57 occurrences of a float
  // literal larger than f32::MAX and reported success, and the shader then failed
  // to compile in Chrome. Tint is stricter, so this checks what naga does not.
  // Range, not exactness: `0.00001` is not exactly representable in f32 and is
  // perfectly legal. What Tint rejects is a magnitude beyond f32's.
  const F32_MAX = 3.4028234663852886e38;
  const outOfRange = [...wgsl.matchAll(/-?\d+\.\d+(?:[eE][-+]?\d+)?/g)]
    .map((m) => m[0])
    .filter((t) => { const v = Number(t); return Number.isFinite(v) && Math.abs(v) > F32_MAX; });
  if (outOfRange.length === 0) ok(`${label}: every float literal is within f32's range`);
  else bad(`${label}: ${outOfRange.length} literal(s) Tint will reject: ${[...new Set(outOfRange)].slice(0,2).join(", ")}`);
  if (wantMasks ? bounds > 0 : bounds === 0) {
    ok(`${label}: masks exactly where raggedness is`);
  } else {
    bad(`${label}: ${bounds} bound checks, expected ${wantMasks ? ">0" : "0"}`);
  }
  // The entry point survives untouched: no naga in this path, so nothing renames it.
  const man = JSON.parse(readFileSync(join(out, `${name}.json`), "utf8"));
  if (man.backend === "wgsl" && man.entryPoint === man.name) {
    ok(`${label}: entry point is spec.name — nothing in this path renames it`);
  } else {
    bad(`${label}: manifest backend=${man.backend} entryPoint=${man.entryPoint} name=${man.name}`);
  }
}

// ---- 2d. the softmax emitter contains no masking of its own ----------------
// The claim under test in docs/004: a second schedule reuses the access layer
// rather than re-deriving where masks go. Measured on the source, not asserted.
console.log("\none emitter");
{
  const src = readFileSync("src/emit-wgsl.ts", "utf8");
  // There is one emitter now, so the claim is sharper than "the second schedule
  // adds no masking": the masking primitives must appear ONLY in the access layer.
  // select( and min( each occur once, both inside emitLoad.
  // Count the MASKING select specifically. `select` is also how relu is emitted,
  // and an earlier version of this check counted that too -- the same false
  // positive as counting loop bounds as masks. The masking one substitutes the
  // pad, so it is the one that mentions PAD.
  const sel = src.split("select(${PAD}").length - 1;
  const min = src.split("[min(").length - 1;
  if (sel === 1 && min === 1) {
    ok(`the pad is substituted from exactly one place, and the clamp from one`);
  } else {
    bad(`masking is emitted from ${sel} pad-select and ${min} clamp sites; expected 1 each`);
  }

  // And no schedule branches on a kernel family.
  const families = ["softmax", "layernorm", "matmul"].filter((f) => {
    const re = new RegExp(`(===|!==)\\s*["'\`]${f}`, "i");
    return re.test(src);
  });
  if (families.length === 0) ok("no branch on a kernel family name");
  else bad(`emit-wgsl.ts branches on: ${families.join(", ")}`);
}

// ---- 2e. the harness artifacts are what they claim to be -------------------
// The default backend flipped to direct, and for a while `npm run demo` built the
// "via MLIR" artifacts without --backend=mlir. The pages then compared the direct
// backend to itself under two labels and nobody noticed until a stale probe asked
// for an entry point that no longer existed.
console.log("\nharness artifacts");
{
  const H = "spike/wgsl-baseline";
  const pairs = [["matmul-mlir", "matmul-direct"], ["matmul-ragged", "matmul-ragged-direct"]];
  for (const [a, b] of pairs) {
    try {
      const wa = readFileSync(join(H, `${a}.wgsl`), "utf8");
      const wb = readFileSync(join(H, `${b}.wgsl`), "utf8");
      if (wa === wb) bad(`${a}.wgsl and ${b}.wgsl are identical — one backend built twice`);
      else ok(`${a} and ${b} are genuinely different backends ` +
              `(${wa.split("\n").length} vs ${wb.split("\n").length} lines)`);
    } catch { bad(`${a}/${b}: missing — run npm run demo`); }
  }
  // Every probe's manifest entry point must exist in the shader it points at.
  for (const [wgsl, man] of [["probe-tessera-ro", "matmul-mlir"],
                             ["probe-direct-roundtrip", "probe-direct-roundtrip"]]) {
    try {
      const src = readFileSync(join(H, `${wgsl}.wgsl`), "utf8");
      const ep = JSON.parse(readFileSync(join(H, `${man}.json`), "utf8")).entryPoint;
      if (src.includes(`fn ${ep}(`)) ok(`${wgsl} declares the entry point ${man}.json names`);
      else bad(`${wgsl}.wgsl has no fn ${ep} — the probe is stale, regenerate it`);
    } catch { bad(`${wgsl}: missing — run npm run demo`); }
  }
}

// ---- 3. the TypeGPU adapter's derivation ----------------------------------
// The adapter builds its bind group layout from the manifest and its pipeline
// from the WGSL, and WebGPU validates against the layout — so a disagreement
// between the two binds the wrong buffers and still runs. Checkable on the CPU,
// so checked here. Skips itself if typegpu was never vendored.
console.log("\ntypegpu adapter");
try {
  const out = execFileSync("node", ["spike/wgsl-baseline/vendor.mjs", "--check"], { stdio: "pipe" })
    .toString().trim();
  if (/not vendored/.test(out)) console.log(`  - ${out.trim()}`);
  else ok(out);
} catch (e) {
  bad(`the vendored typegpu would not load in a browser:\n${e.stdout?.toString() ?? e.message}`);
}
try {
  const out = execFileSync("node", ["spike/wgsl-baseline/check-typegpu-layout.mjs"], { stdio: "pipe" })
    .toString().trim();
  const last = out.split("\n").filter(Boolean).pop() ?? "";
  if (/not vendored|nothing built/.test(out)) console.log(`  - ${last.trim()}`);
  else ok(last.trim());
} catch (e) {
  bad(`manifest and emitted WGSL disagree:\n${e.stdout?.toString() ?? e.message}`);
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
