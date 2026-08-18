#!/usr/bin/env node
/**
 * make-probes.mjs — isolate one variable at a time in the hand-vs-tessera gap.
 *
 *   node make-probes.mjs
 *
 * The paired measurement says tessera's kernel runs ~1.5x slower than the
 * hand-written one. Two structural differences are visible in the WGSL:
 *
 *   A. storage access mode. The hand-written kernel declares its two input
 *      buffers `var<storage, read>`; naga emits `read_write` for all three,
 *      because nothing in the SPIR-V says the inputs are read-only. On Metal
 *      read-only storage lowers to `const device`, which can use the read-only
 *      data cache and permits stronger aliasing assumptions.
 *
 *   B. control flow. naga emits four unstructured `loop {} continuing {}` blocks
 *      carrying 144 `phi_` variables, where the hand-written kernel has ten
 *      ordinary `for` loops. That is more pressure on whatever Tint and the
 *      Metal compiler do with live ranges.
 *
 * Only A is cheap to isolate, and it can be probed from BOTH sides — make the
 * known-good kernel worse, and make the generated kernel better. If the gap is
 * A, the two probes move toward each other and meet.
 *
 * Two-sided is what makes this conclusive. A one-sided change that happens to
 * help could be helping for some unrelated reason; a change that both slows the
 * fast kernel and speeds the slow one by the same amount is the variable.
 */

import { readFileSync, writeFileSync } from "node:fs";

// ---- probe A1: hand-written, but with the inputs made read_write ------------
const hand = readFileSync("matmul.wgsl", "utf8");
const handRW = hand
  .replace(
    "@group(0) @binding(0) var<storage, read>       a : array<f32>;",
    "@group(0) @binding(0) var<storage, read_write> a : array<f32>;")
  .replace(
    "@group(0) @binding(1) var<storage, read>       b : array<f32>;",
    "@group(0) @binding(1) var<storage, read_write> b : array<f32>;");

if (handRW === hand) throw new Error("probe A1: no substitution made — did matmul.wgsl change?");
writeFileSync("probe-hand-rw.wgsl", handRW);
console.log("  probe-hand-rw.wgsl      hand-written, inputs forced to read_write");

// ---- probe A2: tessera output, but with the inputs marked read --------------
// Legitimate: nothing writes to arg_0_ or arg_1_. This is what the SPIR-V would
// say if the NonWritable decoration were emitted from the binding's `read` mode,
// which tessera already knows and currently drops on the floor.
const mlir = readFileSync("matmul-mlir.wgsl", "utf8");
let mlirRO = mlir;
for (const arg of ["arg_0_", "arg_1_"]) {
  const re = new RegExp(`var<storage, read_write> ([A-Za-z0-9_]*${arg}):`);
  if (!re.test(mlirRO)) throw new Error(`probe A2: could not find ${arg}`);
  mlirRO = mlirRO.replace(re, "var<storage, read> $1:");
}
if (/var<storage, read>/.test(mlirRO) === false) throw new Error("probe A2: no substitution");
writeFileSync("probe-tessera-ro.wgsl", mlirRO);
console.log("  probe-tessera-ro.wgsl   tessera output, inputs marked read");

// ---- probe D: the direct WGSL pushed through naga and back -----------------
// MLIR is entirely out of this one. The round-trip acquires naga's
// loop{}/continuing{} form but NOT the 144 phi variables or the 4 bitcasts that
// only the MLIR path has, so it separates the two remaining candidates for the
// 1.57x: if this stays fast, the loop form is innocent.
//
// naga also INSERTS a workgroup zero-init that neither the direct nor the MLIR
// path pays:
//
//     if (local_invocation_index == 0u) { As = array<f32,1024>(); Bs = ...; }
//     workgroupBarrier();
//
// One invocation writing 2048 floats plus a barrier, once per workgroup. Leaving
// it in would mean the probe measures that instead of the loop form, so it is
// removed. Safe for this kernel specifically: the staging loops write every
// element of both arrays before any read (1024 / 256 invocations = exactly 4
// each), and the barrier guards nothing, since no shared write precedes it.
{
  const { execFileSync } = await import("node:child_process");
  const tmp = "probe-direct-roundtrip.spv";
  execFileSync("naga", ["matmul-direct.wgsl", tmp]);
  execFileSync("naga", [tmp, "probe-direct-roundtrip.wgsl"]);

  let rt = readFileSync("probe-direct-roundtrip.wgsl", "utf8");
  const zeroInit =
    /\n\s*if \(_e\d+ == 0u\) \{\n\s*global_\d+ = array<f32, \d+>\(\);\n\s*global_\d+ = array<f32, \d+>\(\);\n\s*\}\n\s*workgroupBarrier\(\);/;
  if (!zeroInit.test(rt)) {
    throw new Error("probe D: naga's workgroup zero-init was not found — has its output changed? " +
                    "Leaving it in would silently make this probe measure the wrong thing.");
  }
  rt = rt.replace(zeroInit, "");
  writeFileSync("probe-direct-roundtrip.wgsl", rt);

  const ep = rt.match(/@compute[^\n]*\n\s*fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/)[1];
  const man = JSON.parse(readFileSync("matmul-direct.json", "utf8"));
  man.entryPoint = ep;
  man.backend = "wgsl-roundtrip";
  writeFileSync("probe-direct-roundtrip.json", JSON.stringify(man, null, 2) + "\n");

  execFileSync("naga", ["probe-direct-roundtrip.wgsl"]);   // still valid?
  console.log(`  probe-direct-roundtrip.wgsl  direct WGSL through naga and back ` +
              `(entry "${ep}", zero-init removed)`);
}

console.log("\nreload the harness; the probes are timed alongside the originals.");
