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

console.log("\nreload the harness; both probes are timed alongside the originals.");
