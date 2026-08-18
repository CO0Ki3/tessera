#!/usr/bin/env node
/**
 * check-typegpu-layout.mjs — the one thing the TypeGPU adapter can get wrong
 * without a GPU noticing.
 *
 *   node check-typegpu-layout.mjs
 *
 * typegpu-runner.js builds a bind group layout from tessera's manifest, while
 * the pipeline is built from tessera's WGSL. Nothing at runtime forces those two
 * descriptions to agree about which buffer is binding 0, whether it is readonly,
 * or what it contains — WebGPU validates the bind group against the LAYOUT, and
 * the layout is the thing being derived. Get the order wrong and every kernel
 * still runs, with a and b swapped.
 *
 * That agreement is checkable on the CPU, so it is checked here rather than
 * being left to a person reading two files side by side. This is also the honest
 * cost of the adapter: TypeGPU removes the untyped bind-group wiring from the
 * host, and moves the remaining risk into the manifest-to-layout derivation,
 * which is at least small enough to test.
 *
 * Skips cleanly if `npm run vendor` has not been run.
 */

import { readFileSync, existsSync } from "node:fs";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

process.chdir(dirname(fileURLToPath(import.meta.url)));

if (!existsSync("vendor/typegpu/index.js")) {
  console.log("  typegpu not vendored — run `npm run vendor`. Skipping.");
  process.exit(0);
}
const tgpu = (await import("./vendor/typegpu/index.js")).default;
const d = await import("./vendor/typegpu/data/index.js");

/** Every @group/@binding a generated kernel declares. */
function declared(src) {
  return [...src.matchAll(
    /@group\((\d+)\)\s*@binding\((\d+)\)\s*var<storage,\s*(read|read_write)>\s*(\w+)\s*:\s*array<(\w+)>/g)]
    .map(([, g, b, mode, name, elem]) => ({ group: +g, binding: +b, mode, name, elem }));
}

const CASES = [
  ["matmul-direct.json", "matmul-direct.wgsl"],
  ["matmul-ragged-direct.json", "matmul-ragged-direct.wgsl"],
  ["softmax.json", "softmax.wgsl"],
  ["layernorm.json", "layernorm.wgsl"],
];

let failures = 0, checked = 0;
for (const [manFile, wgslFile] of CASES) {
  if (!existsSync(manFile) || !existsSync(wgslFile)) {
    console.log(`  ${manFile.padEnd(28)} skipped (not built)`);
    continue;
  }
  const man = JSON.parse(readFileSync(manFile, "utf8"));
  const decls = declared(readFileSync(wgslFile, "utf8"));

  // Exactly what typegpu-runner.js constructs, from exactly the same input.
  const layout = tgpu.bindGroupLayout(Object.fromEntries(man.bindings.map((b) => [b.name, {
    storage: d.arrayOf(d.f32, b.elements),
    access: b.mode === "read" ? "readonly" : "mutable",
    visibility: ["compute"],
  }])));
  const order = Object.keys(layout.entries);        // TypeGPU numbers by property order

  const bad = [];
  if (decls.length !== man.bindings.length) {
    bad.push(`WGSL declares ${decls.length} storage bindings, the manifest has ${man.bindings.length}`);
  }
  man.bindings.forEach((bd, i) => {
    const dcl = decls.find((x) => x.binding === i);
    if (!dcl) { bad.push(`nothing is @binding(${i}) in the WGSL`); return; }
    if (dcl.group !== 0) bad.push(`${bd.name}: WGSL says @group(${dcl.group}), the adapter binds group 0`);
    if (dcl.elem !== "f32") bad.push(`${bd.name}: WGSL element type is ${dcl.elem}, the adapter says f32`);
    const want = bd.mode === "read" ? "read" : "read_write";
    if (dcl.mode !== want) bad.push(`${bd.name}: WGSL is <storage, ${dcl.mode}>, the manifest says "${bd.mode}"`);
    if (order[i] !== dcl.name) {
      bad.push(`binding ${i}: TypeGPU would put "${order[i]}" there, the WGSL declares "${dcl.name}"`);
    }
  });

  checked++;
  failures += bad.length;
  console.log(`  ${manFile.padEnd(28)} ${String(man.bindings.length).padStart(2)} bindings  ` +
    (bad.length ? `FAIL\n      ${bad.join("\n      ")}` : "agree"));
}

if (!checked) { console.log("  nothing built — run `npm run demo` first."); process.exit(0); }
console.log(failures
  ? `\n  ${failures} disagreement(s): the adapter would bind the wrong buffers.`
  : `\n  manifest, emitted WGSL and TypeGPU layout agree on all ${checked}.`);
process.exit(failures ? 1 : 0);
