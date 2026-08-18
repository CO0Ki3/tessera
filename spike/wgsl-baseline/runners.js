// runners.js — one place that decides which resource layer a kernel runs on.
//
// Three pages (matmul, ragged, rowwise) each want the same thing: given WGSL, a
// manifest and some inputs, hand me something with once()/result()/destroy().
// Two of those pages want it twice, because tessera can now get its resources
// either by hand or from TypeGPU (see docs/005), and the only way to know the
// adapter is right is to run both and compare.
//
// The branch lives here rather than in each page. Three copies of
// `if (runtime === "typegpu")` is exactly the duplication this project spent a
// week removing from the emitter, and it would put the optional-dependency
// handling in three places too — which is where it would rot, since only one of
// them gets read after a TypeGPU upgrade.

import { compile } from "./run.js";
import { createRunner } from "./measure.js";

// ---------------------------------------------------------------------------
// The optional dependency, loaded once
// ---------------------------------------------------------------------------
// tessera must not hard-require a pre-1.0 runtime to run its own kernels, so the
// adapter is imported dynamically and its absence is a skipped row, not a broken
// page. The ERROR is kept: an earlier version reported every failure as
// "run `npm run vendor`", which was a lie the one time it mattered — TypeGPU had
// been vendored, and the real problem was that its published ESM reads
// `process.env` at module top level. A page that guesses at why it skipped
// something costs more than one that says it does not know.
let adapter = null, adapterError = null, attempted = false;

async function loadAdapter() {
  if (attempted) return adapter;
  attempted = true;
  try {
    ({ createTypeGPURunner: adapter } = await import("./typegpu-runner.js"));
  } catch (e) {
    adapterError = e;
    console.warn("typegpu adapter unavailable:", e);
  }
  return adapter;
}

/** `{ available, reason }` — reason is a real message, never a guess. */
export async function typegpuStatus() {
  await loadAdapter();
  return {
    available: !!adapter,
    reason: adapter ? null
      : adapterError ? `${adapterError.name}: ${adapterError.message}`
      : "adapter not loaded",
  };
}

// ---------------------------------------------------------------------------
// The one constructor
// ---------------------------------------------------------------------------
/**
 * @param {"raw"|"typegpu"} runtime  which resource layer owns the buffers
 * @returns a runner, or null if the TypeGPU adapter was asked for and is absent
 *
 * `manifest` is read, never re-derived — the dispatch and the binding list are
 * the compiler's decisions. Both host bugs this project has had came from a page
 * recomputing one of them.
 */
export async function makeRunner(ctx, { runtime = "raw", source, manifest, inputs, label, batch = 1 }) {
  if (runtime === "typegpu") {
    if (!await loadAdapter()) return null;
    // The adapter compiles and allocates together, because with TypeGPU the bind
    // group layout comes from the manifest rather than from the pipeline
    // (`getBindGroupLayout(0)`). That inversion is the point of it: the layout is
    // derived from what the compiler wrote down, not read back out of the shader.
    return { runner: await adapter(ctx, source, manifest, inputs, label, { batch }), messages: [] };
  }
  const { pipeline, messages } = await compile(ctx.device, source, manifest.entryPoint, label);
  return { runner: createRunner(ctx, pipeline, inputs, manifest, label, { batch }), messages };
}

/**
 * Fetch a WGSL file and its manifest together, or null if either is missing.
 * Every page was doing this by hand with a slightly different failure mode.
 */
export async function fetchKernel(wgslPath, manifestPath) {
  try {
    const [w, m] = await Promise.all([fetch(wgslPath), fetch(manifestPath)]);
    if (!w.ok || !m.ok) return null;
    return { source: await w.text(), manifest: await m.json() };
  } catch {
    return null;
  }
}
