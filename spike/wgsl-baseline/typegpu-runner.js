// typegpu-runner.js — the same kernel, with TypeGPU owning the resources.
//
// WHY. tessera has no host-side story of its own. run.js and measure.js
// hand-roll buffer creation, bind groups, dispatch and readback, and the
// bind-group wiring in particular is untyped index-matching that has already
// gone wrong twice in this project's history: a hardcoded entry point, and a
// recomputed dispatch that silently skipped every ragged tail block. TypeGPU is
// exactly that missing layer and is better at it than anything here would be, so
// the question worth answering is not "should tessera compete with it" but
// "can tessera sit on it, and what does that cost".
//
// WHAT FITS. Two thirds of it. What does NOT fit is pipeline creation:
// `TgpuComputePipeline`'s descriptor is `{ compute: TgpuComputeFn<Input> }` and
// nothing else, and a TgpuComputeFn only exists if the body went through
// TypeGPU's own TS→WGSL path. Buffers have a wrap-an-existing-GPUBuffer
// overload; pipelines have no equivalent. This is structural, not a gap to work
// around — TypeGPU's value is that it knows what a resource contains, and it
// cannot know that about WGSL somebody else emitted.
//
// So the seam is:
//
//   TypeGPU   bind group layout, typed buffers, bind group, upload, readback
//   raw       createShaderModule, createComputePipeline, dispatchWorkgroups
//
// and it is a clean seam only because `root.unwrap(layout)` yields a real
// GPUBindGroupLayout. That lets the pipeline be built with an EXPLICIT layout
// instead of `layout: "auto"`, which is what makes the two halves agree about
// binding indices at all; with "auto" the two would each invent their own and
// the bind group would be rejected.
//
// The dependency stays optional on purpose. A compiler that hard-requires a
// pre-1.0 runtime is worse than one that does not, so this is a peer of the raw
// runner rather than a replacement, and the harness runs both against the same
// oracle. `once()` is not reimplemented here — it comes from measure.js's
// createDispatcher, so the timed path is the same code and any difference in
// the numbers is the resource layer and nothing else.

import tgpu from "./vendor/typegpu/index.js";
import * as d from "./vendor/typegpu/data/index.js";
import { createDispatcher } from "./measure.js";

export async function createTypeGPURunner(ctx, wgsl, manifest, inputs, label, { batch = 1 } = {}) {
  const { device } = ctx;
  const { dispatch, bindings } = manifest;
  const root = tgpu.initFromDevice({ device });

  // ---- layout, straight from the manifest -----------------------------------
  // TypeGPU assigns binding indices in property order, which is the same order
  // tessera numbers them in, so the two agree without either being told.
  const layout = tgpu.bindGroupLayout(Object.fromEntries(bindings.map((bd) => [bd.name, {
    storage: d.arrayOf(d.f32, bd.elements),
    access: bd.mode === "read" ? "readonly" : "mutable",
    visibility: ["compute"],
  }])));

  // ---- typed buffers --------------------------------------------------------
  const bufs = Object.fromEntries(bindings.map((bd) =>
    [bd.name, root.createBuffer(d.arrayOf(d.f32, bd.elements)).$usage("storage")]));

  bindings.filter((bd) => bd.mode === "read").forEach((bd, k) => {
    const src = inputs[k];
    if (!src || src.length !== bd.elements) {
      throw new Error(`${label}: input ${k} for binding "${bd.name}" should have ` +
                      `${bd.elements} elements, got ${src ? src.length : "nothing"}`);
    }
    // The ArrayBuffer overload, not the number[] one. A Float32Array of 786432
    // elements has no business becoming a JS array on the way to the GPU.
    bufs[bd.name].write(src.buffer);
  });

  const outName = bindings.find((bd) => bd.mode === "write")?.name;
  if (!outName) throw new Error(`${label}: no output binding in the manifest`);

  // ---- the part TypeGPU cannot do -------------------------------------------
  const module = device.createShaderModule({ code: wgsl, label });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter((m) => m.type === "error");
  if (errs.length) {
    throw new Error(`${label}: WGSL rejected:\n` +
      errs.map((m) => `  ${m.lineNum}:${m.linePos}  ${m.message}`).join("\n"));
  }
  const pipeline = await device.createComputePipelineAsync({
    layout: device.createPipelineLayout({ bindGroupLayouts: [root.unwrap(layout)] }),
    compute: { module, entryPoint: manifest.entryPoint },
    label,
  });

  const { once, destroy: destroyTimer } = createDispatcher(ctx, {
    pipeline, bindGroup: root.unwrap(root.createBindGroup(layout, bufs)), dispatch, label, batch,
  });

  /**
   * Readback, with TypeGPU owning the staging buffer and the mapping — the one
   * place where it removes code outright rather than just typing it. It hands
   * back `number[]` because the schema says array<f32>, so the Float32Array the
   * oracle wants is rebuilt here; that is the cost of going through a typed
   * layer, and it is off the timed path.
   */
  async function result() {
    return Float32Array.from(await bufs[outName].read());
  }

  function destroy() {
    destroyTimer();
    root.destroy();          // owns every buffer it created
  }

  return { once, result, destroy, dispatch, label, batch };
}
