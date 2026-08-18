// measure.js — the smallest honest GPU measurement.
//
// WHY THIS EXISTS. The first version of this harness timed each kernel with one
// dispatch, in a fixed order, with no warm-up. It produced, across three
// sessions, these numbers for the SAME unoptimised shader:
//
//     45q ... 22q          (2x apart)
//
// and this ordering, in one session:
//
//     hand 27q, tessera 27q, hand-read_write 10q, tessera-read 19q, noopt 22q
//
// which reads as "making the hand kernel worse made it 2.7x faster". It did not.
// The GPU ramps its clocks as work arrives, so position in the sequence moved
// the numbers more than the kernels did — and a conclusion of "tessera is 1.5x
// slower", drawn from two paired samples that happened to agree, was wrong.
//
// So: warm up, repeat, and INTERLEAVE. Interleaving is the part that matters.
// Running all of A then all of B lets drift masquerade as a difference between
// A and B; round-robin spreads any drift across every variant equally.
//
// Reported statistic is the MINIMUM. For a fixed amount of GPU work, noise is
// one-sided — contention, ramping and scheduling can only make a sample slower,
// never faster than the hardware can go. The median moves with how busy the
// machine was; the minimum converges on the kernel. Median and spread are
// reported alongside so a suspiciously wide distribution is visible rather than
// silently averaged away.

/** Chrome quantises timestamp-query results to this. Measured, not assumed. */
export const QUANTUM_NS = 65536;

/**
 * Allocate everything one kernel needs, once, so the timed loop does no
 * allocation and no readback. Buffers are shared across repetitions on purpose:
 * we are timing the kernel, not the allocator.
 */
/**
 * `dispatch` MUST come from the compiler's manifest, not be recomputed here.
 *
 * This function previously derived it as [N/64, M/64, 1]. That is neither
 * integer division nor a ceiling, so for 750x1000 it asked for 11.71875 x 15.625
 * workgroups and the last block in each dimension was never dispatched — every
 * tail element stayed zero, and the ragged kernel looked like it had a mask bug.
 * The masks were correct; the host ignored a manifest that said [12, 16, 1] and
 * recomputed it wrongly. Same mistake as hardcoding the entry point, twice over.
 *
 * The rule: anything the compiler already decided is read, never re-derived.
 */
export function createRunner(ctx, pipeline, inputs, manifest, label) {
  const { device, hasTimestamps } = ctx;
  const { dispatch, bindings } = manifest;
  if (!Array.isArray(dispatch) || dispatch.length !== 3 || !dispatch.every(Number.isInteger)) {
    throw new Error(`${label}: manifest.dispatch must be three integers, got ${JSON.stringify(dispatch)}`);
  }

  // One buffer per binding, sized and ordered by the manifest. Nothing here
  // knows how many operands a schedule has: matmul declares three bindings and
  // softmax two, and this reads whichever the compiler wrote down.
  const bufs = bindings.map((bd) => device.createBuffer({
    size: bd.elements * 4,
    usage: GPUBufferUsage.STORAGE
      | (bd.mode === "read" ? GPUBufferUsage.COPY_DST : GPUBufferUsage.COPY_SRC),
    label: `${label}:${bd.name}`,
  }));

  const reads = bindings.map((bd, i) => [bd, i]).filter(([bd]) => bd.mode === "read");
  reads.forEach(([bd, i], k) => {
    const src = inputs[k];
    if (!src || src.length !== bd.elements) {
      throw new Error(`${label}: input ${k} for binding "${bd.name}" should have ` +
                      `${bd.elements} elements, got ${src ? src.length : "nothing"}`);
    }
    device.queue.writeBuffer(bufs[i], 0, src);
  });

  const outIdx = bindings.findIndex((bd) => bd.mode === "write");
  if (outIdx < 0) throw new Error(`${label}: no output binding in the manifest`);
  const outBytes = bindings[outIdx].elements * 4;
  const readback = device.createBuffer({
    size: outBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: `${label}:rb`,
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bindings.map((bd, i) => ({ binding: i, resource: { buffer: bufs[i] } })),
  });

  let querySet = null, queryResolve = null, queryRead = null;
  if (hasTimestamps) {
    querySet = device.createQuerySet({ type: "timestamp", count: 2 });
    queryResolve = device.createBuffer({
      size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    queryRead = device.createBuffer({
      size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  /** One timed dispatch. Returns GPU nanoseconds, or null without timestamps. */
  async function once() {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass(
      hasTimestamps
        ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } }
        : {});
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(dispatch[0], dispatch[1], dispatch[2]);
    pass.end();
    if (hasTimestamps) {
      encoder.resolveQuerySet(querySet, 0, 2, queryResolve, 0);
      encoder.copyBufferToBuffer(queryResolve, 0, queryRead, 0, 16);
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    if (!hasTimestamps) return null;
    await queryRead.mapAsync(GPUMapMode.READ);
    const ts = new BigInt64Array(queryRead.getMappedRange().slice(0));
    queryRead.unmap();
    return Number(ts[1] - ts[0]);
  }

  /** Read the output once, for the correctness check. Not part of timing. */
  async function result() {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(bufs[outIdx], 0, readback, 0, outBytes);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    return out;
  }

  function destroy() {
    for (const b of [...bufs, readback, queryResolve, queryRead]) b?.destroy();
    querySet?.destroy();
  }

  return { once, result, destroy, dispatch, label };
}

/**
 * Warm up, then interleave `reps` rounds across every runner.
 * onProgress(done, total) is called so a multi-second measurement can say so.
 */
export async function measureInterleaved(runners, { warmup = 8, reps = 25, onProgress } = {}) {
  const samples = new Map(runners.map((r) => [r.label, []]));

  for (let w = 0; w < warmup; w++) {
    for (const r of runners) await r.once();
    onProgress?.(w + 1, warmup + reps);
  }

  for (let i = 0; i < reps; i++) {
    for (const r of runners) {
      const ns = await r.once();
      if (ns !== null) samples.get(r.label).push(ns);
    }
    onProgress?.(warmup + i + 1, warmup + reps);
  }

  const stats = new Map();
  for (const [label, ns] of samples) {
    if (!ns.length) { stats.set(label, null); continue; }
    const sorted = [...ns].sort((x, y) => x - y);
    const min = sorted[0];
    const median = sorted[(sorted.length / 2) | 0];
    const max = sorted[sorted.length - 1];
    stats.set(label, {
      n: sorted.length,
      minNs: min, medianNs: median, maxNs: max,
      minQ: min / QUANTUM_NS,
      medianQ: median / QUANTUM_NS,
      maxQ: max / QUANTUM_NS,
      // How much of the distribution is noise. A spread of many quanta means
      // the machine, not the kernel, is what varied.
      spreadQ: (max - min) / QUANTUM_NS,
      allQuantised: sorted.every((x) => x % QUANTUM_NS === 0),
    });
  }
  return stats;
}
