// run.js — WebGPU host code for the baseline matmul spike.
//
// Nothing here is clever. It is the minimum correct host program, written out
// longhand, because tessera will have to generate the equivalent and the point
// of the spike is to know exactly what "the equivalent" is.

const WORKGROUP = { x: 16, y: 16, z: 1 };

export async function initDevice() {
  if (!navigator.gpu) {
    throw new Error(
      "navigator.gpu is undefined — this browser has no WebGPU. " +
      "Chrome 113+, Edge, Safari 26+ or Firefox with WebGPU enabled.",
    );
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("requestAdapter() returned null — no compatible GPU adapter.");

  // timestamp-query is optional everywhere. Ask for it, run without it if absent.
  const wantTimestamps = adapter.features.has("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures: wantTimestamps ? ["timestamp-query"] : [],
  });

  device.lost.then((info) => {
    console.error("GPUDevice lost:", info.reason, info.message);
  });

  const info = adapter.info ?? {};
  return {
    device,
    adapter,
    hasTimestamps: wantTimestamps,
    // The limits that actually constrain the tile choice. Verifying these on
    // real hardware is half the point of the spike: the 8192 B of workgroup
    // storage this kernel uses is sized against the 16384 B guaranteed FLOOR,
    // and the invocation count sits exactly at the guaranteed 256.
    limits: {
      maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupSizeY: device.limits.maxComputeWorkgroupSizeY,
      maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
    },
    adapterInfo: {
      vendor: info.vendor ?? "(unknown)",
      architecture: info.architecture ?? "(unknown)",
      device: info.device ?? "(unknown)",
      description: info.description ?? "",
    },
  };
}

export async function compile(device, wgslSource, entryPoint = "main", label = "matmul_relu_f32") {
  const module = device.createShaderModule({ code: wgslSource, label });

  // Surface compilation diagnostics explicitly. A shader that fails to compile
  // still yields a pipeline, and the failure otherwise shows up much later as a
  // confusing validation error at dispatch time.
  const compilationInfo = await module.getCompilationInfo();
  const messages = compilationInfo.messages.map((m) => ({
    type: m.type,
    line: m.lineNum,
    col: m.linePos,
    message: m.message,
  }));
  if (messages.some((m) => m.type === "error")) {
    const detail = messages
      .filter((m) => m.type === "error")
      .map((m) => `  ${m.line}:${m.col}  ${m.message}`)
      .join("\n");
    throw new Error(`WGSL compilation failed:\n${detail}`);
  }

  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module, entryPoint },
    label,
  });

  return { module, pipeline, messages };
}

export async function runMatmul(ctx, pipeline, a, b, { M, N, K }) {
  const { device, hasTimestamps } = ctx;

  const bytesA = a.byteLength;
  const bytesB = b.byteLength;
  const bytesC = M * N * 4;

  const bufA = device.createBuffer({
    size: bytesA, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: "a",
  });
  const bufB = device.createBuffer({
    size: bytesB, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: "b",
  });
  const bufC = device.createBuffer({
    size: bytesC, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, label: "c",
  });
  const readback = device.createBuffer({
    size: bytesC, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: "readback",
  });

  device.queue.writeBuffer(bufA, 0, a);
  device.queue.writeBuffer(bufB, 0, b);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufA } },
      { binding: 1, resource: { buffer: bufB } },
      { binding: 2, resource: { buffer: bufC } },
    ],
  });

  // Timestamps, if available. Note that in-pass writeTimestamp was removed from
  // the spec for all implementations (gpuweb#2190), so pass-boundary
  // timestampWrites is the only path — there is no finer-grained option to miss.
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

  const dispatchX = N / 64;   // 12
  const dispatchY = M / 64;   // 16

  const encoder = device.createCommandEncoder({ label: "matmul" });
  const pass = encoder.beginComputePass({
    label: "matmul",
    ...(hasTimestamps
      ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } }
      : {}),
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(dispatchX, dispatchY, 1);
  pass.end();

  if (hasTimestamps) {
    encoder.resolveQuerySet(querySet, 0, 2, queryResolve, 0);
    encoder.copyBufferToBuffer(queryResolve, 0, queryRead, 0, 16);
  }
  encoder.copyBufferToBuffer(bufC, 0, readback, 0, bytesC);

  const wallStart = performance.now();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const wallMs = performance.now() - wallStart;

  await readback.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();

  let gpuMs = null, gpuNs = null;
  if (hasTimestamps) {
    await queryRead.mapAsync(GPUMapMode.READ);
    const ts = new BigInt64Array(queryRead.getMappedRange().slice(0));
    gpuNs = Number(ts[1] - ts[0]);
    gpuMs = gpuNs / 1e6;
    queryRead.unmap();
  }

  for (const buf of [bufA, bufB, bufC, readback, queryResolve, queryRead]) buf?.destroy();
  querySet?.destroy();

  const flops = 2 * M * N * K;

  // Chrome quantizes timestamp-query results unless developer features are
  // enabled. Measured on this machine, EVERY sample is an exact multiple of
  // 2^16 ns = 65.536 us, so that is the real resolution floor — differences
  // smaller than one quantum are invisible, and a comparison is only meaningful
  // when the two kernels are several quanta apart.
  const QUANTUM_NS = 65536;
  const quantised = gpuNs !== null && gpuNs % QUANTUM_NS === 0;

  return {
    out,
    wallMs,
    gpuMs,
    gpuNs,
    quantumNs: quantised ? QUANTUM_NS : null,
    quanta: quantised ? gpuNs / QUANTUM_NS : null,
    dispatch: [dispatchX, dispatchY, 1],
    workgroup: [WORKGROUP.x, WORKGROUP.y, WORKGROUP.z],
    workgroupBytes: (64 * 16 + 16 * 64) * 4,
    // wallMs includes submit + queue latency, so this is a floor on real cost,
    // not a kernel measurement. gpuMs is the honest number when present — and
    // Chrome quantizes timestamps to 100us unless developer features are on,
    // so a suspiciously round gpuMs means you are reading quantized data.
    gflopsWall: flops / (wallMs / 1000) / 1e9,
    gflopsGpu: gpuMs ? flops / (gpuMs / 1000) / 1e9 : null,
  };
}
