/**
 * emit-mlir.ts — tessera IR to MLIR text.
 *
 * A direct port of spike/mlir-spirv/gen-matmul-mlir.py, which was written by
 * hand and verified end to end: its output lowers through mlir-opt to SPIR-V,
 * passes spirv-val, is accepted by naga, and runs in Chrome bit-identical to the
 * hand-written WGSL. Porting it rather than rewriting it means the skeleton's
 * correctness test is a byte-for-byte diff against a known-good artifact.
 *
 * Two codegen rules discovered in that spike are load-bearing here:
 *
 *   1. Vectors must be the memref ELEMENT type, never a wide load over a scalar
 *      memref — the latter lowers to a spirv.Bitcast on a pointer, which the
 *      Logical addressing model forbids. This emitter is scalar throughout, so
 *      it does not trip the rule; a vectorising path must obey it.
 *   2. min/max must be cmpf+select, never arith.maximumf/maxnumf, whose IEEE
 *      NaN-propagation semantics emit OpIsNan and naga rejects it outright.
 *
 * Buffers are flat 1-D memrefs, matching how the WGSL sees `array<f32>`. That
 * sidesteps strided memrefs and subviews, where MemRefToSPIRV is least tolerant.
 */

import { PAD_LITERAL, type AxisIR, type KernelIR } from "./ir.ts";

const SC = "#spirv.storage_class<StorageBuffer>";
const WG = "#spirv.storage_class<Workgroup>";

export interface EmitOptions {
  /**
   * Fully unroll the reduction-block loop (trip count = tile.bk, a literal).
   *
   * This is a codegen decision, not an optimisation: the emitter already writes
   * out the TM*TN fragment as straight-line FMAs, and the bk loop is the same
   * kind of statically-known bound. It exists because of a measured difference —
   * the hand-written WGSL and tessera's both have a `for k in 0..bk` at source
   * level, but naga lowers tessera's to `loop {} continuing {}` with phi
   * variables, and the two are not equally optimisable downstream. Unrolling
   * here removes 16 loop-carried values and one loop level, so it tests whether
   * that structure is what costs the measured 1.57x.
   */
  unrollK?: boolean;
}

export function emitMLIR(k: KernelIR, opts: EmitOptions = {}): string {
  // ---- what this backend can actually express --------------------------------
  //
  // It emits ONE schedule: two parallel axes, one reduction, two read operands
  // staged into workgroup memory, a register fragment, optionally relu, one store.
  // That was every kernel when it was written. It is two of ten now, and until
  // this gate existed the other eight did not fail — they came out as a matmul.
  //
  // Measured, not assumed. fused-softmax emitted zero `math.exp`, zero
  // `arith.maximumf` and zero `arith.divf` against sixteen exps in the WGSL: the
  // softmax was dropped entirely and the toolchain reported success. matmul-bt's
  // MLIR differed from plain matmul's by the name and one stride constant, index
  // order unchanged — reading `b[k][n]` out of a buffer holding `b[n][k]`.
  //
  // Emitting something that is not what was written is the failure this project's
  // admission rule exists to prevent, and the rule applies to a backend too. This
  // runs before anything destructures the IR, because that is what it protects:
  // `const [gm, gn] = k.grid` crashes on a one-axis kernel, and a TypeError with
  // no context is worse than a refusal.
  {
    const say = (why: string): never => {
      throw new Error(
        `${k.name}: ${why}. The MLIR backend emits one schedule — two parallel ` +
        `axes, one reduction, two staged operands, a register fragment — and is ` +
        `kept as a second oracle for that shape (docs/002 §5). Build with the ` +
        `default (direct) backend.`);
    };
    if (k.multiPad) say("the body names more than one identity element");
    if (k.schedule !== "matmul") say(`the body is a ${k.schedule} schedule`);
    if (k.grid.length !== 2) say(`the body has ${k.grid.length} parallel axes, not 2`);
    if (k.reduce.length !== 1) say(`the body reduces over ${k.reduce.length} axes, not 1`);
    const reads = k.bindings.filter((x) => x.mode === "read");
    if (reads.length !== 2) say(`the body reads ${reads.length} bindings, not 2`);
    const tp = k.bindings.filter((x) => x.transposed).map((x) => x.name);
    if (tp.length) say(`"${tp[0]}" is read transposed, which this backend ignores`);
    for (const st of k.body?.steps ?? []) {
      if (st.k === "derived" || st.k === "derivedFrag") say(`"${st.name}" is a derived value`);
      if (st.k === "reduce" || st.k === "store") {
        if (st.inner.length) say(`the pass over "${st.axis}" contains a nested contraction`);
        if (st.locals.length) say(`the pass over "${st.axis}" declares a local accumulator`);
      }
      if (st.k === "reduce") {
        for (const up of st.updates) {
          if (up.k !== "mma") say(`the reduction uses ${up.k}, and this backend emits mma only`);
        }
      }
    }
    const accs = k.body?.accs ?? [];
    if (accs.length !== 1) say(`the body has ${accs.length} accumulators, not 1`);
  }

  const { bm, bn, bk } = k.tile;
  const [wgx, wgy] = k.workgroup;
  const [tm, tn] = k.fragment;
  const threads = wgx * wgy;
  const nacc = tm * tn;
  const ty = k.dtype;

  const [gm, gn] = k.grid;
  const rk = k.reduce[0];

  const extent = new Map([...k.grid, ...k.reduce].map((a) => [a.name, a.extent]));
  const bind = (mode: "read" | "write", i: number) => k.bindings.filter((b) => b.mode === mode)[i];
  const a = bind("read", 0), b = bind("read", 1), c = bind("write", 0);
  /** Row stride of a row-major [axes0, axes1] buffer is the extent of axes1. */
  const stride = (x: typeof a) => extent.get(x.axes[1])!;

  const out: string[] = [];
  const E = (s = "", i = 0) => out.push(s ? "  ".repeat(i) + s : "");
  const body: string[] = [];
  const B = (s = "", i = 3) => body.push(s ? "  ".repeat(i) + s : "");

  E(`module attributes {gpu.container_module} {`);
  E(`gpu.module @kernels {`, 1);
  E(`gpu.func @${k.name}(%${a.name}: memref<${a.elements}x${ty}, ${SC}>,`, 2);
  E(`                 %${b.name}: memref<${b.elements}x${ty}, ${SC}>,`, 2);
  E(`                 %${c.name}: memref<${c.elements}x${ty}, ${SC}>) kernel`, 2);
  E(`    attributes {spirv.entry_point_abi = ` +
    `#spirv.entry_point_abi<workgroup_size = [${wgx}, ${wgy}, 1]>} {`, 2);

  // ---- constants -----------------------------------------------------------
  const ragged = (a: AxisIR) => a.fit === "ragged";
  /** The masked-load fill. Not the accumulator's zero — see the constants below. */
  const PAD = k.pad === "zero" ? "%f0 " : "%fpad ";
  const anyRagged = [...k.grid, ...k.reduce].some(ragged);

  const needed = new Set<number>([
    0, 1, bk, bm, bn, rk.extent, gn.extent, threads, tm, tn, bm * bk, bk * bn,
  ]);
  if (anyRagged) {
    // Extents to compare against, and the last valid flat index of each buffer
    // so an out-of-range load can be clamped to a legal address instead of
    // branching. Emitting these only when something is ragged keeps the exact
    // case byte-identical to what it was before masks existed.
    for (const ax of [...k.grid, ...k.reduce]) needed.add(ax.extent);
    for (const bd of k.bindings) needed.add(bd.elements - 1);
  }
  for (let i = 0; i < Math.max(tm, tn); i++) needed.add(i);   // fragment offsets
  for (const v of [...needed].sort((x, y) => x - y)) B(`%c${v} = arith.constant ${v} : index`);
  B(`%f0 = arith.constant 0.0 : ${ty}`);
  // The accumulator's zero and the mask's identity are different constants, and
  // conflating them was only invisible while every identity was zero.
  if (anyRagged && k.pad !== "zero") {
    B(`%fpad = arith.constant ${PAD_LITERAL[k.pad]} : ${ty}`);
  }
  B();

  // ---- workgroup staging ---------------------------------------------------
  B(`%As = memref.alloc() : memref<${bm * bk}x${ty}, ${WG}>`);
  B(`%Bs = memref.alloc() : memref<${bk * bn}x${ty}, ${WG}>`);
  B();

  // ---- ids -----------------------------------------------------------------
  B(`%bx = gpu.block_id x`);
  B(`%by = gpu.block_id y`);
  B(`%tx = gpu.thread_id x`);
  B(`%ty = gpu.thread_id y`);
  B(`%tid0 = arith.muli %ty, %c${wgx} : index`);
  B(`%tid = arith.addi %tid0, %tx : index`);
  B(`%ty4 = arith.muli %ty, %c${tm} : index`);
  B(`%tx4 = arith.muli %tx, %c${tn} : index`);
  B(`%rowBase = arith.muli %by, %c${bm} : index`);
  B(`%colBase = arith.muli %bx, %c${bn} : index`);
  B();

  // ---- outer reduction loop ------------------------------------------------
  const accs = Array.from({ length: nacc }, (_, i) => `%acc${i}`);
  const types = Array(nacc).fill(ty).join(", ");
  B(`%out:${nacc} = scf.for %kk = %c0 to %c${rk.extent} step %c${bk} ` +
    `iter_args(${accs.map((x) => `${x} = %f0`).join(", ")}) -> (${types}) {`);

  // stage A: As[i] = a[(rowBase + i/BK)*strideA + (kk + i%BK)]
  B(`scf.for %i = %tid to %c${bm * bk} step %c${threads} {`, 4);
  B(`%r = arith.divui %i, %c${bk} : index`, 5);
  B(`%cc = arith.remui %i, %c${bk} : index`, 5);
  B(`%gr = arith.addi %rowBase, %r : index`, 5);
  B(`%gr2 = arith.muli %gr, %c${stride(a)} : index`, 5);
  B(`%gc = arith.addi %kk, %cc : index`, 5);
  B(`%off = arith.addi %gr2, %gc : index`, 5);
  if (ragged(gm) || ragged(rk)) {
    // Masked load, branchless: clamp the address so the access is always legal,
    // then select the identity for lanes that were out of range. A branch here
    // would diverge across a workgroup for exactly one block per row.
    const conds: string[] = [];
    if (ragged(gm)) { B(`%okr = arith.cmpi ult, %gr, %c${gm.extent} : index`, 5); conds.push("%okr"); }
    if (ragged(rk)) { B(`%okc = arith.cmpi ult, %gc, %c${rk.extent} : index`, 5); conds.push("%okc"); }
    B(conds.length === 2 ? `%ok = arith.andi %okr, %okc : i1` : `%ok = ${conds[0]} : i1`, 5);
    B(`%safe = arith.minui %off, %c${a.elements - 1} : index`, 5);
    B(`%raw = memref.load %${a.name}[%safe] : memref<${a.elements}x${ty}, ${SC}>`, 5);
    B(`%val = arith.select %ok, %raw, ${PAD}: ${ty}`, 5);
  } else {
    B(`%val = memref.load %${a.name}[%off] : memref<${a.elements}x${ty}, ${SC}>`, 5);
  }
  B(`memref.store %val, %As[%i] : memref<${bm * bk}x${ty}, ${WG}>`, 5);
  B(`}`, 4);

  // stage B: Bs[i] = b[(kk + i/BN)*strideB + (colBase + i%BN)]
  B(`scf.for %i = %tid to %c${bk * bn} step %c${threads} {`, 4);
  B(`%r = arith.divui %i, %c${bn} : index`, 5);
  B(`%cc = arith.remui %i, %c${bn} : index`, 5);
  B(`%gr = arith.addi %kk, %r : index`, 5);
  B(`%gr2 = arith.muli %gr, %c${stride(b)} : index`, 5);
  B(`%gc = arith.addi %colBase, %cc : index`, 5);
  B(`%off = arith.addi %gr2, %gc : index`, 5);
  if (ragged(rk) || ragged(gn)) {
    const conds: string[] = [];
    if (ragged(rk)) { B(`%okr = arith.cmpi ult, %gr, %c${rk.extent} : index`, 5); conds.push("%okr"); }
    if (ragged(gn)) { B(`%okc = arith.cmpi ult, %gc, %c${gn.extent} : index`, 5); conds.push("%okc"); }
    B(conds.length === 2 ? `%ok = arith.andi %okr, %okc : i1` : `%ok = ${conds[0]} : i1`, 5);
    B(`%safe = arith.minui %off, %c${b.elements - 1} : index`, 5);
    B(`%raw = memref.load %${b.name}[%safe] : memref<${b.elements}x${ty}, ${SC}>`, 5);
    B(`%val = arith.select %ok, %raw, ${PAD}: ${ty}`, 5);
  } else {
    B(`%val = memref.load %${b.name}[%off] : memref<${b.elements}x${ty}, ${SC}>`, 5);
  }
  B(`memref.store %val, %Bs[%i] : memref<${bk * bn}x${ty}, ${WG}>`, 5);
  B(`}`, 4);

  B(`gpu.barrier`, 4);

  // ---- inner accumulate ----------------------------------------------------
  // Emitted either as a loop over bk, or fully unrolled. See EmitOptions.
  let finalAcc: string[];

  if (opts.unrollK) {
    // Needs the literal step indices as constants.
    const extra = new Set<number>();
    for (let s = 0; s < bk; s++) extra.add(s);
    const missing = [...extra].filter((v) => !needed.has(v)).sort((x, y) => x - y);
    for (const v of missing) B(`%ck${v} = arith.constant ${v} : index`, 4);
    const kconst = (s: number) => (needed.has(s) ? `%c${s}` : `%ck${s}`);

    let cur = [...accs];
    for (let s = 0; s < bk; s++) {
      for (let m = 0; m < tm; m++) {
        B(`%u${s}am${m}a = arith.addi %ty4, %c${m} : index`, 4);
        B(`%u${s}am${m}b = arith.muli %u${s}am${m}a, %c${bk} : index`, 4);
        B(`%u${s}am${m}c = arith.addi %u${s}am${m}b, ${kconst(s)} : index`, 4);
        B(`%u${s}af${m} = memref.load %As[%u${s}am${m}c] : memref<${bm * bk}x${ty}, ${WG}>`, 4);
      }
      for (let n = 0; n < tn; n++) {
        B(`%u${s}bn${n}a = arith.muli ${kconst(s)}, %c${bn} : index`, 4);
        B(`%u${s}bn${n}b = arith.addi %tx4, %c${n} : index`, 4);
        B(`%u${s}bn${n}c = arith.addi %u${s}bn${n}a, %u${s}bn${n}b : index`, 4);
        B(`%u${s}bf${n} = memref.load %Bs[%u${s}bn${n}c] : memref<${bk * bn}x${ty}, ${WG}>`, 4);
      }
      const next: string[] = [];
      for (let m = 0; m < tm; m++) {
        for (let n = 0; n < tn; n++) {
          const i = m * tn + n;
          B(`%u${s}p${i} = arith.mulf %u${s}af${m}, %u${s}bf${n} : ${ty}`, 4);
          B(`%u${s}s${i} = arith.addf ${cur[i]}, %u${s}p${i} : ${ty}`, 4);
          next.push(`%u${s}s${i}`);
        }
      }
      cur = next;
    }
    finalAcc = cur;
  } else {
    B(`%acc:${nacc} = scf.for %k = %c0 to %c${bk} step %c1 ` +
      `iter_args(${accs.map((_, i) => `%i${i} = ${accs[i]}`).join(", ")}) -> (${types}) {`, 4);

    for (let m = 0; m < tm; m++) {
      B(`%am${m}a = arith.addi %ty4, %c${m} : index`, 5);
      B(`%am${m}b = arith.muli %am${m}a, %c${bk} : index`, 5);
      B(`%am${m}c = arith.addi %am${m}b, %k : index`, 5);
      B(`%af${m} = memref.load %As[%am${m}c] : memref<${bm * bk}x${ty}, ${WG}>`, 5);
    }
    for (let n = 0; n < tn; n++) {
      B(`%bn${n}a = arith.muli %k, %c${bn} : index`, 5);
      B(`%bn${n}b = arith.addi %tx4, %c${n} : index`, 5);
      B(`%bn${n}c = arith.addi %bn${n}a, %bn${n}b : index`, 5);
      B(`%bf${n} = memref.load %Bs[%bn${n}c] : memref<${bk * bn}x${ty}, ${WG}>`, 5);
    }

    const yields: string[] = [];
    for (let m = 0; m < tm; m++) {
      for (let n = 0; n < tn; n++) {
        const i = m * tn + n;
        B(`%p${i} = arith.mulf %af${m}, %bf${n} : ${ty}`, 5);
        B(`%s${i} = arith.addf %i${i}, %p${i} : ${ty}`, 5);
        yields.push(`%s${i}`);
      }
    }
    B(`scf.yield ${yields.join(", ")} : ${types}`, 5);
    B(`}`, 4);
    finalAcc = accs.map((_, i) => `%acc#${i}`);
  }

  B(`gpu.barrier`, 4);
  B(`scf.yield ${finalAcc.join(", ")} : ${types}`, 4);
  B(`}`, 3);
  B();

  // ---- relu + store --------------------------------------------------------
  for (let m = 0; m < tm; m++) {
    B(`%sr${m}a = arith.addi %ty4, %c${m} : index`);
    B(`%sr${m} = arith.addi %rowBase, %sr${m}a : index`);
    B(`%sr${m}n = arith.muli %sr${m}, %c${stride(c)} : index`);
    for (let n = 0; n < tn; n++) {
      const i = m * tn + n;
      B(`%sc${i}a = arith.addi %tx4, %c${n} : index`);
      B(`%sc${i} = arith.addi %colBase, %sc${i}a : index`);
      B(`%so${i} = arith.addi %sr${m}n, %sc${i} : index`);
      // Rule 2: cmpf+select, not maximumf/maxnumf. See the header.
      B(`%rc${i} = arith.cmpf ogt, %out#${i}, %f0 : ${ty}`);
      B(`%rl${i} = arith.select %rc${i}, %out#${i}, %f0 : ${ty}`);
      if (ragged(gm) || ragged(gn)) {
        // A store cannot be clamped -- writing to a legal-but-wrong address
        // corrupts a real element -- so this one is a genuine conditional.
        const conds: string[] = [];
        if (ragged(gm)) { B(`%sm${i} = arith.cmpi ult, %sr${m}, %c${gm.extent} : index`); conds.push(`%sm${i}`); }
        if (ragged(gn)) { B(`%sn${i} = arith.cmpi ult, %sc${i}, %c${gn.extent} : index`); conds.push(`%sn${i}`); }
        B(conds.length === 2 ? `%sk${i} = arith.andi %sm${i}, %sn${i} : i1` : `%sk${i} = ${conds[0]} : i1`);
        B(`scf.if %sk${i} {`);
        B(`  memref.store %rl${i}, %${c.name}[%so${i}] : memref<${c.elements}x${ty}, ${SC}>`);
        B(`}`);
      } else {
        B(`memref.store %rl${i}, %${c.name}[%so${i}] : memref<${c.elements}x${ty}, ${SC}>`);
      }
    }
  }
  B();
  B(`gpu.return`);

  out.push(...body);
  E(`}`, 2);
  E(`}`, 1);
  E(`}`);
  return out.join("\n") + "\n";
}
