/**
 * emit-wgsl.ts — tessera IR straight to WGSL, bypassing MLIR entirely.
 *
 * This is the A/B that settles docs/002 §4. The MLIR path crosses four
 * translation boundaries the hand-written kernel does not
 * (IR → MLIR → SPIR-V → naga → WGSL), and by elimination that is the remaining
 * candidate for the 1.57× gap: three hypotheses about tessera making worse
 * codegen choices were each probed and refuted.
 *
 * For the comparison to mean anything, this backend must make the SAME decisions
 * as the MLIR one and differ ONLY in how it emits them. So it deliberately
 * mirrors emit-mlir.ts: same tile, same fragment layout, same staging loop
 * structure, same clamp-and-select for masked loads, same conditional store, the
 * same rolled k loop. Anything that differs here is a confound, not a result.
 *
 * Two differences are unavoidable and are the point:
 *   - loops come out structured, because WGSL is structured, where the MLIR path
 *     goes through an unstructured IR and naga reconstructs `loop{}/continuing{}`
 *   - storage access modes are `read` where the surface says `read`, because the
 *     IR knows the binding mode. The MLIR path drops that on the floor.
 *     (Probe A measured access mode at 1.00×, so this is not a confound.)
 *
 * It is also the third oracle. With three backends agreeing bit-for-bit, a future
 * divergence localises to one of them in a single comparison instead of a
 * twelve-pass bisection.
 */

import { PAD_LITERAL, type AxisIR, type BindingIR, type KernelIR } from "./ir.ts";

export function emitWGSL(k: KernelIR): string {
  const { bm, bn, bk } = k.tile;
  const [wgx, wgy] = k.workgroup;
  const [tm, tn] = k.fragment;
  const threads = wgx * wgy;
  const nacc = tm * tn;
  const ty = k.dtype;

  const [gm, gn] = k.grid;
  const rk = k.reduce[0];
  const ragged = (a: AxisIR) => a.fit === "ragged";
  /** The masked-load fill, by name. `zero` is not the only possibility. */
  const PAD = PAD_LITERAL[k.pad];

  const extent = new Map([...k.grid, ...k.reduce].map((a) => [a.name, a.extent]));
  const pick = (mode: "read" | "write", i: number) => k.bindings.filter((b) => b.mode === mode)[i];
  const a = pick("read", 0), b = pick("read", 1), c = pick("write", 0);
  const softmax = k.schedule === "softmax";
  const stride = (x: typeof a) => extent.get(x.axes[1])!;

  const L: string[] = [];
  const P = (s = "", i = 0) => L.push(s ? "  ".repeat(i) + s : "");
  const u = (n: number) => `${n}u`;


  // ---- the access layer ----------------------------------------------------
  // Reading and writing a binding is the same operation regardless of schedule:
  // flatten the logical coordinates row-major over the binding's own axes, and
  // guard exactly the axes that are ragged. Nothing here knows what a matmul is,
  // which is the property being tested — a second schedule should be able to use
  // it unchanged.
  const axisOf = new Map([...k.grid, ...k.reduce].map((x) => [x.name, x]));

  /** The bound checks a binding needs at these coordinates, or [] if exact. */
  const guards = (bd: BindingIR, i0: string, i1: string): string[] => {
    const out: string[] = [];
    for (const [j, coord] of [[0, i0], [1, i1]] as const) {
      const ax = axisOf.get(bd.axes[j])!;
      if (ragged(ax)) out.push(`${coord} < ${u(ax.extent)}`);
    }
    return out;
  };

  /** `<lhs> = <the value at (i0,i1)>;` — clamped and selected when masked. */
  const emitLoad = (lhs: string, bd: BindingIR, i0: string, i1: string, ind: number) => {
    P(`let off = ${i0} * ${u(stride(bd))} + ${i1};`, ind);
    const g = guards(bd, i0, i1);
    if (g.length === 0) {
      P(`${lhs} = ${bd.name}[off];`, ind);
    } else {
      // Clamp then select: branchless, and identical in meaning to what the MLIR
      // backend emits. A branch would diverge for exactly one block per row.
      P(`${lhs} = select(${PAD}, ${bd.name}[min(off, ${u(bd.elements - 1)})], ${g.join(" && ")});`, ind);
    }
  };

  /** A store cannot be clamped: a legal-but-wrong address corrupts a real element. */
  const emitStore = (bd: BindingIR, i0: string, i1: string, val: string, ind: number) => {
    const g = guards(bd, i0, i1);
    const write = `${bd.name}[${i0} * ${u(stride(bd))} + ${i1}] = ${val};`;
    if (g.length === 0) { P(write, ind); return; }
    P(`if (${g.join(" && ")}) {`, ind);
    P(write, ind + 1);
    P(`}`, ind);
  };


  /**
   * Softmax: three passes over the reduce axis, with a cross-lane row reduction
   * between them.
   *
   * Every access below goes through emitLoad / emitStore unchanged — the same
   * functions the matmul schedule uses. No mask is placed here, and no index is
   * flattened here; both come from the binding's axes and which of them are
   * ragged. That reuse is the thing docs/004 items 2-4 are measuring.
   */
  function emitSoftmax(): void {
    const rowOf = (m: number) => `rowBase + ty * ${u(tm)} + ${u(m)}`;
    const colOf = (n: number) => `nn + tx * ${u(tn)} + ${u(n)}`;
    const NEG = PAD_LITERAL.negInf;

    /** One masked read of x at fragment cell (m, n), bound to `v`. */
    const readCell = (m: number, n: number, use: string, ind: number) => {
      P(`{`, ind);
      P(`let row = ${rowOf(m)};`, ind + 1);
      P(`let col = ${colOf(n)};`, ind + 1);
      emitLoad(`let v`, a, "row", "col", ind + 1);
      P(use, ind + 1);
      P(`}`, ind);
    };

    /** Combine `local[m]` across the ${wgx} lanes that share a row. */
    const acrossLanes = (local: string, combine: (acc: string, x: string) => string, init: string) => {
      for (let m = 0; m < tm; m++) {
        P(`scratch[(ty * ${u(tm)} + ${u(m)}) * ${u(wgx)} + tx] = ${local}[${m}];`, 1);
      }
      P(`workgroupBarrier();`, 1);
      for (let m = 0; m < tm; m++) {
        P(`{`, 1);
        P(`var r = ${init};`, 2);
        P(`for (var j : u32 = 0u; j < ${u(wgx)}; j = j + 1u) {`, 2);
        P(`r = ${combine("r", `scratch[(ty * ${u(tm)} + ${u(m)}) * ${u(wgx)} + j]`)};`, 3);
        P(`}`, 2);
        P(`${local}[${m}] = r;`, 2);
        P(`}`, 1);
      }
      P(`workgroupBarrier();`, 1);
    };

    const nExtent = rk.extent, nBlock = bn;

    // ---- pass 1: the row maximum
    P(`var mx : array<${ty}, ${tm}>;`, 1);
    for (let m = 0; m < tm; m++) P(`mx[${m}] = ${NEG};`, 1);
    P(`for (var nn : u32 = 0u; nn < ${u(nExtent)}; nn = nn + ${u(nBlock)}) {`, 1);
    for (let m = 0; m < tm; m++) {
      for (let n = 0; n < tn; n++) readCell(m, n, `mx[${m}] = max(mx[${m}], v);`, 2);
    }
    P(`}`, 1);
    acrossLanes("mx", (r, x) => `max(${r}, ${x})`, NEG);

    // ---- pass 2: the row sum of exp(x - max)
    P(`var sm : array<${ty}, ${tm}>;`, 1);
    for (let m = 0; m < tm; m++) P(`sm[${m}] = 0.0;`, 1);
    P(`for (var nn : u32 = 0u; nn < ${u(nExtent)}; nn = nn + ${u(nBlock)}) {`, 1);
    for (let m = 0; m < tm; m++) {
      for (let n = 0; n < tn; n++) {
        // exp(negInf - mx) is 0, so masked lanes contribute nothing to the sum.
        // That is the identity conversion the surface records in expTile's type.
        readCell(m, n, `sm[${m}] = sm[${m}] + exp(v - mx[${m}]);`, 2);
      }
    }
    P(`}`, 1);
    acrossLanes("sm", (r, x) => `${r} + ${x}`, "0.0");

    // ---- pass 3: normalise and store
    P(`for (var nn : u32 = 0u; nn < ${u(nExtent)}; nn = nn + ${u(nBlock)}) {`, 1);
    for (let m = 0; m < tm; m++) {
      for (let n = 0; n < tn; n++) {
        P(`{`, 2);
        P(`let row = ${rowOf(m)};`, 3);
        P(`let col = ${colOf(n)};`, 3);
        emitLoad(`let v`, a, "row", "col", 3);
        emitStore(c, "row", "col", `exp(v - mx[${m}]) / sm[${m}]`, 3);
        P(`}`, 2);
      }
    }
    P(`}`, 1);
  }

  P(`// Generated by tessera from a TypeScript kernel. Do not edit.`);
  P(`//`);
  P(`//   ${k.name}: ${k.grid.map((x) => `${x.name}=${x.extent}`).join(" x ")} x ${rk.name}=${rk.extent}`);
  P(`//   schedule ${k.schedule}   workgroup ${wgx}x${wgy}x1   fragment ${tm}x${tn}`);
  P(`//   ${k.maskedLoads.length ? `masked: ${k.maskedLoads.join(" ")}   pad ${k.pad} (${PAD})` : "no masks: every axis divides its block"}`);
  P();

  // ---- bindings ------------------------------------------------------------
  // The access mode comes from the surface. `input()` is read-only and says so.
  for (const [i, bd] of k.bindings.entries()) {
    const mode = bd.mode === "read" ? "read" : "read_write";
    P(`@group(0) @binding(${i}) var<storage, ${mode}> ${bd.name} : array<${ty}>;`);
  }
  P();
  if (softmax) {
    // No operand staging: every element is read once per pass, so staging buys
    // nothing. What IS needed is scratch for the cross-lane row reduction —
    // one slot per row per lane.
    P(`var<workgroup> scratch : array<${ty}, ${bm * wgx}>;   // rows x lanes`);
  } else {
    P(`var<workgroup> As : array<${ty}, ${bm * bk}>;`);
    P(`var<workgroup> Bs : array<${ty}, ${bk * bn}>;`);
  }
  P();

  P(`@compute @workgroup_size(${wgx}, ${wgy}, 1)`);
  P(`fn ${k.name}(@builtin(workgroup_id)        wg  : vec3<u32>,`);
  P(`${" ".repeat(k.name.length + 3)}@builtin(local_invocation_id) lid : vec3<u32>) {`);

  P(softmax ? `let blockRow = wg.x;` : `let blockRow = wg.y;`, 1);
  if (!softmax) P(`let blockCol = wg.x;`, 1);
  P(`let tx  = lid.x;`, 1);
  P(`let ty  = lid.y;`, 1);
  if (!softmax) P(`let tid = ty * ${u(wgx)} + tx;`, 1);
  P(`let rowBase = blockRow * ${u(bm)};`, 1);
  if (!softmax) P(`let colBase = blockCol * ${u(bn)};`, 1);
  P();

  if (softmax) { emitSoftmax(); P(`}`); return L.join("\n") + "\n"; }

  P(`var acc : array<${ty}, ${nacc}>;`, 1);
  P();

  // ---- reduction over blocks ----------------------------------------------
  P(`for (var kk : u32 = 0u; kk < ${u(rk.extent)}; kk = kk + ${u(bk)}) {`, 1);

  // stage A
  P(`for (var i : u32 = tid; i < ${u(bm * bk)}; i = i + ${u(threads)}) {`, 2);
  P(`let r  = i / ${u(bk)};`, 3);
  P(`let cc = i % ${u(bk)};`, 3);
  P(`let gr = rowBase + r;`, 3);
  P(`let gc = kk + cc;`, 3);
  emitLoad("As[i]", a, "gr", "gc", 3);
  P(`}`, 2);

  // stage B
  P(`for (var i : u32 = tid; i < ${u(bk * bn)}; i = i + ${u(threads)}) {`, 2);
  P(`let r  = i / ${u(bn)};`, 3);
  P(`let cc = i % ${u(bn)};`, 3);
  P(`let gr = kk + r;`, 3);
  P(`let gc = colBase + cc;`, 3);
  emitLoad("Bs[i]", b, "gr", "gc", 3);
  P(`}`, 2);

  P(`workgroupBarrier();`, 2);
  P();

  // ---- accumulate ----------------------------------------------------------
  P(`for (var k : u32 = 0u; k < ${u(bk)}; k = k + 1u) {`, 2);
  for (let m = 0; m < tm; m++) {
    P(`let af${m} = As[(ty * ${u(tm)} + ${u(m)}) * ${u(bk)} + k];`, 3);
  }
  for (let n = 0; n < tn; n++) {
    P(`let bf${n} = Bs[k * ${u(bn)} + (tx * ${u(tn)} + ${u(n)})];`, 3);
  }
  for (let m = 0; m < tm; m++) {
    for (let n = 0; n < tn; n++) {
      const i = m * tn + n;
      P(`acc[${i}] = acc[${i}] + af${m} * bf${n};`, 3);
    }
  }
  P(`}`, 2);

  P();
  P(`workgroupBarrier();`, 2);
  P(`}`, 1);
  P();

  // ---- relu + store --------------------------------------------------------
  for (let m = 0; m < tm; m++) {
    P(`let row${m} = rowBase + ty * ${u(tm)} + ${u(m)};`, 1);
    for (let n = 0; n < tn; n++) {
      const i = m * tn + n;
      P(`let col${i} = colBase + tx * ${u(tn)} + ${u(n)};`, 1);
      // relu as a comparison + select, matching the MLIR backend exactly:
      // `max()` there lowered with NaN-propagating semantics that naga rejects,
      // and this form is also what the CPU oracle computes.
      const val = `select(0.0, acc[${i}], acc[${i}] > 0.0)`;
      emitStore(c, `row${m}`, `col${i}`, val, 1);
    }
  }

  P(`}`);
  return L.join("\n") + "\n";
}
