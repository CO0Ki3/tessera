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
import type { Expr, RowExpr } from "./body.ts";

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
  const rowwise = k.schedule === "rowwise";
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
   * The row-wise schedule, emitted from the parsed body.
   *
   * There is no softmax path and no layernorm path. The accumulators, the passes
   * over the reduce axis, the derived row values and the store all come out of
   * `k.body`, which was read from the statements the author wrote. Adding
   * layernorm after softmax added no lines here.
   *
   * Every access still goes through emitLoad / emitStore — the same functions the
   * matmul schedule uses — so no mask is placed here and no index is flattened
   * here. That was measured in docs/004 part 1 and has not changed.
   */
  function emitRowwise(): void {
    const body = k.body;
    if (!body) throw new Error(`${k.name}: rowwise schedule with no parsed body`);
    const NEG = PAD_LITERAL.negInf;
    const rowOf = (m: number) => `rowBase + ty * ${u(tm)} + ${u(m)}`;
    const colOf = (n: number) => `nn + tx * ${u(tn)} + ${u(n)}`;
    const idx = new Map(body.accs.map((a, i) => [a.name, i]));
    const slot = (name: string, m: number) => `${name}[${m}]`;

    /** A block expression, as WGSL, for fragment cell (m, n) with `v` already loaded. */
    const blockExpr = (e: Expr, m: number): string => {
      switch (e.k) {
        case "tile":   return `v`;
        case "unary":  return e.op === "sq"
          ? `(${blockExpr(e.a, m)} * ${blockExpr(e.a, m)})`
          : `exp(${blockExpr(e.a, m)})`;
        case "rowOp": {
          const a = blockExpr(e.a, m), r = rowExpr(e.row, m);
          return e.op === "sub" ? `(${a} - ${r})` : e.op === "mul" ? `(${a} * ${r})` : `(${a} / ${r})`;
        }
      }
    };

    /** A row expression, as WGSL, for the m-th row this invocation owns. */
    const rowExpr = (r: RowExpr, m: number): string => {
      switch (r.k) {
        case "acc":  return slot(r.name, m);
        case "mean": return `(${rowExpr(r.a, m)} / ${nExtent}.0)`;
        case "rstd": {
          const mu = rowExpr(r.mean, m);
          return `inverseSqrt((${rowExpr(r.sumSq, m)} / ${nExtent}.0) - ${mu} * ${mu} + ${r.eps})`;
        }
      }
    };

    const nExtent = rk.extent, nBlock = bn;

    // ---- accumulators
    for (const acc of body.accs) {
      P(`var ${acc.name} : array<${ty}, ${tm}>;`, 1);
      const init = acc.init === "negInf" ? NEG : PAD_LITERAL[acc.init];
      for (let m = 0; m < tm; m++) P(`${slot(acc.name, m)} = ${init};`, 1);
    }
    P();

    // ---- steps, in source order
    for (const step of body.steps) {
      if (step.k === "derived") {
        // A derived row value is computed once per row this invocation owns,
        // after the passes that produced its inputs and before whatever uses it.
        P(`var ${step.name} : array<${ty}, ${tm}>;`, 1);
        for (let m = 0; m < tm; m++) P(`${slot(step.name, m)} = ${rowExpr(step.expr, m)};`, 1);
        P();
        continue;
      }
      const pass = step;
      P(`for (var nn : u32 = 0u; nn < ${u(nExtent)}; nn = nn + ${u(nBlock)}) {`, 1);
      for (let m = 0; m < tm; m++) {
        for (let n = 0; n < tn; n++) {
          P(`{`, 2);
          P(`let row = ${rowOf(m)};`, 3);
          P(`let col = ${colOf(n)};`, 3);
          if (pass.k === "reduce") {
            // Every update in a pass reads the same cell once.
            emitLoad(`let v`, bindingOf(pass.updates[0].value), "row", "col", 3);
            for (const up of pass.updates) {
              const val = blockExpr(up.value, m);
              P(up.op === "max"
                ? `${slot(up.acc, m)} = max(${slot(up.acc, m)}, ${val});`
                : `${slot(up.acc, m)} = ${slot(up.acc, m)} + ${val};`, 3);
            }
          } else {
            emitLoad(`let v`, bindingOf(pass.value), "row", "col", 3);
            emitStore(byName(pass.binding), "row", "col", blockExpr(pass.value, m), 3);
          }
          P(`}`, 2);
        }
      }
      P(`}`, 1);

      // A reduce pass ends by combining each accumulator across the lanes that
      // share a row. How many accumulators there are is a property of the body.
      if (pass.k === "reduce") {
        const touched = [...new Set(pass.updates.map((x) => x.acc))];
        for (const nm of touched) {
          for (let m = 0; m < tm; m++) {
            P(`scratch[(ty * ${u(tm)} + ${u(m)}) * ${u(wgx)} + tx] = ${slot(nm, m)};`, 1);
          }
          P(`workgroupBarrier();`, 1);
          const op = pass.updates.find((x) => x.acc === nm)!.op;
          const init = op === "max" ? NEG : "0.0";
          for (let m = 0; m < tm; m++) {
            P(`{`, 1);
            P(`var r = ${init};`, 2);
            P(`for (var j : u32 = 0u; j < ${u(wgx)}; j = j + 1u) {`, 2);
            const cell = `scratch[(ty * ${u(tm)} + ${u(m)}) * ${u(wgx)} + j]`;
            P(op === "max" ? `r = max(r, ${cell});` : `r = r + ${cell};`, 3);
            P(`}`, 2);
            P(`${slot(nm, m)} = r;`, 2);
            P(`}`, 1);
          }
          P(`workgroupBarrier();`, 1);
        }
      }
      P();
    }
  }

  /** The binding a block expression ultimately reads from. */
  function bindingOf(e: Expr): BindingIR {
    while (e.k !== "tile") e = e.k === "unary" ? e.a : e.a;
    return byName(e.binding);
  }
  function byName(n: string): BindingIR {
    const bd = k.bindings.find((x) => x.name === n);
    if (!bd) throw new Error(`no binding named ${n}`);
    return bd;
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
  if (rowwise) {
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

  P(rowwise ? `let blockRow = wg.x;` : `let blockRow = wg.y;`, 1);
  if (!rowwise) P(`let blockCol = wg.x;`, 1);
  P(`let tx  = lid.x;`, 1);
  P(`let ty  = lid.y;`, 1);
  if (!rowwise) P(`let tid = ty * ${u(wgx)} + tx;`, 1);
  P(`let rowBase = blockRow * ${u(bm)};`, 1);
  if (!rowwise) P(`let colBase = blockCol * ${u(bn)};`, 1);
  P();

  if (rowwise) { emitRowwise(); P(`}`); return L.join("\n") + "\n"; }

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
