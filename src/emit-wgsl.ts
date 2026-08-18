/**
 * emit-wgsl.ts — tessera IR to WGSL, one emitter for every schedule.
 *
 * There used to be two: a fragment path that staged two operands and accumulated an
 * outer product, and a row-wise path that folded one cell at a time out of global
 * memory. docs/004 R10 called that split honest — "different memory schedules, not
 * different spellings of one" — and it was wrong. Both are a contraction, and the
 * differences fall out of the operands' index sets. See src/contraction.ts.
 *
 * What the plan decides, so that this file does not:
 *   - which lane dimension carries which accumulate axis (the contiguous axis takes
 *     `x`, which is coalescing rather than taste)
 *   - which operands are staged (the ones reused across an accumulate axis they omit)
 *   - whether lane partials must be combined (only when the contract axis got a lane)
 *   - how many sequential contract steps each invocation walks
 *
 * Every access still goes through the access layer below, so no mask is placed here
 * and no index is flattened here — those come from the binding's axes and which of
 * them are ragged.
 */

import { PAD_LITERAL, type AxisIR, type BindingIR, type KernelIR } from "./ir.ts";
import { planContraction, type ContractionPlan } from "./contraction.ts";
import type { Expr, FragExpr, RowExpr, Update } from "./body.ts";

export function emitWGSL(k: KernelIR): string {
  const [wgx, wgy] = k.workgroup;
  const ty = k.dtype;
  const body = k.body;
  if (!body) throw new Error(`${k.name}: no parsed body`);

  const accumulate = k.grid;
  const contract = k.reduce[0];
  const out = k.bindings.find((b) => b.mode === "write")!;
  const reads = k.bindings.filter((b) => b.mode === "read");

  const plan = planContraction(
    accumulate, contract,
    reads.map((b) => ({ binding: b.name, axes: b.axes })),
    k.workgroup,
    out.axes[1],                      // the output's last index is contiguous
  );

  const L: string[] = [];
  const P = (s = "", i = 0) => L.push(s ? "  ".repeat(i) + s : "");
  const u = (n: number) => `${n}u`;
  const PAD = PAD_LITERAL[k.pad];
  const ragged = (a: AxisIR) => a.fit === "ragged";
  const axisOf = new Map([...k.grid, ...k.reduce].map((x) => [x.name, x]));
  const byName = (n: string) => k.bindings.find((x) => x.name === n)!;
  /**
   * The workgroup buffer staging an operand. Prefixed rather than suffixed: the
   * obvious `${binding}s` turns a binding named `a` into `as`, which is a WGSL
   * reserved keyword. Any scheme that derives identifiers from user-supplied names
   * has to assume they will collide with the target language eventually.
   */
  const stageName = (binding: string) => `stage_${binding}`;
  /**
   * Names the emitter invents. They share a scope with the accumulators the user
   * named, so they carry a prefix no surface identifier can produce: layernorm
   * calls an accumulator `s`, which shadowed a loop counter also called `s` and
   * turned `s[0]` into an index into a u32. Same class as `a` staging into `as`.
   */
  const T_ = (n: string) => `t_${n}`;
  const stride = (bd: BindingIR) => axisOf.get(bd.axes[1])!.extent;
  const laneW = { x: wgx, y: wgy } as const;

  // ---- the access layer ------------------------------------------------------
  // Reading and writing a binding is the same operation regardless of schedule:
  // flatten row-major over the binding's own axes, and guard exactly the axes that
  // are ragged. Nothing here knows what a contraction is.
  const guards = (bd: BindingIR, i0: string, i1: string): string[] => {
    const g: string[] = [];
    for (const [j, coord] of [[0, i0], [1, i1]] as const) {
      const ax = axisOf.get(bd.axes[j])!;
      if (ragged(ax)) g.push(`${coord} < ${u(ax.extent)}`);
    }
    return g;
  };
  let offN = 0;
  const emitLoad = (lhs: string, bd: BindingIR, i0: string, i1: string, ind: number) => {
    const off = `off${offN++}`;
    P(`let ${off} = ${i0} * ${u(stride(bd))} + ${i1};`, ind);
    const g = guards(bd, i0, i1);
    P(g.length === 0
      ? `${lhs} = ${bd.name}[${off}];`
      // Clamp then select: branchless. A branch would diverge for exactly one
      // block per row.
      : `${lhs} = select(${PAD}, ${bd.name}[min(${off}, ${u(bd.elements - 1)})], ${g.join(" && ")});`, ind);
  };
  const emitStore = (bd: BindingIR, i0: string, i1: string, val: string, ind: number) => {
    const g = guards(bd, i0, i1);
    const w = `${bd.name}[${i0} * ${u(stride(bd))} + ${i1}] = ${val};`;
    if (!g.length) { P(w, ind); return; }
    // A store cannot be clamped: a legal-but-wrong address corrupts a real element.
    P(`if (${g.join(" && ")}) {`, ind); P(w, ind + 1); P(`}`, ind);
  };

  // ---- geometry from the plan ------------------------------------------------
  const slices = plan.lanes.perAxis;                       // accumulate axis -> lane, slice
  const sliceOf = new Map(slices.map((s) => [s.axis, s]));
  const frag = plan.lanes.fragment;
  const contractLaneW = plan.lanes.contractLanes.reduce((n, l) => n * laneW[l], 1);
  const contractBlock = contract.block;
  const seqDepth = contractBlock / contractLaneW;
  const staged = plan.operands.filter((o) => o.staged);

  /** Every fragment cell, as an assignment of a slice index to each accumulate axis. */
  const cells: Record<string, number>[] = [{}];
  for (const s of slices) {
    const next: Record<string, number>[] = [];
    for (const c of cells) for (let i = 0; i < s.slice; i++) next.push({ ...c, [s.axis]: i });
    cells.length = 0; cells.push(...next);
  }
  const cellIndex = (c: Record<string, number>) =>
    slices.reduce((n, s) => n * s.slice + c[s.axis], 0);
  /** The value name an operand contributes at this cell — projected onto its own axes. */
  const operandVal = (bindingName: string, c: Record<string, number>) => {
    const bd = byName(bindingName);
    const own = bd.axes.filter((a) => sliceOf.has(a)).map((a) => c[a]);
    return `v_${bindingName}${own.map((i) => `_${i}`).join("")}`;
  };
  /** Global coordinate of an accumulate axis at slice index i. */
  const accCoord = (axis: string, i: number) => {
    const s = sliceOf.get(axis)!;
    return `base_${axis} + ${s.lane === "x" ? "tx" : "ty"} * ${u(s.slice)} + ${u(i)}`;
  };

  // ---- header ---------------------------------------------------------------
  P(`// Generated by tessera from a TypeScript kernel. Do not edit.`);
  P(`//`);
  P(`//   ${k.name}   accumulate (${plan.accumulate.join(", ")})   contract ${plan.contract}`);
  P(`//   lanes ${slices.map((s) => `${s.axis}->${s.lane}(${s.slice})`).join(" ")}` +
    `${plan.lanes.contractLanes.length ? `  ${plan.contract}->${plan.lanes.contractLanes.join("")}` : ""}` +
    `   fragment ${frag}   ${seqDepth} sequential steps`);
  P(`//   staged: ${staged.map((o) => o.binding).join(", ") || "nothing — no operand is reused"}` +
    `   workgroup ${plan.workgroupElements * 4} B`);
  P(`//   ${k.maskedLoads.length ? `masked: ${k.maskedLoads.join(" ")}   pad ${k.pad} (${PAD})` : "no masks: every axis divides its block"}`);
  P();

  for (const [i, bd] of k.bindings.entries()) {
    P(`@group(0) @binding(${i}) var<storage, ${bd.mode === "read" ? "read" : "read_write"}> ${bd.name} : array<${ty}>;`);
  }
  P();
  for (const o of staged) P(`var<workgroup> ${stageName(o.binding)} : array<${ty}, ${o.stagedElements}>;`);
  if (!staged.length && plan.lanes.needsCombine) {
    P(`var<workgroup> scratch : array<${ty}, ${plan.workgroupElements}>;   // fragment x lanes`);
  }
  P();

  P(`@compute @workgroup_size(${wgx}, ${wgy}, 1)`);
  P(`fn ${k.name}(@builtin(workgroup_id)        wg  : vec3<u32>,`);
  P(`${" ".repeat(k.name.length + 3)}@builtin(local_invocation_id) lid : vec3<u32>) {`);
  P(`let tx = lid.x;`, 1);
  P(`let ty = lid.y;`, 1);
  P(`let tid = ty * ${u(wgx)} + tx;`, 1);
  // Dispatch is [x, y, z]; accumulate axes were listed grid-order, and derive()
  // maps the last one to x for two axes and the first for one.
  accumulate.forEach((a, i) => {
    const wgComp = accumulate.length === 2 ? (i === 0 ? "wg.y" : "wg.x") : "wg.x";
    P(`let base_${a.name} = ${wgComp} * ${u(a.block)};`, 1);
  });
  P();

  // ---- accumulators ----------------------------------------------------------
  for (const acc of body.accs) {
    P(`var ${acc.name} : array<${ty}, ${frag}>;`, 1);
    const init = PAD_LITERAL[acc.init];
    for (let c = 0; c < frag; c++) P(`${acc.name}[${c}] = ${init};`, 1);
  }
  P();

  // ---- the contraction --------------------------------------------------------
  const reduceSteps = body.steps.filter((s) => s.k === "reduce");
  for (const step of reduceSteps) emitContractionPass(step.updates);

  function emitContractionPass(updates: readonly Update[]): void {
    P(`for (var ${T_("cb")} : u32 = 0u; ${T_("cb")} < ${u(contract.extent)}; ${T_("cb")} = ${T_("cb")} + ${u(contractBlock)}) {`, 1);

    for (const o of staged) {
      const bd = byName(o.binding);
      const dims = bd.axes.map((a) => axisOf.get(a)!.block);
      P(`for (var ${T_("i")} : u32 = tid; ${T_("i")} < ${u(o.stagedElements)}; ${T_("i")} = ${T_("i")} + ${u(wgx * wgy)}) {`, 2);
      P(`let ${T_("r")} = ${T_("i")} / ${u(dims[1])};`, 3);
      P(`let ${T_("cc")} = ${T_("i")} % ${u(dims[1])};`, 3);
      const coord = (axis: string, local: string) =>
        axis === contract.name ? `${T_("cb")} + ${local}` : `base_${axis} + ${local}`;
      P(`let ${T_("gr")} = ${coord(bd.axes[0], T_("r"))};`, 3);
      P(`let ${T_("gc")} = ${coord(bd.axes[1], T_("cc"))};`, 3);
      emitLoad(`${stageName(o.binding)}[${T_("i")}]`, bd, T_("gr"), T_("gc"), 3);
      P(`}`, 2);
    }
    if (staged.length) { P(`workgroupBarrier();`, 2); P(); }

    // The part of a staged address that does not move with the contract step is
    // loop-invariant by construction — it is the invocation's own slice — so it is
    // computed once rather than once per step per operand.
    const hoisted = new Map<string, string>();
    for (const o of staged) {
      const bd = byName(o.binding);
      const dims = bd.axes.map((a) => axisOf.get(a)!.block);
      const accAxis = bd.axes.find((a) => sliceOf.has(a))!;
      const sl = sliceOf.get(accAxis)!;
      const lane = sl.lane === "x" ? "tx" : "ty";
      for (let i = 0; i < sl.slice; i++) {
        const nm = T_(`base_${o.binding}_${i}`);
        // Row-major: the accumulate axis is either the row or the column of the
        // staged block, and which one decides what the invariant term looks like.
        hoisted.set(`${o.binding}_${i}`, nm);
        P(bd.axes[0] === accAxis
          ? `let ${nm} = (${lane} * ${u(sl.slice)} + ${u(i)}) * ${u(dims[1])};`
          : `let ${nm} = ${lane} * ${u(sl.slice)} + ${u(i)};`, 2);
      }
    }
    P(`for (var ${T_("s")} : u32 = 0u; ${T_("s")} < ${u(seqDepth)}; ${T_("s")} = ${T_("s")} + 1u) {`, 2);
    // The contract coordinate this invocation handles at this step. When lanes split
    // the contraction each lane takes a contiguous run, which keeps the reads coalesced.
    const laneId = plan.lanes.contractLanes.includes("x") ? "tx" : "ty";
    // Two coordinates, not one. `lc` is the offset INSIDE the contract block —
    // what a staged read wants — and `ci` is the global index a direct read wants.
    // Deriving the first from the second meant emitting `ci - cb`, which is an add
    // and a subtract that cancel, repeated once per staged operand slice, and which
    // stops the address being an obvious affine function of the loop variable.
    P(plan.lanes.needsCombine
      ? `let ${T_("lc")} = ${laneId} * ${u(seqDepth)} + ${T_("s")};`
      : `let ${T_("lc")} = ${T_("s")};`, 3);
    P(`let ${T_("ci")} = ${T_("cb")} + ${T_("lc")};`, 3);

    // One read per operand per combination of the accumulate axes it mentions.
    for (const o of plan.operands) {
      const bd = byName(o.binding);
      const ownAxes = bd.axes.filter((a) => sliceOf.has(a));
      const combos: Record<string, number>[] = [{}];
      for (const a of ownAxes) {
        const n: Record<string, number>[] = [];
        for (const c of combos) for (let i = 0; i < sliceOf.get(a)!.slice; i++) n.push({ ...c, [a]: i });
        combos.length = 0; combos.push(...n);
      }
      for (const c of combos) {
        const name = operandVal(o.binding, c);
        if (o.staged) {
          const dims = bd.axes.map((a) => axisOf.get(a)!.block);
          const accAxis = bd.axes.find((a) => sliceOf.has(a))!;
          const base = hoisted.get(`${o.binding}_${c[accAxis]}`)!;
          // base already carries the invariant term; the step contributes the rest.
          P(bd.axes[0] === accAxis
            ? `let ${name} = ${stageName(o.binding)}[${base} + ${T_("lc")}];`
            : `let ${name} = ${stageName(o.binding)}[${T_("lc")} * ${u(dims[1])} + ${base}];`, 4);
        } else {
          const g0 = bd.axes[0] === contract.name ? T_("ci") : accCoord(bd.axes[0], c[bd.axes[0]]);
          const g1 = bd.axes[1] === contract.name ? T_("ci") : accCoord(bd.axes[1], c[bd.axes[1]]);
          emitLoad(`let ${name}`, bd, `(${g0})`, `(${g1})`, 4);
        }
      }
    }
    // Update every fragment cell. Each cell picks the operand value belonging to
    // its own slice indices, projected onto that operand's axes — which is what
    // makes an outer product and a fold the same code.
    for (const c of cells) {
      const i = cellIndex(c);
      for (const up of updates) {
        if (up.k === "mma") {
          P(`${up.acc}[${i}] = ${up.acc}[${i}] + ` +
            `${blockExpr(up.a, c)} * ${blockExpr(up.b, c)};`, 4);
        } else {
          const v = blockExpr(up.value, c);
          P(up.op === "max"
            ? `${up.acc}[${i}] = max(${up.acc}[${i}], ${v});`
            : `${up.acc}[${i}] = ${up.acc}[${i}] + ${v};`, 4);
        }
      }
    }
    P(`}`, 2);
    if (staged.length) P(`workgroupBarrier();`, 2);
    P(`}`, 1);
    P();

    if (plan.lanes.needsCombine) emitCombine(updates);
  }

  /**
   * Combine the lane partials. Needed exactly when the contract axis was given a
   * lane dimension, which the plan decided and this does not second-guess.
   */
  function emitCombine(updates: readonly Update[]): void {
    const accs = [...new Set(updates.map((x) => x.acc))];
    // The lane dimension the contraction was split across, and the other one.
    const dim = plan.lanes.contractLanes.includes("x") ? "x" : "y";
    const laneId = dim === "x" ? "tx" : "ty";
    const other = dim === "x" ? "ty" : "tx";
    const width = laneW[dim];
    for (const nm of accs) {
      const u0 = updates.find((x) => x.acc === nm)!;
      const op = u0.k === "fold" ? u0.op : "sum";
      const init = op === "max" ? PAD_LITERAL.negInf : "0.0";
      for (let c = 0; c < frag; c++) {
        P(`scratch[(${other} * ${u(frag)} + ${u(c)}) * ${u(width)} + ${laneId}] = ${nm}[${c}];`, 1);
      }
      P(`workgroupBarrier();`, 1);
      for (let c = 0; c < frag; c++) {
        P(`{`, 1);
        P(`var ${T_("r")} = ${init};`, 2);
        P(`for (var ${T_("j")} : u32 = 0u; ${T_("j")} < ${u(width)}; ${T_("j")} = ${T_("j")} + 1u) {`, 2);
        const cell = `scratch[(${other} * ${u(frag)} + ${u(c)}) * ${u(width)} + ${T_("j")}]`;
        P(op === "max" ? `${T_("r")} = max(${T_("r")}, ${cell});` : `${T_("r")} = ${T_("r")} + ${cell};`, 3);
        P(`}`, 2);
        P(`${nm}[${c}] = ${T_("r")};`, 2);
        P(`}`, 1);
      }
      P(`workgroupBarrier();`, 1);
    }
    P();
  }

  /** A block-valued expression at one fragment cell. */
  function blockExpr(e: Expr, c: Record<string, number>): string {
    switch (e.k) {
      case "tile":  return operandVal(e.binding, c);
      case "unary": return e.op === "sq"
        ? `(${blockExpr(e.a, c)} * ${blockExpr(e.a, c)})`
        : `exp(${blockExpr(e.a, c)})`;
      case "rowOp": {
        const a = blockExpr(e.a, c), r = rowExpr(e.row, c);
        return e.op === "sub" ? `(${a} - ${r})` : e.op === "mul" ? `(${a} * ${r})` : `(${a} / ${r})`;
      }
    }
  }
  function rowExpr(r: RowExpr, c: Record<string, number>): string {
    switch (r.k) {
      case "acc":  return `${r.name}[${cellIndex(c)}]`;
      case "mean": return `(${rowExpr(r.a, c)} / ${contract.extent}.0)`;
      case "rstd": {
        const mu = rowExpr(r.mean, c);
        return `inverseSqrt((${rowExpr(r.sumSq, c)} / ${contract.extent}.0) - ${mu} * ${mu} + ${r.eps})`;
      }
    }
  }
  function fragExpr(f: FragExpr, c: Record<string, number>): string {
    return f.k === "acc" ? `${f.name}[${cellIndex(c)}]`
      : `select(0.0, ${fragExpr(f.a, c)}, ${fragExpr(f.a, c)} > 0.0)`;
  }

  // ---- derived row values, then the store ------------------------------------
  for (const st of body.steps) {
    if (st.k !== "derived") continue;
    P(`var ${st.name} : array<${ty}, ${frag}>;`, 1);
    for (const c of cells) P(`${st.name}[${cellIndex(c)}] = ${rowExpr(st.expr, c)};`, 1);
    P();
  }

  // Whether the store walks the contraction is a property of the output's axes:
  // if the output is indexed BY the contract axis it must, and if it is not it
  // writes the fragment once. Nobody chooses this either.
  const storeWalksContraction = out.axes.includes(contract.name);
  const storeStep = body.steps.find((x) => x.k === "store" || x.k === "storeFrag");
  if (!storeStep) throw new Error(`${k.name}: no store`);

  if (storeWalksContraction && storeStep.k === "store") {
    P(`for (var ${T_("cb")} : u32 = 0u; ${T_("cb")} < ${u(contract.extent)}; ${T_("cb")} = ${T_("cb")} + ${u(contractBlock)}) {`, 1);
    // The part of a staged address that does not move with the contract step is
    // loop-invariant by construction — it is the invocation's own slice — so it is
    // computed once rather than once per step per operand.
    const hoisted = new Map<string, string>();
    for (const o of staged) {
      const bd = byName(o.binding);
      const dims = bd.axes.map((a) => axisOf.get(a)!.block);
      const accAxis = bd.axes.find((a) => sliceOf.has(a))!;
      const sl = sliceOf.get(accAxis)!;
      const lane = sl.lane === "x" ? "tx" : "ty";
      for (let i = 0; i < sl.slice; i++) {
        const nm = T_(`base_${o.binding}_${i}`);
        // Row-major: the accumulate axis is either the row or the column of the
        // staged block, and which one decides what the invariant term looks like.
        hoisted.set(`${o.binding}_${i}`, nm);
        P(bd.axes[0] === accAxis
          ? `let ${nm} = (${lane} * ${u(sl.slice)} + ${u(i)}) * ${u(dims[1])};`
          : `let ${nm} = ${lane} * ${u(sl.slice)} + ${u(i)};`, 2);
      }
    }
    P(`for (var ${T_("s")} : u32 = 0u; ${T_("s")} < ${u(seqDepth)}; ${T_("s")} = ${T_("s")} + 1u) {`, 2);
    const laneId = plan.lanes.contractLanes.includes("x") ? "tx" : "ty";
    // Two coordinates, not one. `lc` is the offset INSIDE the contract block —
    // what a staged read wants — and `ci` is the global index a direct read wants.
    // Deriving the first from the second meant emitting `ci - cb`, which is an add
    // and a subtract that cancel, repeated once per staged operand slice, and which
    // stops the address being an obvious affine function of the loop variable.
    P(plan.lanes.needsCombine
      ? `let ${T_("lc")} = ${laneId} * ${u(seqDepth)} + ${T_("s")};`
      : `let ${T_("lc")} = ${T_("s")};`, 3);
    P(`let ${T_("ci")} = ${T_("cb")} + ${T_("lc")};`, 3);
    for (const o of plan.operands) {
      const bd = byName(o.binding);
      for (const c of cells) {
        const g0 = bd.axes[0] === contract.name ? T_("ci") : accCoord(bd.axes[0], c[bd.axes[0]]);
        const g1 = bd.axes[1] === contract.name ? T_("ci") : accCoord(bd.axes[1], c[bd.axes[1]]);
        emitLoad(`let ${operandVal(o.binding, c)}`, bd, `(${g0})`, `(${g1})`, 4);
      }
    }
    for (const c of cells) {
      const g0 = out.axes[0] === contract.name ? T_("ci") : accCoord(out.axes[0], c[out.axes[0]]);
      const g1 = out.axes[1] === contract.name ? T_("ci") : accCoord(out.axes[1], c[out.axes[1]]);
      emitStore(out, `(${g0})`, `(${g1})`, blockExpr(storeStep.value, c), 4);
    }
    P(`}`, 2);
    P(`}`, 1);
  } else if (storeStep.k === "storeFrag") {
    for (const c of cells) {
      emitStore(out, `(${accCoord(out.axes[0], c[out.axes[0]])})`,
                `(${accCoord(out.axes[1], c[out.axes[1]])})`, fragExpr(storeStep.value, c), 1);
    }
  } else {
    throw new Error(`${k.name}: the store's form and the output's axes disagree`);
  }

  P(`}`);
  return L.join("\n") + "\n";
}
