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
import type {
  Expr, FragExpr, RowExpr, Update, Step,
} from "./body.ts";

export function emitWGSL(k: KernelIR): string {
  const [wgx, wgy] = k.workgroup;
  const ty = k.dtype;
  const body = k.body;
  if (!body) throw new Error(`${k.name}: no parsed body`);

  const accumulate = k.grid;
  const contract = k.reduce[0];
  const out = k.bindings.find((b) => b.mode === "write")!;
  const reads = k.bindings.filter((b) => b.mode === "read");

  /** The binding as MEMORY has it: `axes` in storage order. */
  const memOf = (n: string) => k.bindings.find((x) => x.name === n)!;
  /**
   * The binding as the TILE has it. `.tileT()` swaps the two axes for every
   * purpose except the address computation — staging dimensions, which axis the
   * contraction runs along, which axis is ragged for the mask. `emitLoad` is the
   * single place that swaps back, because it is the only one that indexes the
   * buffer.
   */
  const logical = (b: BindingIR): BindingIR =>
    b.transposed ? { ...b, axes: [b.axes[1], b.axes[0]] as const } : b;
  const byName = (n: string) => logical(memOf(n));

  /**
   * One plan per reduction axis.
   *
   * A pass over `n` and a pass over `k` stage different operands and walk
   * different extents, but they share their LANE geometry: `planContraction`
   * derives `perAxis`, `contractLanes` and `fragment` from `accumulate` and the
   * contiguous axis alone, and neither depends on which axis is contracted. That
   * is what makes several contractions in one body tractable without touching the
   * register layout — and it is checked below rather than assumed, because the
   * whole emitter reads the shared geometry off one of them.
   */
  /**
   * Which bindings a pass actually reads.
   *
   * With one contraction per kernel every read binding took part in it, so the
   * plan could be built from all of them. With two, a pass over `n` may read only
   * `p` while a pass over `k` reads only `q` — and handing a pass an operand whose
   * axes it does not walk asks for the global coordinate of an axis that is
   * neither accumulated nor contracted here, which has no meaning.
   */
  const readsOf = (step: Step): string[] => {
    const out2 = new Set<string>();
    const walk = (e: Expr): void => {
      if (e.k === "tile") { out2.add(e.binding); return; }
      if (e.k === "unary" || e.k === "rowOp") return walk(e.a);
    };
    if (step.k === "store" && !step.fromFrag) walk(step.value as Expr);
    if (step.k === "reduce") {
      for (const upd of step.updates) {
        if (upd.k === "fold") walk(upd.value);
        if (upd.k === "mma") { walk(upd.a); walk(upd.b); }
      }
    }
    return [...out2];
  };
  const readsByAxis = new Map<string, Set<string>>();
  for (const st of body.steps) {
    if (st.k !== "reduce" && st.k !== "store") continue;
    const set = readsByAxis.get(st.axis) ?? new Set<string>();
    for (const b of readsOf(st)) set.add(b);
    readsByAxis.set(st.axis, set);
  }
  /**
   * The plan for a pass, given the axes whose loops enclose it.
   *
   * An inner contraction accumulates over the grid PLUS those axes: they are
   * fixed while it runs, so they are free to it. `for n { for k { … } }` gives the
   * inner pass accumulate (m, n) — a matmul's shape — where the outer one has (m).
   */
  const planWith = (ax: AxisIR, enclosing: readonly AxisIR[], names: ReadonlySet<string>) =>
    planContraction(
      // Minus the axis being contracted. An enclosing loop's axis is free to the
      // pass inside it — unless that pass is the one contracting it. Attention's
      // inner contraction runs over `d`, which is also a grid axis because the
      // output is indexed by it, and accumulating over it there would ask for
      // three lanes from a workgroup that has two.
      [...accumulate, ...enclosing].filter((a2) => a2.name !== ax.name), ax,
      reads.filter((b) => names.has(b.name)).map((b) => ({ binding: b.name, axes: logical(b).axes })),
      k.workgroup,
      out.axes[1],
    );

  // Only the axes reduced at the TOP level get a plan from the kernel's accumulate
  // set. A nested axis is never contracted against that set — its pass sees the
  // enclosing loops as accumulate axes too — so building one for it would ask for
  // the coordinate of an axis nothing there walks. That was the first wall the
  // nested probe hit: `operand "b" mentions unknown axis "n"`.
  const topAxes = new Set(body.steps
    .filter((st) => st.k === "reduce" || st.k === "store")
    .map((st) => (st as Extract<Step, { k: "reduce" }>).axis));
  const planFor = new Map(k.reduce.filter((ax) => topAxes.has(ax.name)).map((ax) => {
    const names = readsByAxis.get(ax.name) ?? new Set(reads.map((b) => b.name));
    return [ax.name, planContraction(
      accumulate, ax,
      reads.filter((b) => names.has(b.name)).map((b) => ({ binding: b.name, axes: logical(b).axes })),
      k.workgroup,
      out.axes[1],                    // the output's last index is contiguous
    )] as const;
  }));
  const plan = planFor.get(contract.name)!;
  for (const [ax, p2] of planFor) {
    const same = JSON.stringify(p2.lanes.perAxis) === JSON.stringify(plan.lanes.perAxis)
              && p2.lanes.fragment === plan.lanes.fragment
              && p2.lanes.contractLanes.join() === plan.lanes.contractLanes.join();
    if (!same) {
      throw new Error(
        `${k.name}: the plan for reduction axis "${ax}" has a different lane layout ` +
        `from "${contract.name}". Contractions in one body must share it; ` +
        `redistributing a fragment between layouts is not implemented.`);
    }
  }
  const axisIR = (n: string) => k.reduce.find((a) => a.name === n)!;

  const L: string[] = [];
  const P = (s = "", i = 0) => L.push(s ? "  ".repeat(i) + s : "");
  const u = (n: number) => `${n}u`;
  /**
   * The identity for THIS binding's out-of-range lanes. Per binding, because one
   * kernel can want two — `zero` where a masked lane must annihilate in a sum,
   * `negInf` where it must not bias a maximum toward itself. Falls back to the
   * kernel-level value for a binding that names none, which happens only when it
   * touches no ragged axis and so has no masked lane to fill.
   */
  const padOf = (bd: BindingIR) => PAD_LITERAL[bd.pad ?? k.pad];
  const ragged = (a: AxisIR) => a.fit === "ragged";
  const axisOf = new Map([...k.grid, ...k.reduce].map((x) => [x.name, x]));
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
    // `bd` arrives in TILE order; the buffer is in memory order. For a `.tileT()`
    // read those differ by exactly a swap of the two indices, which is why the
    // access layer needed nothing else — it already took both indices from its
    // caller rather than deriving them.
    const mem = memOf(bd.name);
    const [m0, m1] = mem.transposed ? [i1, i0] : [i0, i1];
    const off = `off${offN++}`;
    P(`let ${off} = ${m0} * ${u(stride(mem))} + ${m1};`, ind);
    const g = guards(mem, m0, m1);
    P(g.length === 0
      ? `${lhs} = ${bd.name}[${off}];`
      // Clamp then select: branchless. A branch would diverge for exactly one
      // block per row.
      : `${lhs} = select(${padOf(mem)}, ${bd.name}[min(${off}, ${u(bd.elements - 1)})], ${g.join(" && ")});`, ind);
  };
  const emitStore = (bd: BindingIR, i0: string, i1: string, val: string, ind: number) => {
    const g = guards(bd, i0, i1);
    const w = `${bd.name}[${i0} * ${u(stride(bd))} + ${i1}] = ${val};`;
    if (!g.length) { P(w, ind); return; }
    // A store cannot be clamped: a legal-but-wrong address corrupts a real element.
    P(`if (${g.join(" && ")}) {`, ind); P(w, ind + 1); P(`}`, ind);
  };

  // ---- geometry from a plan --------------------------------------------------
  /**
   * Everything the emitter derives from one contraction's plan.
   *
   * This used to be a block of module-level constants, which was right while a
   * kernel had exactly one contraction. A nested one has its own: an inner
   * contraction accumulates over the grid PLUS every axis whose loop encloses it,
   * because those are fixed while it runs. In `for n { for k { … } }` the inner
   * pass accumulates (m, n) and the outer one (m), so their fragments are
   * different shapes — 4x4 against 4 — and everything below follows from that.
   */
  function geomOf(gplan: ContractionPlan, cax: AxisIR, loopVar: string) {
    const slices = gplan.lanes.perAxis;                    // accumulate axis -> lane, slice
    const sliceOf = new Map(slices.map((s) => [s.axis, s]));
    const frag = gplan.lanes.fragment;
    const contractLaneW = gplan.lanes.contractLanes.reduce((n, l) => n * laneW[l], 1);
    const seqDepth = cax.block / contractLaneW;

    /** Every fragment cell, as an assignment of a slice index to each accumulate axis. */
    const cells: Record<string, number>[] = [{}];
    for (const sl of slices) {
      const next: Record<string, number>[] = [];
      for (const c of cells) for (let i = 0; i < sl.slice; i++) next.push({ ...c, [sl.axis]: i });
      cells.length = 0; cells.push(...next);
    }
    const cellIndex = (c: Record<string, number>) =>
      slices.reduce((n, sl) => n * sl.slice + c[sl.axis], 0);
    /** The value name an operand contributes at this cell — projected onto its own axes. */
    const operandVal = (bindingName: string, c: Record<string, number>) => {
      const bd = byName(bindingName);
      const own = bd.axes.filter((a) => sliceOf.has(a)).map((a) => c[a]);
      return `v_${bindingName}${own.map((i) => `_${i}`).join("")}`;
    };
    /** Global coordinate of an accumulate axis at slice index i. */
    const accCoord = (axis: string, i: number) => {
      const sl = sliceOf.get(axis)!;
      return `base_${axis} + ${sl.lane === "x" ? "tx" : "ty"} * ${u(sl.slice)} + ${u(i)}`;
    };
    return {
      plan: gplan, cax, loopVar, slices, sliceOf, frag, contractLaneW, seqDepth,
      contractBlock: cax.block, staged: gplan.operands.filter((o) => o.staged),
      cells, cellIndex, operandVal, accCoord,
    };
  }
  type Geom = ReturnType<typeof geomOf>;

  const G = geomOf(plan, contract, T_("cb"));
  const { slices, sliceOf, frag, contractLaneW, contractBlock, seqDepth, staged,
          cells, cellIndex, operandVal, accCoord } = G;

  /**
   * Workgroup storage, against the 16384 B every WebGPU implementation must
   * offer. The harness has always checked this at run time and the compiler has
   * only reported it; a kernel that cannot be dispatched should not compile.
   *
   * It is not hypothetical. Attention with this project's usual 64x64x16 tile
   * needs 28672 B — staging q, k, v AND the score fragment it has to redistribute
   * — so the tile has to shrink before the kernel is expressible at all.
   */
  let wgFloats = 0;

  // ---- header ---------------------------------------------------------------
  P(`// Generated by tessera from a TypeScript kernel. Do not edit.`);
  P(`//`);
  P(`//   ${k.name}   accumulate (${plan.accumulate.join(", ")})   contract ${plan.contract}`);
  P(`//   lanes ${slices.map((s) => `${s.axis}->${s.lane}(${s.slice})`).join(" ")}` +
    `${plan.lanes.contractLanes.length ? `  ${plan.contract}->${plan.lanes.contractLanes.join("")}` : ""}` +
    `   fragment ${frag}   ${seqDepth} sequential steps`);
  P(`//   staged: ${staged.map((o) => o.binding).join(", ") || "nothing — no operand is reused"}` +
    `   workgroup ${plan.workgroupElements * 4} B`);
  const padNote = k.bindings.filter((b) => b.pad).map((b) => `${b.name}=${b.pad}`).join(" ");
  P(`//   ${k.maskedLoads.length ? `masked: ${k.maskedLoads.join(" ")}   pad ${padNote || k.pad}` : "no masks: every axis divides its block"}`);
  P();

  for (const [i, bd] of k.bindings.entries()) {
    P(`@group(0) @binding(${i}) var<storage, ${bd.mode === "read" ? "read" : "read_write"}> ${bd.name} : array<${ty}>;`);
  }
  P();
  // Every pass's staged operands, nested ones included. The outer pass of a
  // nested kernel may stage nothing at all while the inner one stages both its
  // operands, which is how `stage_a` came to be read without being declared.
  {
    const decls = new Map<string, number>();
    const collect = (st: Step, enclosing: readonly AxisIR[]): void => {
      if (st.k !== "reduce" && st.k !== "store") return;
      const names = new Set(readsOf(st));
      for (const inner of st.inner) for (const b of readsOf(inner)) names.add(b);
      const pl = enclosing.length ? planWith(axisIR(st.axis), enclosing, names)
                                  : planFor.get(st.axis)!;
      for (const o of pl.operands) {
        if (o.staged) decls.set(o.binding, Math.max(decls.get(o.binding) ?? 0, o.stagedElements));
      }
      for (const inner of st.inner) collect(inner, [...enclosing, axisIR(st.axis)]);
    };
    for (const st of body.steps) collect(st, []);
    for (const [b, n] of decls) P(`var<workgroup> ${stageName(b)} : array<${ty}, ${n}>;`);
    wgFloats += [...decls.values()].reduce((n2, x) => n2 + x, 0);
  }
  const reducesFrag = body.steps.some((x) => x.k === "derived" && x.expr.k === "reduceFrag");
  if (reducesFrag) {
    // rows-in-the-tile x lanes-along-the-reduced-axis. Staged operands keep their
    // own buffers; this is additional, and the two together must still fit the
    // 16384 B floor.
    const [row, red] = slices;
    wgFloats += laneW[row.lane] * row.slice * laneW[red.lane];
    P(`var<workgroup> scratch : array<${ty}, ${laneW[row.lane] * row.slice * laneW[red.lane]}>;` +
      `   // tile rows x lanes along ${red.axis}`);
  } else if (!staged.length && plan.lanes.needsCombine) {
    wgFloats += plan.workgroupElements;
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

  {
    // 16384 B is the floor every WebGPU implementation must offer, so it is the
    // number to compile against rather than whatever this machine happens to
    // allow. A kernel over it cannot be dispatched anywhere.
    const bytes = wgFloats * (k.dtype === "f16" ? 2 : 4);
    if (bytes > 16384) {
      throw new Error(
        `${k.name}: ${bytes} B of workgroup storage, over the 16384 B floor every ` +
        `WebGPU implementation guarantees. Shrink the tile — the staged operands ` +
        `and any redistributed fragment all live at once.`);
    }
  }

  // ---- accumulators ----------------------------------------------------------
  for (const acc of body.accs) {
    // A row accumulator holds one value per ROW cell; a fragment holds one per
    // cell of the whole 2-D fragment. Those were the same number while a body had
    // one kind of accumulator, which is why this could use `frag` for both.
    const n = acc.kind === "row" ? slices[0].slice : frag;
    P(`var ${acc.name} : array<${ty}, ${n}>;`, 1);
    const init = PAD_LITERAL[acc.init];
    for (let c = 0; c < n; c++) P(`${acc.name}[${c}] = ${init};`, 1);
  }
  P();

  // ---- the contraction --------------------------------------------------------
  const reduceSteps = body.steps.filter((s) => s.k === "reduce");
  for (const step of reduceSteps) emitPass(step, []);
  // A store pass that ran an inner contraction is emitted here; a flat one is
  // handled by the store loop below, which is where it has always been.
  for (const step of body.steps) {
    if (step.k === "store" && (step.locals.length || step.inner.length)) emitPass(step, []);
  }

  /**
   * One pass: its local accumulators, the passes nested inside it, and its own
   * contraction loop.
   *
   * `enclosing` is the axes whose loops are already open. They are fixed while
   * this pass runs, so they are ACCUMULATE axes to it — which is the whole reason
   * an inner contraction has a different fragment shape from the outer one, and
   * why the plan cannot be built once per axis for the kernel.
   */
  function emitPass(step: Extract<Step, { k: "reduce" } | { k: "store" }>,
                    enclosing: readonly AxisIR[]): void {
    const cx = axisIR(step.axis);
    const names = new Set(readsOf(step));
    for (const inner of step.inner) for (const b of readsOf(inner)) names.add(b);
    const cplan = enclosing.length ? planWith(cx, enclosing, names) : planFor.get(step.axis)!;
    const g = geomOf(cplan, cx, enclosing.length ? T_(`cb_${step.axis}`) : T_("cb"));
    const staged = g.staged;
    const cBlock = cx.block;
    const updates = step.k === "reduce" ? step.updates : [];

    // Locals and nested passes come FIRST, and outside this pass's own loop is
    // wrong — the inner contraction has to run once per step of this one, since
    // its fragment is the scores for this block and nothing else.
    if (step.locals.length || step.inner.length) {
      P(`for (var ${g.loopVar} : u32 = 0u; ${g.loopVar} < ${u(cx.extent)}; ` +
        `${g.loopVar} = ${g.loopVar} + ${u(cBlock)}) {`, 1);
      // The enclosing axis is fixed for the nested passes, so it is an accumulate
      // axis to them and needs a base: this loop's own step.
      P(`let base_${cx.name} = ${g.loopVar};`, 2);
      const inGeom = geomOf(
        planWith(axisIR(step.inner[0].axis), [...enclosing, cx],
                 new Set(step.inner.flatMap((x) => readsOf(x)))),
        axisIR(step.inner[0].axis), T_(`cb_${step.inner[0].axis}`));
      for (const loc of step.locals) {
        P(`var ${loc.name} : array<${ty}, ${inGeom.frag}>;`, 2);
        for (let c = 0; c < inGeom.frag; c++) P(`${loc.name}[${c}] = ${PAD_LITERAL[loc.init]};`, 2);
      }
      for (const inner of step.inner) emitPass(inner, [...enclosing, cx]);
      emitOuterUpdates(step, g, inGeom);
      P(`}`, 1);
      P();
      return;
    }
    // How many sequential steps this invocation walks for THIS axis. Derived
    // from the pass's own block and the lanes the contraction was split across.
    const cDepth = g.seqDepth;
    P(`for (var ${g.loopVar} : u32 = 0u; ${g.loopVar} < ${u(cx.extent)}; ${g.loopVar} = ${g.loopVar} + ${u(cBlock)}) {`, 1);

    for (const o of staged) {
      const bd = byName(o.binding);
      const dims = bd.axes.map((a) => axisOf.get(a)!.block);
      P(`for (var ${T_("i")} : u32 = tid; ${T_("i")} < ${u(o.stagedElements)}; ${T_("i")} = ${T_("i")} + ${u(wgx * wgy)}) {`, 2);
      P(`let ${T_("r")} = ${T_("i")} / ${u(dims[1])};`, 3);
      P(`let ${T_("cc")} = ${T_("i")} % ${u(dims[1])};`, 3);
      const coord = (axis: string, local: string) =>
        axis === cx.name ? `${g.loopVar} + ${local}` : `base_${axis} + ${local}`;
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
      const accAxis = bd.axes.find((a) => g.sliceOf.has(a))!;
      const sl = g.sliceOf.get(accAxis)!;
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
    P(`for (var ${T_("s")} : u32 = 0u; ${T_("s")} < ${u(cDepth)}; ${T_("s")} = ${T_("s")} + 1u) {`, 2);
    // The contract coordinate this invocation handles at this step. When lanes split
    // the contraction each lane takes a contiguous run, which keeps the reads coalesced.
    const laneId = cplan.lanes.contractLanes.includes("x") ? "tx" : "ty";
    // Two coordinates, not one. `lc` is the offset INSIDE the contract block —
    // what a staged read wants — and `ci` is the global index a direct read wants.
    // Deriving the first from the second meant emitting `ci - cb`, which is an add
    // and a subtract that cancel, repeated once per staged operand slice, and which
    // stops the address being an obvious affine function of the loop variable.
    P(cplan.lanes.needsCombine
      ? `let ${T_("lc")} = ${laneId} * ${u(cDepth)} + ${T_("s")};`
      : `let ${T_("lc")} = ${T_("s")};`, 3);
    P(`let ${T_("ci")} = ${g.loopVar} + ${T_("lc")};`, 3);

    // One read per operand per combination of the accumulate axes it mentions.
    for (const o of cplan.operands) {
      const bd = byName(o.binding);
      const ownAxes = bd.axes.filter((a) => g.sliceOf.has(a));
      const combos: Record<string, number>[] = [{}];
      for (const a of ownAxes) {
        const n: Record<string, number>[] = [];
        for (const c of combos) for (let i = 0; i < g.sliceOf.get(a)!.slice; i++) n.push({ ...c, [a]: i });
        combos.length = 0; combos.push(...n);
      }
      for (const c of combos) {
        const name = g.operandVal(o.binding, c);
        if (o.staged) {
          const dims = bd.axes.map((a) => axisOf.get(a)!.block);
          const accAxis = bd.axes.find((a) => g.sliceOf.has(a))!;
          const base = hoisted.get(`${o.binding}_${c[accAxis]}`)!;
          // base already carries the invariant term; the step contributes the rest.
          P(bd.axes[0] === accAxis
            ? `let ${name} = ${stageName(o.binding)}[${base} + ${T_("lc")}];`
            : `let ${name} = ${stageName(o.binding)}[${T_("lc")} * ${u(dims[1])} + ${base}];`, 4);
        } else {
          const g0 = bd.axes[0] === cx.name ? T_("ci") : g.accCoord(bd.axes[0], c[bd.axes[0]]);
          const g1 = bd.axes[1] === cx.name ? T_("ci") : g.accCoord(bd.axes[1], c[bd.axes[1]]);
          emitLoad(`let ${name}`, bd, `(${g0})`, `(${g1})`, 4);
        }
      }
    }
    // Update every fragment cell. Each cell picks the operand value belonging to
    // its own slice indices, projected onto that operand's axes — which is what
    // makes an outer product and a fold the same code.
    for (const c of g.cells) {
      const i = g.cellIndex(c);
      for (const up of updates) {
        if (up.k === "mma") {
          P(`${up.acc}[${i}] = ${up.acc}[${i}] + ` +
            `${blockExpr(up.a, c, g)} * ${blockExpr(up.b, c, g)};`, 4);
        } else {
          // A tile fold reads one block per contraction step; a fragment fold
          // reads a value already whole in registers. Inside a contraction loop
          // only the first can occur — a fragment fold belongs to the OUTER pass,
          // after its inner contraction has finished.
          if (up.k === "foldFrag") {
            throw new Error(`${k.name}: a fragment fold inside a contraction loop; `
              + `the fragment is not complete until the loop ends.`);
          }
          if (up.k === "mmaFrag") {
            // THE REDISTRIBUTION, and what is left of G4. `P` is laid out by the
            // contraction that built it, with the axis this one sums along spread
            // across the lanes — four of the sixty-four values per invocation.
            // Reading it in place would sum a quarter of each row.
            //
            // The mechanism is to stage it through workgroup memory, exactly as an
            // operand read from a buffer is staged, with registers as the source.
            // What makes it more than plumbing is the budget: attention needs q, k,
            // v AND the staged fragment alive at once, which is 28672 B at this
            // project's usual 64x64x16 tile against a 16384 B floor. The tile has
            // to shrink to 32x32x16 before the kernel is expressible at all.
            throw new Error(
              `${k.name}: contracting a fragment along an axis it holds on a lane. ` +
              `The fragment has to be staged through workgroup memory first, which ` +
              `is not implemented. See docs/004 R12, G4.`);
          }
          const v = blockExpr(up.value, c, g);
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

    if (cplan.lanes.needsCombine) emitCombine(updates);
  }

  /**
   * Combine the lane partials. Needed exactly when the contract axis was given a
   * lane dimension, which the plan decided and this does not second-guess.
   */
  /**
   * A fragment expression indexed by the INNER geometry.
   *
   * `fragExpr` closes over the outer cellIndex, which is the wrong one for a
   * fragment the inner contraction laid out. Elementwise forms recurse; a row
   * operation does not, because its row value is indexed by the outer geometry
   * and mixing the two silently is exactly the class of bug this emitter keeps
   * being rewritten to avoid.
   */
  function innerFrag(f: FragExpr, c: Record<string, number>, inG: Geom): string {
    switch (f.k) {
      case "acc":   return `${f.name}[${inG.cellIndex(c)}]`;
      case "map":   return `select(0.0, ${innerFrag(f.a, c, inG)}, ${innerFrag(f.a, c, inG)} > 0.0)`;
      case "unary": {
        const a2 = innerFrag(f.a, c, inG);
        return f.op === "exp" ? `exp(${a2})` : `(${a2} * ${a2})`;
      }
      case "rowOp": {
        // The row value is indexed by the outer geometry and the fragment by the
        // inner one, which sounds like it needs a redistribution and does not:
        // `rowExpr` indexes a row by its ROW-axis cell, and emitOuterUpdates has
        // already refused the case where that axis is laid out differently in the
        // two. So the same cell record answers both.
        const a2 = innerFrag(f.a, c, inG), r = rowExpr(f.row, c);
        return f.op === "sub" ? `(${a2} - ${r})` : f.op === "mul" ? `(${a2} * ${r})` : `(${a2} / ${r})`;
      }
    }
  }

  /**
   * What the outer pass does with the fragment its inner contraction built.
   *
   * The fragment is laid out by the INNER geometry — `(m, n)` on `(y, x)` — and
   * the accumulator by the outer one, `(m)` on `(y)`. Folding along n therefore
   * crosses the x lanes, the same shape as emitReduceFrag, and lands one value
   * per outer cell. That the surviving axis keeps its lane and slice in both is
   * what makes this a fold rather than a redistribution.
   */
  function emitOuterUpdates(step: Extract<Step, { k: "reduce" } | { k: "store" }>,
                            g: Geom, inG: Geom): void {
    const [row, red] = inG.slices;
    if (step.k === "store") {
      // Writing the inner fragment out. No cross-lane anything: the inner
      // geometry already gives both coordinates — the row axis from the grid, the
      // reduced axis from this loop's step plus the invocation's own lane — so
      // each cell knows where it goes.
      if (!step.fromFrag) {
        throw new Error(
          `${k.name}: a pass with a nested contraction stores something other than ` +
          `its fragment`);
      }
      const bd = byName(step.binding);
      for (const c of inG.cells) {
        emitStore(bd, `(${inG.accCoord(row.axis, c[row.axis])})`,
                  `(${inG.accCoord(red.axis, c[red.axis])})`,
                  innerFrag(step.value as FragExpr, c, inG), 2);
      }
      P();
      return;
    }
    if (row.axis !== g.slices[0].axis || row.lane !== g.slices[0].lane
        || row.slice !== g.slices[0].slice) {
      throw new Error(
        `${k.name}: the inner contraction lays out "${row.axis}" as ` +
        `${row.lane}(${row.slice}) where the outer one has ` +
        `${g.slices[0].lane}(${g.slices[0].slice}). Redistributing a fragment ` +
        `between layouts is not implemented.`);
    }
    const width = laneW[red.lane];
    const redLane = red.lane === "x" ? "tx" : "ty";
    const rowLane = row.lane === "x" ? "tx" : "ty";
    const slot = (rc: number, lane: string) =>
      `scratch[(${rowLane} * ${u(row.slice)} + ${u(rc)}) * ${u(width)} + ${lane}]`;

    for (const up of step.updates) {
      if (up.k === "mmaFrag") {
        // THE REDISTRIBUTION — all that is left of G4.
        //
        // The inner contraction laid `P` out with the axis this one sums along
        // spread across the lanes: four of the sixty-four `n` values per
        // invocation. Summing it in place gives a quarter of each row.
        //
        // The mechanism is to stage `P` through workgroup memory, exactly as an
        // operand read from a buffer is staged, with registers as the source. The
        // budget is what makes it more than plumbing: q, k, v and the staged
        // fragment are alive at once, which is 28672 B at a 64x64x16 tile against
        // a 16384 B floor. 32x32x16 fits, at 10240 B.
        throw new Error(
          `${k.name}: contracting a fragment along an axis it holds on a lane. ` +
          `Staging it through workgroup memory is not implemented — and at this ` +
          `tile it would not fit either. See docs/004 R12, G4.`);
      }
      if (up.k !== "foldFrag") {
        throw new Error(`${k.name}: a pass with a nested contraction folds its fragment`);
      }
      const init = up.op === "max" ? PAD_LITERAL.negInf : "0.0";
      const fold = (acc: string, v: string) =>
        up.op === "max" ? `${acc} = max(${acc}, ${v});` : `${acc} = ${acc} + ${v};`;
      for (let rc = 0; rc < row.slice; rc++) {
        P(`{`, 2);
        P(`var ${T_("p")} = ${init};`, 3);
        for (let nc = 0; nc < red.slice; nc++) {
          P(fold(T_("p"), innerFrag(up.value, { [row.axis]: rc, [red.axis]: nc }, inG)), 3);
        }
        P(`${slot(rc, redLane)} = ${T_("p")};`, 3);
        P(`}`, 2);
      }
      P(`workgroupBarrier();`, 2);
      for (let rc = 0; rc < row.slice; rc++) {
        P(`{`, 2);
        P(`var ${T_("r")} = ${init};`, 3);
        P(`for (var ${T_("j")} : u32 = 0u; ${T_("j")} < ${u(width)}; ${T_("j")} = ${T_("j")} + 1u) {`, 3);
        P(fold(T_("r"), slot(rc, T_("j"))), 4);
        P(`}`, 3);
        P(fold(`${up.acc}[${rc}]`, T_("r")), 3);
        P(`}`, 2);
      }
      P(`workgroupBarrier();`, 2);
    }
  }

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

  /**
   * Reduce a fragment along its non-row axis into a row vector.
   *
   * This cannot be an expression, which is why `rowExpr` refuses it: the
   * fragment is spread across lanes, so the reduction crosses them and needs a
   * barrier. `accumulate = (m, n)` puts n on the x lane, so one row lives in all
   * 16 x lanes at four columns each, and reducing along n is two steps — fold the
   * four columns this invocation holds, then fold across the lanes.
   *
   * The same shape as emitCombine, which folds a contraction split across lanes,
   * but not the same code: there the accumulator is one-dimensional and indexed
   * by the row cell, here the fragment is two-dimensional and only one of its
   * axes is being folded.
   */
  function emitReduceFrag(name: string, r: Extract<RowExpr, { k: "reduceFrag" }>): void {
    if (slices.length !== 2) {
      throw new Error(`${k.name}: reducing a fragment needs two accumulate axes, got ${slices.length}`);
    }
    const [row, red] = slices;
    const width = laneW[red.lane];
    const rowLane = row.lane === "x" ? "tx" : "ty";
    const redLane = red.lane === "x" ? "tx" : "ty";
    const init = r.op === "max" ? PAD_LITERAL.negInf : "0.0";
    const fold = (acc: string, v: string) =>
      r.op === "max" ? `${acc} = max(${acc}, ${v});` : `${acc} = ${acc} + ${v};`;
    const slot = (rc: number, lane: string) =>
      `scratch[(${rowLane} * ${u(row.slice)} + ${u(rc)}) * ${u(width)} + ${lane}]`;

    P(`var ${name} : array<${ty}, ${row.slice}>;`, 1);
    for (let rc = 0; rc < row.slice; rc++) {
      P(`{`, 1);
      P(`var ${T_("p")} = ${init};`, 2);
      for (let nc = 0; nc < red.slice; nc++) {
        P(fold(T_("p"), fragExpr(r.a, { [row.axis]: rc, [red.axis]: nc })), 2);
      }
      P(`${slot(rc, redLane)} = ${T_("p")};`, 2);
      P(`}`, 1);
    }
    P(`workgroupBarrier();`, 1);
    for (let rc = 0; rc < row.slice; rc++) {
      P(`{`, 1);
      P(`var ${T_("r")} = ${init};`, 2);
      P(`for (var ${T_("j")} : u32 = 0u; ${T_("j")} < ${u(width)}; ${T_("j")} = ${T_("j")} + 1u) {`, 2);
      P(fold(T_("r"), slot(rc, T_("j"))), 3);
      P(`}`, 2);
      P(`${name}[${u(rc).slice(0, -1)}] = ${T_("r")};`, 2);
      P(`}`, 1);
    }
    P(`workgroupBarrier();`, 1);
    P();
  }

  /** A block-valued expression at one fragment cell. */
  /**
   * A block-valued expression at one fragment cell, named by a GEOMETRY.
   *
   * Which suffixes an operand's value carries depends on which of its axes are
   * accumulated, and that differs between an inner contraction and the outer one:
   * `b` over `[k, n]` is `v_b` where only `m` accumulates and `v_b_0..3` where
   * `n` does too. Reading the name off the wrong geometry produced `v_b` used
   * against `v_b_0` defined, which naga caught and tessera did not.
   */
  function blockExpr(e: Expr, c: Record<string, number>, gm: Geom = G): string {
    switch (e.k) {
      case "tile":  return gm.operandVal(e.binding, c);
      case "unary": return e.op === "sq"
        ? `(${blockExpr(e.a, c, gm)} * ${blockExpr(e.a, c, gm)})`
        : `exp(${blockExpr(e.a, c, gm)})`;
      case "rowOp": {
        const a = blockExpr(e.a, c, gm), r = rowExpr(e.row, c);
        return e.op === "sub" ? `(${a} - ${r})` : e.op === "mul" ? `(${a} * ${r})` : `(${a} / ${r})`;
      }
    }
  }
  function rowExpr(r: RowExpr, c: Record<string, number>): string {
    switch (r.k) {
      case "acc":  return `${r.name}[${c[slices[0].axis]}]`;
      case "reduceFrag": throw new Error(
        `${k.name}: a fragment reduction is materialised by emitReduceFrag, ` +
        `not read as an expression — it needs a barrier`);
      case "mean": return `(${rowExpr(r.a, c)} / ${contract.extent}.0)`;
      case "rstd": {
        const mu = rowExpr(r.mean, c);
        return `inverseSqrt((${rowExpr(r.sumSq, c)} / ${contract.extent}.0) - ${mu} * ${mu} + ${r.eps})`;
      }
    }
  }
  function fragExpr(f: FragExpr, c: Record<string, number>): string {
    switch (f.k) {
      case "acc":   return `${f.name}[${cellIndex(c)}]`;
      case "map":   return `select(0.0, ${fragExpr(f.a, c)}, ${fragExpr(f.a, c)} > 0.0)`;
      case "unary": {
        const a = fragExpr(f.a, c);
        return f.op === "exp" ? `exp(${a})` : `(${a} * ${a})`;
      }
      case "rowOp": {
        const a = fragExpr(f.a, c), r = rowExpr(f.row, c);
        return f.op === "sub" ? `(${a} - ${r})` : f.op === "mul" ? `(${a} * ${r})` : `(${a} / ${r})`;
      }
    }
  }

  // ---- derived row values, then the store ------------------------------------
  for (const st of body.steps) {
    if (st.k === "derivedFrag") {
      // Elementwise over the invocation's own cells: no lane crosses, no barrier.
      P(`var ${st.name} : array<${ty}, ${frag}>;`, 1);
      for (const c of cells) P(`${st.name}[${cellIndex(c)}] = ${fragExpr(st.expr, c)};`, 1);
      P();
      continue;
    }
    if (st.k !== "derived") continue;
    if (st.expr.k === "reduceFrag") { emitReduceFrag(st.name, st.expr); continue; }
    P(`var ${st.name} : array<${ty}, ${frag}>;`, 1);
    for (const c of cells) P(`${st.name}[${cellIndex(c)}] = ${rowExpr(st.expr, c)};`, 1);
    P();
  }

  // Whether the store walks the contraction is a property of the output's axes:
  // if the output is indexed BY the contract axis it must, and if it is not it
  // writes the fragment once. Nobody chooses this either.
  const storeStep = body.steps.find((x) => x.k === "store" || x.k === "storeFrag");
  if (!storeStep) throw new Error(`${k.name}: no store`);
  // A store inside a loop walks THAT loop's axis, which need not be the kernel's
  // only reduction axis any more.
  const sx = storeStep.k === "store" ? axisIR(storeStep.axis) : contract;
  const splan = planFor.get(sx.name)!;
  const sDepth = sx.block / contractLaneW;
  const storeWalksContraction = out.axes.includes(sx.name);

  const storeWasNested = storeStep.k === "store"
    && (storeStep.locals.length > 0 || storeStep.inner.length > 0);
  if (!storeWasNested && storeWalksContraction && storeStep.k === "store") {
    P(`for (var ${T_("cb")} : u32 = 0u; ${T_("cb")} < ${u(sx.extent)}; ${T_("cb")} = ${T_("cb")} + ${u(sx.block)}) {`, 1);
    // The part of a staged address that does not move with the contract step is
    // loop-invariant by construction — it is the invocation's own slice — so it is
    // computed once rather than once per step per operand.
    const hoisted = new Map<string, string>();
    for (const o of splan.operands.filter((o) => o.staged)) {
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
    P(`for (var ${T_("s")} : u32 = 0u; ${T_("s")} < ${u(sDepth)}; ${T_("s")} = ${T_("s")} + 1u) {`, 2);
    const laneId = splan.lanes.contractLanes.includes("x") ? "tx" : "ty";
    // Two coordinates, not one. `lc` is the offset INSIDE the contract block —
    // what a staged read wants — and `ci` is the global index a direct read wants.
    // Deriving the first from the second meant emitting `ci - cb`, which is an add
    // and a subtract that cancel, repeated once per staged operand slice, and which
    // stops the address being an obvious affine function of the loop variable.
    P(splan.lanes.needsCombine
      ? `let ${T_("lc")} = ${laneId} * ${u(sDepth)} + ${T_("s")};`
      : `let ${T_("lc")} = ${T_("s")};`, 3);
    P(`let ${T_("ci")} = ${T_("cb")} + ${T_("lc")};`, 3);
    for (const o of splan.operands) {
      const bd = byName(o.binding);
      for (const c of cells) {
        const g0 = bd.axes[0] === sx.name ? T_("ci") : accCoord(bd.axes[0], c[bd.axes[0]]);
        const g1 = bd.axes[1] === sx.name ? T_("ci") : accCoord(bd.axes[1], c[bd.axes[1]]);
        emitLoad(`let ${operandVal(o.binding, c)}`, bd, `(${g0})`, `(${g1})`, 4);
      }
    }
    for (const c of cells) {
      const g0 = out.axes[0] === sx.name ? T_("ci") : accCoord(out.axes[0], c[out.axes[0]]);
      const g1 = out.axes[1] === sx.name ? T_("ci") : accCoord(out.axes[1], c[out.axes[1]]);
      emitStore(out, `(${g0})`, `(${g1})`,
                storeStep.fromFrag ? fragExpr(storeStep.value as FragExpr, c)
                                   : blockExpr(storeStep.value as Expr, c), 4);
    }
    P(`}`, 2);
    P(`}`, 1);
  } else if (storeStep.k === "storeFrag") {
    for (const c of cells) {
      emitStore(out, `(${accCoord(out.axes[0], c[out.axes[0]])})`,
                `(${accCoord(out.axes[1], c[out.axes[1]])})`, fragExpr(storeStep.value, c), 1);
    }
  } else if (storeWasNested) {
    // Already emitted above, inside its own pass — the fragment it writes only
    // exists there.
  } else {
    throw new Error(`${k.name}: the store's form and the output's axes disagree`);
  }

  P(`}`);
  return L.join("\n") + "\n";
}
