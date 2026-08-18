/**
 * contraction.ts — the one derivation both schedules come out of.
 *
 * The emitter used to have two paths. A register fragment accumulating an outer
 * product from two staged operands looked like a different animal from a row
 * accumulator folding one cell at a time out of global memory, and docs/004 R10
 * recorded that as an honest split: "different memory schedules, not different
 * spellings of one".
 *
 * They are the same shape read at the right altitude. Both are a CONTRACTION:
 *
 *     matmul     accumulate over (m, n), contract k   operands a[m,k], b[k,n]
 *     row-wise   accumulate over (m),    contract n   operand  x[m,n]
 *
 * and the two things that looked like schedule choices are consequences of the
 * operands' index sets:
 *
 *   STAGING. An operand that does not mention every accumulate axis is REUSED
 *   across the ones it omits — `a[m,k]` is read by every n in the fragment — and
 *   reuse is the only thing workgroup staging buys. `x[m,n]` mentions both the
 *   accumulate axis and the contract axis, so each element is touched once and
 *   staging it would be pure cost. That is why matmul stages and softmax does not,
 *   and it is a property of the indices rather than a decision anyone made.
 *
 *   CROSS-LANE REDUCTION. Accumulate axes are distributed across the workgroup's
 *   lane dimensions. If they do not use them all, the leftover lanes are given to
 *   the contract axis — which means each lane holds a partial, and the partials
 *   must be combined at the end. matmul's (m,n) fill both lane dimensions, so no
 *   combine. row-wise's (m) fills one, so tx splits the contraction and a combine
 *   is required. Again: derived, not chosen.
 */

import type { AxisIR } from "./ir.ts";

export interface LaneAssignment {
  /** Accumulate axis name -> which lane dimension carries it, and how wide a slice. */
  readonly perAxis: readonly { readonly axis: string; readonly lane: "x" | "y"; readonly slice: number }[];
  /** Lane dimensions left over, handed to the contract axis. */
  readonly contractLanes: readonly ("x" | "y")[];
  /** Elements of the accumulator each invocation owns. */
  readonly fragment: number;
  /** True when the contract axis is split across lanes and partials must be combined. */
  readonly needsCombine: boolean;
}

export interface OperandPlan {
  readonly binding: string;
  /** The axes this operand is indexed by, in order. */
  readonly axes: readonly string[];
  /** Accumulate axes it omits — the ones it is reused across. */
  readonly reusedAcross: readonly string[];
  /** Stage it into workgroup memory iff it is reused. */
  readonly staged: boolean;
  /** Elements of workgroup memory it needs when staged. */
  readonly stagedElements: number;
}

export interface ContractionPlan {
  readonly accumulate: readonly string[];
  readonly contract: string;
  readonly lanes: LaneAssignment;
  readonly operands: readonly OperandPlan[];
  readonly workgroupElements: number;
}

/**
 * `x` is the faster-moving lane dimension, so it goes to whichever axis is
 * CONTIGUOUS IN MEMORY — the last index of the output binding — regardless of
 * whether that axis is accumulated or contracted. Consecutive invocations then
 * touch consecutive addresses, which is the whole of coalescing.
 *
 * matmul's output is c[m,n], so n takes x and m takes y; the contract axis k gets
 * no lanes and each invocation walks it sequentially. Row-wise output is y[m,n]
 * with n contracted, so the CONTRACT axis takes x — which is why row-wise ends up
 * with lane partials to combine and matmul does not. One rule, both answers.
 */
const LANES = ["y", "x"] as const;

export function planContraction(
  accumulate: readonly AxisIR[],
  contract: AxisIR,
  operands: readonly { binding: string; axes: readonly string[] }[],
  workgroup: readonly [number, number, number],
  /** The last index of the output binding — the axis contiguous in memory. */
  contiguous: string,
): ContractionPlan {
  const [wgx, wgy] = workgroup;
  const width = { x: wgx, y: wgy } as const;

  if (accumulate.length > LANES.length) {
    throw new Error(`${accumulate.length} accumulate axes; a workgroup has ${LANES.length} lane dimensions`);
  }

  // `x` belongs to the contiguous axis. If that axis is contracted, the accumulate
  // axes take what is left and the contraction ends up split across lanes.
  const contiguousIsAccumulated = accumulate.some((a) => a.name === contiguous);
  const take = contiguousIsAccumulated
    ? LANES.slice(LANES.length - accumulate.length)          // ..., x for the last
    : LANES.slice(0, accumulate.length);                     // y first, x left over

  const ordered = contiguousIsAccumulated
    ? [...accumulate].sort((p, q) => (p.name === contiguous ? 1 : 0) - (q.name === contiguous ? 1 : 0))
    : accumulate;

  const perAxis = ordered.map((a, i) => {
    const lane = take[i];
    if (a.block % width[lane] !== 0) {
      throw new Error(
        `axis "${a.name}" is blocked at ${a.block}, which ${width[lane]} lanes do not divide — ` +
        `the per-invocation slice would not be a constant`);
    }
    return { axis: a.name, lane, slice: a.block / width[lane] };
  });

  const contractLanes = LANES.filter((l) => !take.includes(l));
  const fragment = perAxis.reduce((n, p) => n * p.slice, 1);

  const accNames = new Set(accumulate.map((a) => a.name));
  const plans: OperandPlan[] = operands.map((o) => {
    const mentions = new Set(o.axes);
    const reusedAcross = [...accNames].filter((n) => !mentions.has(n));
    const staged = reusedAcross.length > 0;
    // A staged operand holds one block of every axis it mentions.
    const stagedElements = staged
      ? o.axes.reduce((n, ax) => {
          const a = [...accumulate, contract].find((x) => x.name === ax);
          if (!a) throw new Error(`operand "${o.binding}" mentions unknown axis "${ax}"`);
          return n * a.block;
        }, 1)
      : 0;
    return { binding: o.binding, axes: o.axes, reusedAcross, staged, stagedElements };
  });

  const staged = plans.filter((p) => p.staged);
  const workgroupElements = staged.length
    ? staged.reduce((n, p) => n + p.stagedElements, 0)
    // Nothing staged, so workgroup memory is only needed for the cross-lane
    // combine: one slot per accumulator element per lane sharing it.
    : fragment * wgy * contractLanes.reduce((n, l) => n * width[l], 1);

  return {
    accumulate: accumulate.map((a) => a.name),
    contract: contract.name,
    lanes: { perAxis, contractLanes, fragment, needsCombine: contractLanes.length > 0 },
    operands: plans,
    workgroupElements,
  };
}
