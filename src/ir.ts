/**
 * tessera IR — what the front end produces and every backend consumes.
 *
 * Deliberately tiny and fully constant-folded. Everything the surface encodes in
 * types has been resolved to a plain number by the time it lands here, because
 * the whole point of the surface is that these ARE compile-time literals. A
 * backend never has to ask "what if this extent is dynamic".
 */

import type { RowBody } from "./body.ts";

export type DTypeName = "f32" | "f16" | "i32";

export type PadName = "zero" | "one" | "negInf" | "posInf";

/**
 * Which schedule the body describes.
 *
 * `rowwise` is not a template: its accumulators, passes and store are READ from
 * the body (see body.ts) rather than matched against a per-kernel shape. softmax
 * and layernorm are both `rowwise` and share every line of the emitter.
 * `matmul` is still recognised by shape, and generalising it is the next step.
 */
export type Schedule = "matmul" | "rowwise";

/**
 * The literal a backend emits for an identity.
 *
 * WGSL has no infinity literal, so the infinities are the largest finite
 * magnitudes instead. For the two uses that matter — annihilating a masked max,
 * and mapping to 0 under exp — a value this large is indistinguishable from
 * infinity in f32, and unlike a bitcast it survives every backend unchanged.
 */
export const PAD_LITERAL: Record<PadName, string> = {
  zero: "0.0",
  one: "1.0",
  // The EXACT f32 maximum, (2 - 2^-23) * 2^127. The obvious-looking
  // "3.4028235e38" is larger than it and does not round-trip, and Tint rejects
  // it outright — "value ... cannot be represented as 'f32'". naga validated it
  // without complaint, so the emitted WGSL passed every local check and only
  // failed in the browser. See assertF32Literals below.
  negInf: "-3.4028234663852886e38",
  posInf: "3.4028234663852886e38",
};

/**
 * Every float literal in emitted WGSL must survive a round trip through f32.
 *
 * naga does not check this — it accepted 57 occurrences of an out-of-range
 * literal and reported "Validation successful", and the failure surfaced only as
 * a shader compilation error in Chrome. So the check lives here, where it costs
 * nothing and catches the whole class rather than the one instance.
 */
export const F32_MAX = 3.4028234663852886e38;

export function assertF32Literals(wgsl: string, where: string): void {
  const bad: string[] = [];
  for (const m of wgsl.matchAll(/-?\d+\.\d+(?:[eE][-+]?\d+)?/g)) {
    const v = Number(m[0]);
    if (!Number.isFinite(v)) continue;
    // RANGE, not exactness. Requiring an exact round trip was wrong and rejected
    // `0.00001` — almost no decimal literal is exactly representable in f32, and
    // rounding to the nearest one is normal and permitted. What Tint rejects is a
    // magnitude beyond f32's, even when it would round back down to the maximum.
    if (Math.abs(v) > F32_MAX) bad.push(m[0]);
  }
  if (bad.length) {
    throw new Error(
      `${where}: ${bad.length} float literal(s) exceed f32's range (${F32_MAX}), ` +
      `which Tint rejects even though naga does not: ${[...new Set(bad)].slice(0, 3).join(", ")}`);
  }
}

export interface AxisIR {
  readonly name: string;
  readonly extent: number;
  readonly block: number;
  readonly fit: "exact" | "ragged";
  /** ceil(extent / block) — the trip count. Folded here, not in a backend. */
  readonly tiles: number;
}

export interface BindingIR {
  readonly name: string;
  /** Axis names, in order. Row-major: [rows, cols]. */
  readonly axes: readonly [string, string];
  readonly dtype: DTypeName;
  readonly mode: "read" | "write";
  /** extent0 * extent1 — the flat element count of the storage buffer. */
  readonly elements: number;
  /**
   * The body reads this binding through `.tileT()`, so the tile it wants is
   * `[axes[1], axes[0]]` while the MEMORY stays `[axes[0], axes[1]]`.
   *
   * Everything that reasons about the tile's shape sees the swapped order; the
   * one place that touches memory swaps back. Set from the body — the surface
   * says which of `.tile` / `.tileT` was written, and a transposed axis stays a
   * type error unless it is said.
   */
  readonly transposed?: boolean;
}

export interface KernelIR {
  readonly name: string;
  readonly schedule: Schedule;
  readonly dtype: DTypeName;
  readonly tile: { readonly bm: number; readonly bn: number; readonly bk: number };
  /** Exactly two, blocked at bm and bn respectively. */
  readonly grid: readonly AxisIR[];
  /** One or more, each blocked at bk. */
  readonly reduce: readonly AxisIR[];
  readonly bindings: readonly BindingIR[];
  /**
   * Loads and stores that need a boundary mask, as "binding:axis" pairs.
   * Derived, not declared: a load through a ragged axis is masked, and that is
   * the whole rule. The surface's job was to make the user name the identity
   * element; deciding WHERE masks go is the compiler's.
   */
  readonly maskedLoads: readonly string[];
  /** Present for `rowwise`: the body, parsed. */
  readonly body?: RowBody;
  /**
   * The identity element for masked loads, by NAME.
   *
   * Not a number, because the identity for a masked max is negative infinity and
   * TypeScript has no literal type for it. Naming it makes it expressible and
   * ties the obligation to the reduction operator rather than to a value.
   */
  readonly pad: PadName;

  // ---- derived, folded once by the front end -------------------------------
  readonly workgroup: readonly [number, number, number];
  /** Per-invocation register fragment: (bm*bn) / (wgx*wgy), as [tm, tn]. */
  readonly fragment: readonly [number, number];
  /** Bytes of workgroup storage: (bm*bk + bk*bn) * dtype bytes. */
  readonly workgroupBytes: number;
  readonly dispatch: readonly [number, number, number];
}

export const DTYPE_BYTES: Record<DTypeName, number> = { f32: 4, f16: 2, i32: 4 };

/** The one place the tile/workgroup/fragment arithmetic lives. */
export function derive(
  tile: { bm: number; bn: number; bk: number },
  dtype: DTypeName,
  grid: readonly AxisIR[],
  workgroup: readonly [number, number, number] = [16, 16, 1],
  schedule: Schedule = "matmul",
): Pick<KernelIR, "workgroup" | "fragment" | "workgroupBytes" | "dispatch"> {
  const [wgx, wgy, wgz] = workgroup;
  const invocations = wgx * wgy * wgz;

  // NOTE: everything below is the MATMUL schedule's budget arithmetic — a 2-D
  // register fragment and two staged operand tiles. A different schedule has a
  // different budget. This is the one place in derive() that is not generic over
  // the axes, and it is honest to say so rather than to pretend otherwise.
  if ((tile.bm * tile.bn) % invocations !== 0) {
    throw new Error(
      `tile ${tile.bm}x${tile.bn} (${tile.bm * tile.bn} elements) is not divisible by ` +
      `${invocations} invocations — the per-invocation fragment would not be a constant fold`,
    );
  }
  if (tile.bm % wgy !== 0 || tile.bn % wgx !== 0) {
    throw new Error(`tile ${tile.bm}x${tile.bn} does not divide evenly over workgroup ${wgx}x${wgy}`);
  }

  // Workgroup memory is the one budget that is genuinely per-schedule: matmul
  // stages two operand tiles, softmax stages nothing and needs a [rows x lanes]
  // scratch for the cross-lane row reduction instead.
  const workgroupBytes = schedule === "matmul"
    ? (tile.bm * tile.bk + tile.bk * tile.bn) * DTYPE_BYTES[dtype]
    : tile.bm * wgx * DTYPE_BYTES[dtype];
  // WebGPU guarantees 16384; tessera reserves headroom so a later double-buffer
  // does not force a tile change.
  if (workgroupBytes > 16384 - 4096) {
    throw new Error(
      `tile needs ${workgroupBytes} B of workgroup storage, over the ${16384 - 4096} B ` +
      `tessera makes available from WebGPU's 16384 B floor`,
    );
  }

  return {
    workgroup,
    fragment: [tile.bm / wgy, tile.bn / wgx],
    workgroupBytes,
    // Dispatch is the product of the grid axes' tile counts, in WebGPU's
    // [x, y, z] order. The matmul convention was [grid[1], grid[0], 1] — column
    // axis on x — and that is kept for two axes so emitted code does not move.
    // One axis maps to x; three map directly.
    dispatch: (grid.length === 2
      ? [grid[1].tiles, grid[0].tiles, 1]
      : [grid[0].tiles, grid[1]?.tiles ?? 1, grid[2]?.tiles ?? 1]) as
      [number, number, number],
  };
}
