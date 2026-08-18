/**
 * tessera IR — what the front end produces and every backend consumes.
 *
 * Deliberately tiny and fully constant-folded. Everything the surface encodes in
 * types has been resolved to a plain number by the time it lands here, because
 * the whole point of the surface is that these ARE compile-time literals. A
 * backend never has to ask "what if this extent is dynamic".
 */

export type DTypeName = "f32" | "f16" | "i32";

export type PadName = "zero" | "one" | "negInf" | "posInf";

/**
 * Which schedule the body describes.
 *
 * An honest admission: tessera recognises a fixed set of schedules rather than
 * compiling an arbitrary body. What the falsification test measures is whether
 * the DERIVED parts — masks, indexing, dispatch — are shared across them, or
 * re-written per schedule. See docs/004.
 */
export type Schedule = "matmul" | "softmax";

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
  negInf: "-3.4028235e38",
  posInf: "3.4028235e38",
};

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
