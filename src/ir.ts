/**
 * tessera IR — what the front end produces and every backend consumes.
 *
 * Deliberately tiny and fully constant-folded. Everything the surface encodes in
 * types has been resolved to a plain number by the time it lands here, because
 * the whole point of the surface is that these ARE compile-time literals. A
 * backend never has to ask "what if this extent is dynamic".
 */

export type DTypeName = "f32" | "f16" | "i32";

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
  readonly dtype: DTypeName;
  readonly tile: { readonly bm: number; readonly bn: number; readonly bk: number };
  /** Exactly two, blocked at bm and bn respectively. */
  readonly grid: readonly AxisIR[];
  /** One or more, each blocked at bk. */
  readonly reduce: readonly AxisIR[];
  readonly bindings: readonly BindingIR[];

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
): Pick<KernelIR, "workgroup" | "fragment" | "workgroupBytes" | "dispatch"> {
  const [wgx, wgy, wgz] = workgroup;
  const invocations = wgx * wgy * wgz;

  if ((tile.bm * tile.bn) % invocations !== 0) {
    throw new Error(
      `tile ${tile.bm}x${tile.bn} (${tile.bm * tile.bn} elements) is not divisible by ` +
      `${invocations} invocations — the per-invocation fragment would not be a constant fold`,
    );
  }
  if (tile.bm % wgy !== 0 || tile.bn % wgx !== 0) {
    throw new Error(`tile ${tile.bm}x${tile.bn} does not divide evenly over workgroup ${wgx}x${wgy}`);
  }

  const workgroupBytes = (tile.bm * tile.bk + tile.bk * tile.bn) * DTYPE_BYTES[dtype];
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
    dispatch: [grid[1].tiles, grid[0].tiles, 1],
  };
}
