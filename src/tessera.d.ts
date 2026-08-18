// tessera -- v0 public surface (block level, named axes).
//
// DESIGN RULE, enforced by review: tsc is allowed only literal equality, union
// membership, indexed access into a GENERATED interface, mapped types with `as`
// remapping over <=8-element tuples, and non-recursive conditionals of depth 1.
// No recursive conditional types. No type-level arithmetic. Every integer
// operation runs in JavaScript -- once at `tessera gen` time, or in the tessera
// checker pass over the typed AST.

import type { Block, TileFit, WorkgroupShape } from "./tessera.gen";

export type { Block, WorkgroupShape };

/* ------------------------------------------------------------------ dtypes */

export interface f32 { readonly dtype: "f32"; readonly bytes: 4 }
export interface f16 { readonly dtype: "f16"; readonly bytes: 2 }
export interface i32 { readonly dtype: "i32"; readonly bytes: 4 }
export type DType = f32 | f16 | i32;

export declare const f32: f32;
export declare const f16: f16;
export declare const i32: i32;

/** WebGPU has no f64 at all. The type's NAME is the diagnostic. */
export interface f64_does_not_exist_in_WebGPU_use_f32 { readonly __never: unique symbol }
export type f64 = f64_does_not_exist_in_WebGPU_use_f32;

export type FeatureName = "shader-f16";

/** Supplied by @webgpu/types in a real project. */
export interface GPUBuffer { readonly size: number }

/* ------------------------------------------------- compile-time admission */

/**
 * Failure carriers live ONLY in parameter position, never in a covariant
 * return. Every rejection is therefore an argument-constraint violation -- the
 * one sound mechanism -- and `never` can never be smuggled through a tuple slot.
 */
export interface MustBeLiteral<Why extends string> { readonly __tessera_error: Why }
export type Literal<N extends number> =
  number extends N
    ? MustBeLiteral<"tensor extent must be a compile-time integer literal">
    : unknown;

/* -------------------------------------------------------------------- axes */

export type Fitness = "exact" | "ragged";

export interface Axis<
  Name extends string,
  Extent extends number,
  B extends number,
  Fit extends Fitness,
> {
  readonly name: Name;
  readonly extent: Extent;
  readonly block: B;
  readonly fit: Fit;
  /** Deliberately `number`. Trip counts are the tessera pass's arithmetic. */
  readonly tiles: number;
}
export type AnyAxis = Axis<string, number, number, Fitness>;

/**
 * Declare an axis whose extent the block divides evenly. tessera's pass proves
 * `extent % block === 0`; if it does not hold you get TSA0301 naming the legal
 * blocks and pointing at `raggedAxis`.
 */
export declare function axis<
  const Name extends string, E extends number, B extends 8 | 16 | 32 | 64 | 128,
>(
  name: Name,
  extent: E & Literal<E>,
  block: B,
): Axis<Name, E, B, "exact">;

/**
 * Declare an axis the block does NOT divide. Every load through it becomes a
 * `RaggedTile`, so the program does not compile until you name the identity
 * value out-of-range lanes read. tessera synthesizes the mask itself.
 */
export declare function raggedAxis<
  const Name extends string, E extends number, B extends 8 | 16 | 32 | 64 | 128,
>(
  name: Name,
  extent: E & Literal<E>,
  block: B,
): Axis<Name, E, B, "ragged">;

/* ------------------------------------------------------------------ tiling */

export interface Tiling<D extends DType, BM extends number, BN extends number, BK extends number> {
  readonly dtype: D;
  readonly bm: BM;
  readonly bn: BN;
  readonly bk: BK;
}

/** Admits only triples that fit the workgroup-storage budget for this dtype. */
export declare function tiling<
  D extends DType,
  BM extends Extract<keyof TileFit[D["bytes"]], number>,
  BN extends Extract<keyof TileFit[D["bytes"]][BM], number>,
  BK extends Extract<TileFit[D["bytes"]][BM][BN], number>,
>(dtype: D, bm: BM, bn: BN, bk: BK): Tiling<D, BM, BN, BK>;

/* ----------------------------------------------- coordinates as capabilities */

/** Branded by axis NAME as well as extent/block: a transposed pair is an error. */
export interface Idx<Name extends string, Extent extends number, B extends number> {
  readonly __axis: Name;
  readonly __extent: Extent;
  readonly __block: B;
}
export type IdxOf<A extends AnyAxis> = Idx<A["name"], A["extent"], A["block"]>;

/* ------------------------------------------------------------------ values */

/**
 * Workgroup-resident block. `Pad` is the value out-of-range lanes carry:
 * `0` for a block proved in bounds, or the identity passed to `.pad()`.
 */
export interface Tile<
  S extends readonly [number, number],
  D extends DType,
  Pad extends PadState = "exact",
> {
  readonly __shape: S;
  readonly __dtype: D;
  readonly __pad: Pad;
  readonly __space: "workgroup";
}

/* ------------------------------------------------------------- identities */
/**
 * The identity element of a reduction, as a NAME rather than a number.
 *
 * Numbers cannot do this job. The identity for a masked max is negative
 * infinity, and TypeScript has no literal type for it — `-Infinity` is unary
 * minus applied to `Infinity`, so it widens to `number` and carries nothing.
 * Naming the identities makes them expressible, and attaches the obligation to
 * the OPERATOR: `mma` sums, so it demands the additive identity; a max demands
 * the max identity. Passing the wrong one is then a type error for the same
 * structural reason that a non-annihilating pad already was.
 */
export type PadName = "zero" | "one" | "negInf" | "posInf";

/** `"exact"` means the axis divides its block, so nothing was ever masked. */
export type PadState = "exact" | PadName | NeedsPad;

export interface Identity<N extends PadName> { readonly __identity: N }

export declare const zero: Identity<"zero">;
export declare const one: Identity<"one">;
export declare const negInf: Identity<"negInf">;
export declare const posInf: Identity<"posInf">;

/** The sentinel is a string so tsc PRINTS the instruction in the diagnostic. */
export type NeedsPad =
  "this block is ragged: call .pad(identity) to say what out-of-range lanes read";

/**
 * A block on an axis the tile size does not divide. It is a `Tile` whose pad is
 * unresolved, so shape inference still works downstream and the only thing that
 * fails is the missing identity.
 */
export interface RaggedTile<S extends readonly [number, number], D extends DType>
  extends Tile<S, D, NeedsPad> {
  /** Name only the identity element; tessera derives the mask from the extents. */
  pad<N extends PadName>(identity: Identity<N>): Tile<S, D, N>;
}

/** Register-resident accumulator, at TILE granularity. The compiler splits it
 *  into per-invocation fragments: (BM*BN)/(wgX*wgY), a constant fold. */
export interface Frag<S extends readonly [number, number], D extends DType> {
  readonly __shape: S;
  readonly __dtype: D;
  readonly __space: "register";
}

/** The only conditional type in the surface. Non-recursive, depth 1. */
export type TileOf<A0 extends AnyAxis, A1 extends AnyAxis, D extends DType> =
  "ragged" extends A0["fit"] | A1["fit"]
    ? RaggedTile<readonly [A0["block"], A1["block"]], D>
    : Tile<readonly [A0["block"], A1["block"]], D, "exact">;

/* ---------------------------------------------------------------- bindings */

export interface Binding<
  Name extends string,
  A0 extends AnyAxis,
  A1 extends AnyAxis,
  D extends DType,
  Mode extends "read" | "write",
> {
  readonly name: Name;
  readonly axes: readonly [A0, A1];
  readonly dtype: D;
  readonly mode: Mode;
}
export type AnyBinding = Binding<string, AnyAxis, AnyAxis, DType, "read" | "write">;

/** Exactly 8 slots: WebGPU maxStorageBuffersPerShaderStage. */
export type BindingList = readonly [
  AnyBinding?, AnyBinding?, AnyBinding?, AnyBinding?,
  AnyBinding?, AnyBinding?, AnyBinding?, AnyBinding?,
];

export declare function input<
  const Name extends string, A0 extends AnyAxis, A1 extends AnyAxis, D extends DType,
>(name: Name, axes: readonly [A0, A1], dtype: D): Binding<Name, A0, A1, D, "read">;

export declare function output<
  const Name extends string, A0 extends AnyAxis, A1 extends AnyAxis, D extends DType,
>(name: Name, axes: readonly [A0, A1], dtype: D): Binding<Name, A0, A1, D, "write">;

/* ----------------------------------------------------------------- handles */
// No index signature and no `.data` anywhere: dynamic indexing is unsayable,
// not diagnosed.

export interface InputHandle<A0 extends AnyAxis, A1 extends AnyAxis, D extends DType> {
  /** Stage one block into workgroup memory. Barriers and the bank-conflict
   *  swizzle are compiler-owned; there is no barrier primitive in this language. */
  tile(i: IdxOf<A0>, j: IdxOf<A1>): TileOf<A0, A1, D>;
}

export interface OutputSlot<S extends readonly [number, number], D extends DType> {
  /** Always edge-clipped. A masked WRITE needs no identity, so none is asked for. */
  store(v: Frag<S, D>): void;
}

export interface OutputHandle<A0 extends AnyAxis, A1 extends AnyAxis, D extends DType> {
  tile(i: IdxOf<A0>, j: IdxOf<A1>): OutputSlot<readonly [A0["block"], A1["block"]], D>;
}

export type HandleOf<B extends AnyBinding> =
  B extends Binding<string, infer A0, infer A1, infer D, "read"> ? InputHandle<A0, A1, D> :
  B extends Binding<string, infer A0, infer A1, infer D, "write"> ? OutputHandle<A0, A1, D> :
  never;

/* --------------------------------------------------------------- tile ops */

export declare function zeros<BM extends number, BN extends number, D extends DType>(
  bm: BM, bn: BN, dtype: D,
): Frag<readonly [BM, BN], D>;

/**
 * `NoInfer` pins K to `a` alone. Without it TypeScript silently ACCEPTS a
 * K-mismatch and emits a confidently wrong type (recon A, hard constraint 3).
 * `Tile<..., 0>` on both operands: a non-annihilating pad would leak into the
 * reduction at a ragged edge, which is a numerics bug that only shows on edges.
 */
export declare function mma<
  BM extends number, BK extends number, BN extends number, D extends DType,
>(
  a: Tile<readonly [BM, BK], D, "exact" | "zero">,
  b: Tile<readonly [NoInfer<BK>, BN], NoInfer<D>, "exact" | "zero">,
  acc: Frag<readonly [NoInfer<BM>, NoInfer<BN>], NoInfer<D>>,
): Frag<readonly [BM, BN], D>;

export declare function relu<S extends readonly [number, number], D extends DType>(
  x: Frag<S, D>): Frag<S, D>;
export declare function exp<S extends readonly [number, number], D extends DType>(
  x: Frag<S, D>): Frag<S, D>;
export declare function scale<S extends readonly [number, number], D extends DType>(
  x: Frag<S, D>, k: number): Frag<S, D>;
export declare function add<S extends readonly [number, number], D extends DType>(
  a: Frag<S, D>, b: Frag<NoInfer<S>, NoInfer<D>>): Frag<S, D>;
export declare function mul<S extends readonly [number, number], D extends DType>(
  a: Frag<S, D>, b: Frag<NoInfer<S>, NoInfer<D>>): Frag<S, D>;
export declare function cast<
  To extends DType, S extends readonly [number, number], D extends DType,
>(x: Frag<S, D>, to: To): Frag<S, To>;

/* ----------------------------------------------------------------- kernel */

export type Ctx<
  Grid extends readonly AnyAxis[],
  Reduce extends readonly AnyAxis[],
  Bs extends BindingList,
> =
  & { readonly [B in NonNullable<Bs[number]> as B["name"]]: HandleOf<B> }
  & {
      readonly at: { readonly [A in Grid[number] as A["name"]]: IdxOf<A> };
      readonly reduce: { readonly [A in Reduce[number] as A["name"]]: Iterable<IdxOf<A>> };
    };

export interface CompileReport {
  readonly workgroup: readonly [number, number, number];
  readonly dispatch: readonly [number, number, number];
  readonly workgroupBytes: number;   // vs the 16384 guaranteed floor
  readonly fragment: readonly [number, number];  // per-invocation = tile / workgroup
  readonly maskedLoads: readonly string[];
  readonly storageBuffers: number;   // vs 8
}

export type HostArgs<Bs extends BindingList> = {
  readonly [B in NonNullable<Bs[number]> as B["name"]]:
    B extends Binding<string, infer A0, infer A1, infer D, "read" | "write">
      ? HostTensor<A0, A1, D> : never;
};
export type GpuArgs<Bs extends BindingList> = {
  readonly [B in NonNullable<Bs[number]> as B["name"]]:
    B extends Binding<string, infer A0, infer A1, infer D, "read" | "write">
      ? GpuTensor<A0, A1, D> : never;
};

export interface Kernel<Bs extends BindingList, Req extends FeatureName> {
  readonly wgsl: string;
  readonly mlir: string;
  readonly report: CompileReport;
  readonly requires: readonly Req[];
  /** THE SAME kernel body, executed by tessera's CPU backend over typed arrays
   *  under plain node, with f32 rounding. The v0 differential oracle: not a
   *  second implementation, so it cannot drift. */
  reference(args: HostArgs<Bs>): void;
  run(device: Device<Req>, args: GpuArgs<Bs>): Promise<void>;
}

export declare function kernel<
  const Grid extends readonly [AnyAxis, ...AnyAxis[]],
  const Reduce extends readonly AnyAxis[],
  const Bs extends BindingList,
  T extends Tiling<DType, number, number, number> = Tiling<DType, number, number, number>,
>(
  spec: {
    readonly name: string;
    /**
     * Optional, and no longer positionally bound to the axes.
     *
     * An `Axis` already carries its own block size, so tying grid[0] to `T["bm"]`
     * in the type was a matmul-shaped assumption: it forced every kernel to have
     * exactly two parallel axes and one reduction axis blocked at `bk`. softmax
     * is `grid: [m], reduce: [n]`, and could not be written at all.
     *
     * `tiling()` still earns its place as the thing that produces legal block
     * triples with good autocomplete. What moved is the coherence check —
     * "this grid axis is blocked at 32 but the tile says 64" is now reported by
     * tessera's pass as one error, rather than by tsc as a seven-diagnostic
     * cascade. docs/001 §7 listed suppressing that cascade as outstanding work.
     */
    readonly tile?: T;
    readonly grid: Grid;
    readonly reduce: Reduce;
    readonly bindings: Bs;
    /** Optional. Default: the largest legal shape whose invocation count
     *  divides BM*BN. Delete it and the program means exactly the same thing. */
    readonly workgroup?: WorkgroupShape;
    /** Optional. Default "row". "grouped" is Triton's L2 swizzle as a policy. */
    readonly order?: "row" | "column" | { readonly grouped: number };
  },
  body: (ctx: Ctx<Grid, Reduce, Bs>) => void,
): Kernel<Bs, T["dtype"] extends f16 ? "shader-f16" : never>;

/* ------------------------------------------------------------------- host */

export interface HostTensor<A0 extends AnyAxis, A1 extends AnyAxis, D extends DType> {
  readonly axes: readonly [A0, A1];
  readonly dtype: D;
  readonly data: ArrayBufferView;
}
export interface GpuTensor<A0 extends AnyAxis, A1 extends AnyAxis, D extends DType> {
  readonly axes: readonly [A0, A1];
  readonly dtype: D;
  readonly buffer: GPUBuffer;
}

/** Feature set as a membership map, so a MORE capable device stays assignable. */
export interface Device<F extends FeatureName = never> {
  readonly supports: { readonly [K in F]: true };
}
export declare function requestDevice<const F extends readonly FeatureName[] = []>(
  features?: F,
): Promise<Device<F[number]>>;

export declare function randn<A0 extends AnyAxis, A1 extends AnyAxis, D extends DType>(
  axes: readonly [A0, A1], dtype: D, seed?: number): HostTensor<A0, A1, D>;
export declare function emptyHost<A0 extends AnyAxis, A1 extends AnyAxis, D extends DType>(
  axes: readonly [A0, A1], dtype: D): HostTensor<A0, A1, D>;
export declare function upload<A0 extends AnyAxis, A1 extends AnyAxis, D extends DType>(
  device: Device, t: HostTensor<A0, A1, D>): GpuTensor<A0, A1, D>;
export declare function download<A0 extends AnyAxis, A1 extends AnyAxis, D extends DType>(
  device: Device, t: GpuTensor<A0, A1, D>): Promise<HostTensor<A0, A1, D>>;
export declare function allClose<A0 extends AnyAxis, A1 extends AnyAxis, D extends DType>(
  a: HostTensor<A0, A1, D>,
  b: HostTensor<NoInfer<A0>, NoInfer<A1>, NoInfer<D>>,
  opts?: { rtol?: number; atol?: number },
): { ok: boolean; maxAbsDiff: number; maxRelDiff: number };
