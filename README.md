# tessera

A compiler that lowers kernels written in TypeScript to WebGPU.

```
kernel.ts ──▶ tsc typed AST ──▶ tessera IR ──▶ WGSL          (default)
                                          └──▶ MLIR ──▶ SPIR-V ──▶ WGSL   (--backend=mlir)
```

## What it does

A transposed axis does not compile:

```ts
acc = mma(a.tile(at.n, k), b.tile(k, at.n), acc);
//                  ^^^^  Type '"n"' is not assignable to type '"k"'
```

A ragged axis with no declared identity element does not compile:

```ts
const M = raggedAxis("m", 1000, T.bm);          // 1000 % 64 = 40
acc = mma(a.tile(at.m, k), b.tile(k, at.n), acc);
//        ^^^^^^^^^^^^^^^  Type '"this block is ragged: call .pad(identity)
//                          to say what out-of-range lanes read"' is not assignable to type '0'
```

Both are the same class of bug: a kernel that is **correct at 1024 and quietly wrong at
1000**, in the tail rows a test suite with aligned sizes never touches. WGSL's robustness
clamping means neither crashes — an out-of-bounds read is remapped, an out-of-bounds write
lands somewhere legal and corrupts an interior element that was already right.

Everything else follows from the same types. From this body:

```ts
let acc = zeros(T.bm, T.bn, f32);
for (const k of reduce.k) {
  acc = mma(a.tile(at.m, k).pad(0), b.tile(k, at.n).pad(0), acc);
}
c.tile(at.m, at.n).store(relu(acc));
```

tessera derives the 4x4 per-invocation register fragment, the dispatch extents, the
workgroup staging loops and their barriers, the flat index arithmetic, and all 18
boundary masks for the ragged tails. The aligned kernel gets **zero** bound checks; the
ragged one gets them exactly where raggedness is. Both are bit-exact against a CPU
oracle — 786432/786432 and 750000/750000 — with the ragged tails included.

## Why TypeScript

- Python has Triton and JAX. A TypeScript developer who wants a WebGPU compute kernel
  writes WGSL by hand, and writes their own masks. That is verified: TypeGPU's own tiled
  matmul and MNIST examples hand-write `if (i >= n) return;`, and in `@typegpu/sort` even
  the reduction's identity element is a runtime parameter.
- TypeScript's literal types can carry a tensor extent, a block size, and an **axis name**.
  The name is what makes a transposed coordinate a type error rather than a wrong number.

## Related work

The type-driven derivation is not new, and this project does not claim it.

- **Mojo `LayoutTensor`** derives maskedness from a comptime `layout` parameter
  (`_tile_is_masked[layout, *tile_sizes]()`), threads it into the resulting type, and
  branches the shared-memory staging copy on it. That is the derivation half, shipped.
- **CUTLASS 3.x / CuTe** carries shapes and strides as static template types and derives
  thread-value partitioning from them — but has the kernel author build the predicate
  tensor by hand.
- **Dex** (2021) made index sets types, so two axes of equal size are still distinct types.
  That is the axis-identity half, years earlier.
- **Halide** (`TailStrategy`) and **TVM/TensorIR** have derived tiling, staging and tail
  predication for about a decade, from schedule values rather than from types.
- **TypeGPU** is the typed-resource layer for WebGPU in TypeScript and is better at that
  job than anything here; if you want typed buffers and a shader body in TS, use it.

What tessera puts together: axis identity *and* the ragged identity element as
obligations inside the same type that drives the derivation, on a TypeScript/WGSL
substrate. See [`docs/003-prior-art.md`](docs/003-prior-art.md), including what that
research did not look at.

## Scope principles

- Cut the language **by type, not by syntax**. The checker is the admission gate.
- `any`, dynamic indexing, prototype manipulation and friends must produce a **clear error** — never a silent fall-off into a slow path. (This is what AssemblyScript got right.)
- General-purpose TS→native migration is an explicit non-goal.

## Where it is

| | |
|---|---|
| `examples/matmul.kernel.ts` | 1024x768x512, bit-identical to a hand-written WGSL kernel, at parity on throughput |
| `examples/matmul-ragged.kernel.ts` | 1000x750x500, no axis dividing its block, 750000/750000 bit-exact |
| backends | direct WGSL (default) and MLIR (`--backend=mlir`), bit-identical to each other |
| `npm test` | MLIR byte-identity against a verified reference, 7 rejected negatives, mask placement, direct-backend validity |

Open: exactly one kernel family. Whether the types *derive* a second one — softmax or
attention over a ragged axis — or whether the emitter grows a code path is the question
that decides whether this is a compiler or a well-typed template collection. See
[`docs/003-prior-art.md`](docs/003-prior-art.md) §5 and
[`docs/002-performance.md`](docs/002-performance.md).

## Documents

- [`docs/001-language-surface.md`](docs/001-language-surface.md) — the surface, why block-level, the backend decision
- [`docs/002-performance.md`](docs/002-performance.md) — the invariant, how to measure, three refuted hypotheses
- [`docs/003-prior-art.md`](docs/003-prior-art.md) — what is not ours, and what is
