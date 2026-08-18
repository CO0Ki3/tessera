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

The type-driven derivation is not new, and this project does not claim it. What follows
was checked by reading the sources, not by recalling them.

**Mojo's `LayoutTensor`** ([`max/kernels/src/layout/layout_tensor.mojo`](https://github.com/modular/modular/blob/main/max/kernels/src/layout/layout_tensor.mojo),
8717 lines, read 2026-08-18) already derives maskedness from compile-time layout:

```mojo
struct LayoutTensor[..., layout: Layout, ..., masked: Bool = False, ...]

def _tile_is_masked[layout: Layout, *tile_sizes: Int]() -> Bool:   # L185
    comptime for axis in range(layout.rank()):
        comptime if product(layout.shape[axis]) % tile_sizes[axis] != 0:
            return True

masked = Self.masked or _tile_is_masked[Self.layout, *tile_sizes]()   # L3244
```

So "the tile size is a compile-time parameter, raggedness is derived from it, and the
result becomes part of the type" is shipped, in production, and predates this. Its
handling is also more careful than a summary suggests: the primary mechanism is
**clipping the runtime shape** so out-of-range elements are never read at all —

```mojo
var shape_i = max(min(tile_sizes[i], cur_dim), 0)   # L3347
```

— which needs no identity element, because nothing out of range is touched. tessera made
the other choice: read out of range and substitute a named identity, so the guard is
branchless. Both are defensible; neither is a fix for the other.

Two things are absent from that file, and they are what tessera adds:

- **Axis identity.** `Layout` is a positional `IntTuple` in the CuTe lineage — the code
  indexes `layout.shape[axis]` over `range(layout.rank())`. Grepping 8717 lines for a
  name-based axis access returns **zero** against 30 positional ones. A transposed
  coordinate is legal there, and a compile error here.
- **The identity element as an obligation.** `masked` is one `Bool` for a whole tensor,
  not a per-axis fact, and the single data fill in the file is hardcoded —
  `fill=Scalar[Self.dtype](0.0)` (L5708) — with no parameter to change it. Nothing asks
  the author what an out-of-range lane should read. In tessera the operator demands it:
  `rowMax` takes `"exact" | "negInf"`, so `.pad(zero)` into a max does not compile.

Also relevant, and also not ours:

- **CUTLASS 3.x / CuTe** carries shapes and strides as static template types and derives
  thread-value partitioning from them — but its own `0y_predication.md` walks the author
  through hand-building a predicate tensor.
- **Dex** (2021) made index sets types, so two axes of equal size are still distinct
  types. That is the axis-identity half, years earlier, in a non-GPU array language.
- **Halide** (`TailStrategy`) and **TVM/TensorIR** have derived tiling, staging and tail
  predication for about a decade — from schedule values rather than from types.
- **`tfjs-backend-webgpu`** already generates WGSL with out-of-range guards; the narrow
  claim is not "generated guards in TS" but "guards driven by shapes the user declared as
  types".
- **TypeGPU** is the typed-resource layer for WebGPU in TypeScript and is better at that
  job than anything here. If you want typed buffers and a shader body in TS, use it — its
  TS→WGSL path is a faithful transliteration, so the masks stay yours to write.

What tessera puts together: **axis identity and the ragged identity element as obligations
inside the same type that drives the derivation**, on a TypeScript/WGSL substrate. See
[`docs/003-prior-art.md`](docs/003-prior-art.md), including what that research did not look
at — no academic search was run, and CubeCL was never reached.

## Scope principles

- Cut the language **by type, not by syntax**. The checker is the admission gate.
- `any`, dynamic indexing, prototype manipulation and friends must produce a **clear error** — never a silent fall-off into a slow path. (This is what AssemblyScript got right.)
- General-purpose TS→native migration is an explicit non-goal.

## Where it is

| | |
|---|---|
| `examples/matmul.kernel.ts` | 1024x768x512, bit-identical to a hand-written WGSL kernel, at parity on throughput |
| `examples/matmul-ragged.kernel.ts` | 1000x750x500, no axis dividing its block, 750000/750000 bit-exact |
| `examples/softmax.kernel.ts` | a second kernel family, 1024x750 with a ragged reduction axis, every row normalised and within 6 ULP |
| backends | direct WGSL (default) and MLIR (`--backend=mlir`), bit-identical to each other |
| `npm test` | MLIR byte-identity against a verified reference, 7 rejected negatives, mask placement, direct-backend validity |

The second kernel family was written to answer one question, with the criteria fixed in
advance: does a new schedule reuse the derivation, or does the emitter grow a code path?
Measured — the 73-line softmax schedule contains **no masking and no index arithmetic at
all**, 2 `emitLoad` and 1 `emitStore` calls, and its 64 bound checks come from the same
access layer the matmul uses. What is *not* shared is the schedule vocabulary: tessera
recognises a fixed set of body shapes and refuses anything else rather than approximating
it, so a third family needs a third recogniser. The honest summary is that the derivation
is general and the vocabulary is finite — roughly where Triton sits.
See [`docs/004-falsification.md`](docs/004-falsification.md).

## Documents

- [`docs/001-language-surface.md`](docs/001-language-surface.md) — the surface, why block-level, the backend decision
- [`docs/002-performance.md`](docs/002-performance.md) — the invariant, how to measure, three refuted hypotheses
- [`docs/003-prior-art.md`](docs/003-prior-art.md) — what is not ours, and what is
- [`docs/004-falsification.md`](docs/004-falsification.md) — compiler or template, criteria fixed before the experiment
