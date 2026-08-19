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

None of the derivation is new, and this project does not claim it. What follows was checked
by reading sources; where a claim rests on documentation rather than source, it says so.

**The derivation half is claimed several times over.** Mojo's `LayoutTensor` derives
maskedness from a comptime layout parameter (`_tile_is_masked`, L185; threaded into the
type at L3244). Halide's `TailStrategy` and TVM/TensorIR have derived tiling, staging and
tail predication for a decade, from schedule values. CUTLASS/CuTe derives thread-value
partitioning from static shapes. **Dex** (2021) made index sets types — so a transposed
axis being a type error is not new either — and its `tile()` / `CodaIx` derive both the
tile count and the ragged remainder from those types.

**What is not claimed is the enforcement.** Padding a masked reduction with the wrong
identity is a bug that appears only at the ragged edge and only for some data — zero-padding
a max is correct until a tail row happens to be entirely negative. Several systems ask the
author for that value:

| | asks | checks it against the operator |
|---|---|---|
| Triton | `tl.load(p, mask=m, other=v)` — and `other` is **optional**, undefined if omitted | no |
| NVIDIA cuTile | `padding_mode=zero_pad` | no |
| CubeCL / cubek | `read_masked(mask, list, index, value)`; `read_checked` defaults to zero | no |
| Futhark, Julia, Finch.jl, HeteroCL | `ne` / `init` / `Element(0.0)` / `hcl.reducer(init, …)` | no |
| tessera | `.pad(negInf)` | **yes** — `rowMax` takes `"exact" \| "negInf"` |

cubek documents the failure mode nearly verbatim — an identity is what *"a masked read past
an operand's valid extent must return instead of a shared zero, since zero is Sum's identity
but biases Max toward it"* — and enforces it with a hand-maintained `match` at two call
sites. That is the best argument that this is worth checking: the bug class is somebody
else's scar, not a hypothetical.

**Two honest scopings.** Mojo and Dex never need an identity, because neither pads: Mojo
clips the runtime shape, Dex gives the tail a smaller dependent type. So this differentiator
holds among padding-based designs — a populous category — and not universally. And
CubeCL's `cubecl-wgpu` backend emits WGSL, so "compile-time-driven GPU kernel compiler
targeting WGSL" is occupied; what is unrefuted is the **TypeScript surface**.

**TypeGPU** is the typed-resource layer for WebGPU in TypeScript and is better at that job
than anything here. If you want typed buffers and a shader body in TS, use it — its TS→WGSL
path is a faithful transliteration, so the masks stay yours to write. tessera does not
compete with it; it **runs on it**. The two fit together at exactly one seam: TypeGPU owns
buffers, bind group layouts, bind groups and readback, while pipeline creation stays raw,
because `TgpuComputePipeline`'s descriptor is `{ compute: TgpuComputeFn }` and a
`TgpuComputeFn` cannot be built from emitted WGSL. See
[`docs/005-runtime.md`](docs/005-runtime.md).

Full accounting, including what three research passes could not look at and why, in
[`docs/003-prior-art.md`](docs/003-prior-art.md).

## Scope principles

- Cut the language **by type, not by syntax**. The checker is the admission gate.
- `any`, dynamic indexing, prototype manipulation and friends must produce a **clear error** — never a silent fall-off into a slow path. (This is what AssemblyScript got right.)
- General-purpose TS→native migration is an explicit non-goal.

## Where it is

| | |
|---|---|
| `examples/matmul.kernel.ts` | 1024x768x512, bit-identical to a hand-written WGSL kernel and at parity on throughput (6 quanta against 7, inside one) |
| `examples/matmul-ragged.kernel.ts` | 1000x750x500, no axis dividing its block, 750000/750000 bit-exact |
| `examples/softmax.kernel.ts` | a second kernel family, 1024x750 with a ragged reduction axis, every row normalised and within 6 ULP |
| `examples/attention.kernel.ts` | `softmax(Q·Kᵀ)·V` — two contractions accumulating over different axis sets, with the score fragment redistributed through workgroup memory between them; 0 violations of the f32 error bound |
| backends | direct WGSL (default) and MLIR (`--backend=mlir`), bit-identical to each other |
| `spike/wgsl-baseline/typegpu-runner.js` | all four kernels with TypeGPU owning buffers, layout, bind group and readback — bit-identical output, and free, sharing the raw runner's timed path verbatim |
| `npm test` | MLIR byte-identity against a verified reference, 7 rejected negatives, mask placement, direct-backend validity, manifest↔WGSL↔TypeGPU layout agreement |

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
- [`docs/005-runtime.md`](docs/005-runtime.md) — the host layer, and what TypeGPU structurally cannot absorb
