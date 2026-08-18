# 003 — Prior art, and the sentence this project can defend

Status: **the idea is partially claimed.** Keep building; change the pitch.

## 1. What is not ours

Two research passes, the second correcting the first. The honest position:

| | who did it first | what they did |
|---|---|---|
| Shape and tile size as **compile-time types** | CuTe / CUTLASS 3.x, Mojo, and the dependent-array line (Futhark, Accelerate, Idris) | `Shape`/`Stride` as `Int<N>` template types; `layout: Layout` as a comptime parameter |
| **Raggedness derived from those types, driving generated masks** | **Mojo `LayoutTensor`** | `_tile_is_masked[layout, *tile_sizes]()` is evaluated at comptime, its result becomes part of the resulting type (`masked = Self.masked or _tile_is_masked[...]`), and the DRAM→SRAM staging copy branches on it: `comptime if is_masked:` emits the bounds-checked path |
| **A transposed axis is a type error** | **Dex** (Google Research, 2021) | index sets are types: `for i:n. for j:m.` gives `i` and `j` distinct types even when `size n == size m` |
| Deriving tiling, staging and tail handling from declared blocking | Halide (`split` + `TailStrategy::GuardWithIf`/`ShiftInwards`), TVM/TensorIR | roughly a decade ago — from schedule *values* rather than types, but done |
| Generating WGSL boundary checks automatically | `tfjs-backend-webgpu`'s `matmul_packed_webgpu.ts`, vendored into onnxruntime-web | emits WGSL with out-of-range-returns-zero guards |
| Compile-time tensor shape arithmetic in TypeScript | `crankfunk/numtype` (Jul 2026) | matmul / broadcast / reduction shape rules |
| Type-safe GPU programming in TypeScript | TypeGPU | typed buffer layouts, bind groups, TS→WGSL transpiler, since v0.1 |

**"The compiler writes your masks so you never do" is Mojo's line, not ours.** It was
the most impressive-sounding claim in this project's pitch and it is the one most
clearly taken. Leading with it invites exactly the comparison it loses.

## 2. What is ours, as far as anything checked can tell

Dex has axis identity without the GPU derivation. Mojo has the derivation without axis
identity — its `Layout` is a positional `IntTuple` in the CuTe lineage, indexed by
`layout.shape[axis]` over `range(layout.rank())`, so a transposed coordinate is legal.
And Mojo's `masked` is one coarse `Bool` for the whole tensor, so nothing obliges the
author to name an identity element.

So the narrow, defensible position is the conjunction:

- **axis identity and the ragged identity element are obligations inside the same type
  that drives the derivation**, and
- the substrate is **TypeScript → WGSL**, where the state of the art today is genuinely
  hand-written masks — verified in TypeGPU's own tiled matmul and MNIST examples, and in
  `@typegpu/sort`, where even the identity element is a plain runtime parameter.

Mojo, CuTe and Dex do not compete for tessera's users: none of them run in a browser or
target WGSL, and a TypeScript developer cannot reach them. But they do constrain what
tessera may claim to have invented. Those are different questions and both answers matter.

## 3. The sentence

> tessera puts tensor shape, tile size and **axis identity** in TypeScript's type system
> as literal types, and derives the WGSL around a six-line kernel body from them —
> register fragment, dispatch extents, workgroup staging and barriers, flat index
> arithmetic, and every ragged-tail mask — so that a transposed axis, or a ragged axis
> with no declared identity element, is a `tsc` error rather than a wrong number.
>
> The type-driven derivation follows Mojo's `LayoutTensor` and CUTLASS/CuTe; axis
> identity as a type follows Dex. What tessera adds is both obligations inside one type,
> on a TypeScript/WGSL substrate.

Lead with **the eliminated error class**, not with the derivation.

## 4. Method note: the first pass was wrong, and why

The first prior-art sweep concluded the derivation half was unclaimed anywhere. It was
wrong — Mojo's `LayoutTensor` was found in minutes once someone thought to look. Mojo is
not obscure.

The cause is structural and worth remembering: **both passes exhausted their WebSearch
budget before the first query**, so every finding came from direct fetches against
already-named targets. That method cannot find a project you cannot already name, and
nobody named Mojo. Treat every negative result in this file as weaker than its phrasing.

## 5. Still not looked at

- **Zero academic search** in either pass — no arXiv, DBLP, ACM DL. Post-Dex work on
  named/nominal axes in dependently-typed array languages (ICFP/OOPSLA/PLDI 2022–2026) is
  the likeliest place an exact hit on the conjunction lives.
- **CubeCL** (Rust, const-generic tile sizes) — the most likely remaining collision; the
  sweep failed to reach its matmul source.
- **Mojo's barrier placement** — the staging copies are type-derived; whether the barriers
  are too is unconfirmed. If they are, another item leaves §2.
- **The identity-element obligation was never specifically searched for.** It is the
  strongest genuinely-unclaimed sliver and rests on the thinnest evidence.
- Nobody has read `layout_tensor.mojo` on this side yet. Do that before publishing.

## 6. What does not depend on any of this

A TypeScript developer writing a WebGPU kernel today hand-writes their masks. That is
verified, it is the problem tessera solves, and it is unaffected by who thought of
type-driven tiling first.
