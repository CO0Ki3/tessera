# 003 — Prior art, and the sentence this project can defend

Status: **closed to the accuracy available.** Three passes, three methods, one verdict.
The idea is partially claimed; what survives is narrower than earlier drafts said, and is
stated below in the form that survived contact with the field.

## 1. The sentence

> Every piece of what tessera derives from declared shapes — register fragment, dispatch
> extents, staging and barriers, index arithmetic, boundary handling — has been derived
> from declared shapes before, by Mojo, Halide/TVM, CuTe, CubeCL and Dex. What tessera
> adds is that **the two facts the derivation depends on are obligations in the type
> rather than conventions in the author's head**: a transposed axis name, and a pad value
> that is not the consuming reduction's identity, are both `tsc` errors — where Triton
> (`other=`), CubeCL (`read_masked`) and NVIDIA's cuTile (`padding_mode=zero_pad`) accept
> them and are correct only by the author's discipline.

## 2. How each system answers raggedness

This table is the most credible thing in this document, and it defuses the strongest
objection available to a reader who knows the field.

| system | answer to a ragged tail | needs an identity? |
|---|---|---|
| **Mojo** `LayoutTensor` | **clips** the runtime shape — `max(min(tile_sizes[i], cur_dim), 0)` (L3347); out-of-range is never read | no — nothing is padded |
| **Dex** | **shrinks the type** — `tile()` splits into `FullTileIx` plus a `CodaIx` whose size is a dependent `Fin` (`prelude.dx:2513-2582`); `tiled_matmul` has no bounds check at all | no — nothing is padded |
| **Triton** | pads: `tl.load(p, mask=m, other=v)` | asks, does not check |
| **cuTile** (NVIDIA, 2025) | pads: `padding_mode=zero_pad` by default | asks, does not check |
| **CubeCL / cubek** | pads: `read_masked(mask, list, index, value)`; `read_checked` defaults to `Scalar::default()` | asks, does not check |
| **tessera** | pads and substitutes branchlessly | **asks and checks against the operator** |

**So (b) is a differentiator among padding-based designs — a real and populous category —
and not a universal one.** Two of the three closest systems never need an identity because
they never pad. A reviewer who knows Mojo will say so, and they will be right.

## 3. What is claimed, and by whom

| | first | what they did |
|---|---|---|
| shape and tile size as compile-time types | CuTe/CUTLASS, Mojo, the dependent-array line | `Shape`/`Stride` as `Int<N>`; `layout: Layout` as a comptime parameter |
| raggedness derived from those types | **Mojo** | `_tile_is_masked` (L185) at comptime, threaded into the type (L3244) |
| **axis identity as a type error** | **Dex** (2021) | index sets are types: `for i:n. for j:m.` keeps `i` and `j` distinct at equal size |
| tiling + staging + tail predication from declared blocking | Halide (`TailStrategy`), TVM/TensorIR | a decade ago, from schedule values |
| **the identity element as a named per-operator fact** | **cubek** (tracel-ai, 2026) | `ReduceLeafKind::identity` maps Sum→0, Max→`min_value`, Min→`max_value` |
| generated WGSL bounds guards | `tfjs-backend-webgpu`, **CubeCL** (`cubecl-wgpu`) | |
| compile-time tensor shape arithmetic in TypeScript | `crankfunk/numtype` | |
| typed GPU resources in TypeScript | TypeGPU | |

### cubek had the same insight, and that is worth citing rather than hiding

`cubek-tile/src/ops/reduce/kind.rs` documents tessera's own motivating failure mode nearly
verbatim: an identity is what *"a masked read past an operand's valid extent must return
instead of a shared zero, since zero is Sum's identity but biases Max toward it (any
negative data)."*

A production kernel library hit this exact bug class and paid for it with a hand-maintained
`match` at two call sites. **That is the best available argument that (b) is worth
enforcing** — the insight is not speculative, it is someone else's scar. What cubek does
not do is enforce it: `read_masked` takes an arbitrary fallback, `read_checked` defaults to
zero, and a downstream user who zero-pads a Max still compiles.

### Triton, precisely, because the strongest line rests on it

`tl.load(pointer, mask=None, other=None)` checks exactly one thing about `other`: that it
is representable in the pointee's element type, plus broadcastability. It checks nothing
about the operator that consumes the tile. `tl.load(p, mask=m, other=0.0)` followed by
`tl.max(x, axis=0)` compiles identically to the correct `other=float("-inf")` version.

And `other` is **optional**: omit it and, per Triton's own documentation, the masked-out
value is *undefined*. So the most-used GPU kernel DSL does not merely fail to check the pad
against the reduction — it does not require one.

*Caveat kept deliberately:* this is from the public API documentation, not from reading
Triton's MLIR verifier passes. A dataflow check tying `other` to a downstream reducer is
documented nowhere and would be an advertised feature if it existed.

## 4. Corrections to earlier drafts of this file

Recorded rather than silently edited, because the pattern matters more than any one error.

- ~~"The identity-element obligation was never specifically searched for; the strongest
  genuinely-unclaimed sliver."~~ The **insight** is claimed, by cubek, a live
  WGSL-targeting project. Only the **enforcement** survives.
- ~~"Nothing asks the author what an out-of-range lane should read."~~ True of
  `layout_tensor.mojo`, false of the field. Triton, cuTile, CubeCL, HeteroCL
  (`hcl.reducer(init, freduce)`), Finch.jl (`Element(0.0)`), Futhark (`ne`) and Julia
  (`init`) all ask. **None checks.** The claim is "several ask; nobody checks."
- ~~"Dex … in a non-GPU array language."~~ Dex has a real CUDA backend —
  `src/lib/LLVM/CUDA.hs` compiles Imp IR to NVVM/PTX.
- ~~"Dex has axis identity without the GPU derivation."~~ Refuted by Dex's own prelude:
  `tile()` derives the tile count and the ragged remainder from index-set types. What Dex
  lacks is the staged execution model — flat SPMD, no shared-memory staging, no barriers.
- ~~"None of them run in a browser or target WGSL."~~ `cubecl-wgpu` emits WGSL.
  "Comptime-driven GPU kernel compiler that outputs WGSL" is **occupied**. The substrate
  claim narrows to the **TypeScript surface**, which is unrefuted.
- "Mojo has no axis names" is defended by a grep of one file, and MAX *does* have
  `SymbolicDim` — `TensorType(dtype, ["batch","seq_len"])`. They are erased before the
  kernel layer (zero hits under `max/kernels`), so the claim holds, but it must be stated
  as **"named dims exist in MAX's graph layer and are erased before `LayoutTensor` sees
  them."**

**Resolved in tessera's favour:** Mojo's barriers are *not* derived. `barrier()` is
hand-written at six sites in `_multistage_gemm_gpu.mojo`, and Modular's own teaching
example is a docstring about placing them by hand. No item leaves §1.

## 5. The nearest neighbour is now CubeCL / cubek

It displaces Mojo: the only system found that touches all three axes at once, and it lands
on the same output substrate.

What it does not do is the whole remaining gap. `pub struct Axis(pub u8)` — M, N and K are
runtime values of one type, so swapping them type-checks. `Transpose<Inner>::to_source_pos`
is *implemented* by swapping the elements of a positional `Coords2d`, which is exactly the
operation tessera's (a) makes a compile error. The bounds predicate
(`row < self.rows && col < self.cols`) is hand-written once per `Layout` impl, five times
over — the CUTLASS pattern, not a type-derived mask.

Two strengthenings worth banking: the 2026 formalisations of CuTe (`arXiv:2603.02298`,
`arXiv:2601.05972`) contain **zero** occurrences of mask, predicate, padding, boundary or
ragged — the hand-built predicate tensor is not a 2023 artifact, it is still outside the
algebra. And Axe (2026) uses the phrase "named axes" but has zero occurrences of "type
system" or "type error" and explicitly assumes shape divisibility.

## 6. Still open, and why a fourth pass is not worth running

Three passes, three methods — an arXiv API sweep, repo cloning with local grep, and
guessed-URL doc fetching — converged on the same verdict. The finding that mattered most
this round, cubek, changed the credits and not the verdict. That is the third consecutive
"verdict unchanged, attribution refined" result.

What remains is an **errand, not a research pass**: two papers behind ACM's Cloudflare with
no preprint — *"Structuring Arrays with Algebraic Shapes"* (Bachurski, Mycroft, Orchard,
ARRAY@PLDI 2025, 10.1145/3736112.3736141) and **Graphene** (ASPLOS 2023, by the CuTe
authors). Roughly 30 minutes each with an institutional login. Beyond that,
POPL/PLDI/OOPSLA/ICFP tables of contents were never browsed, so an ACM-only paper is
invisible to all three passes.

### The structural limit, stated once because it bounds every negative here

**All three passes had WebSearch at 200/200 before their first query.** Nothing in this
file was found by open-ended search. cubek was found by cloning a repo and noticing matmul
had moved out of it; the arXiv candidates came from a category sweep. The residual risk is
therefore not "someone did the whole conjunction and it was missed" — that would have to be
invisible to arXiv, GitHub and every vendor blog — but "someone wrote (b) down before cubek
did", which changes attribution rather than the claim, now that the claim is about
enforcement rather than insight.

That budget exhaustion is a harness configuration problem, not a research one, and it is
now the single largest determinant of what this research can see.

## 7. What does not depend on any of this

A TypeScript developer writing a WebGPU kernel today hand-writes their masks. That is
verified, it is the problem tessera solves, and it is unaffected by who thought of
type-driven tiling first.
