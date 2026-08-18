# 001 — Language surface: block-level with named axes

Status: **decided** (surface) / **open** (backend — see §5)
Date: 2026-08-14
Verification artifacts: `spike/surface/`

## 1. Decision

The kernel surface is **block-level, Triton-shaped, with named axes**. The user declares a tile
and a set of named axes, then writes a body over whole blocks.

```ts
const T = tiling(f32, 64, 64, 16);        // the only required perf knob
const M = axis("m", 1024, T.bm);          // Axis<"m", 1024, 64, "exact">
const N = axis("n",  768, T.bn);
const K = axis("k",  512, T.bk);

export const matmulRelu = kernel(
  { name: "matmul_relu_f32", tile: T, grid: [M, N], reduce: [K],
    bindings: [input("a", [M, K], f32), input("b", [K, N], f32), output("c", [M, N], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      acc = mma(a.tile(at.m, k), b.tile(k, at.n), acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  },
);
```

No thread indices, no barriers, no strides, no pointer arithmetic, and — the load-bearing part —
**no hand-written masks**. Every extent and block size is a numeric literal in the type, so
boundary conditions are synthesized. Raggedness is a type: `raggedAxis("m", 1000, 64)` makes every
load through that axis a `RaggedTile`, and the file does not compile until `.pad(0)` names the
identity element. That is the one fact the compiler genuinely cannot infer.

The whole surface contains **zero recursive conditional types and zero type-level arithmetic**.
Integer math runs once, in plain JavaScript, in tessera's own checker pass.

### Rejected alternatives

| Stance | Why not |
|---|---|
| Whole-tensor (`A @ B`, no loops) | Concentrates 100% of risk in the `linalg → SPIR-V` leg, which upstream does not ship (§5). Its own de-risk plan is a per-op WGSL template emitter — architecturally TF.js — which meets the v0 milestone while leaving the compilation thesis untested. **Deferred to v1 as a façade layer**, ~2-3 weeks on top of this core. The layering only works in this direction. |
| Native TS loop nests (explicit SIMT) | Requires the author to know register blocking and memory coalescing. CUDA engineers are the stated non-audience. |
| Type-system maximalist | Requires every tensor extent to be pre-registered in a generated table: `axis("m", 11008, 64)` — a real Llama FFN width — is rejected until you rerun codegen. Inverts the contract of a type system. Its *type vocabulary* was grafted wholesale. |

## 2. What the type system catches

At block level the caught bugs are ones that **do not throw**: a transposed coordinate, a tile-size
disagreement, looping the wrong axis, a non-annihilating pad reaching a reduction. These produce a
kernel that is correct at 1024 and quietly wrong at 1000. That is where shapes-in-types earn their
keep — at whole-tensor level a K-mismatch is a bug one runtime assert also catches.

Verbatim diagnostics, reproduced under `typescript@6.0.3`, `strict: true`:

```
b.tile(at.n, k)        TS2345: 'IdxOf<Axis<"n",768,64,"exact">>' is not assignable to
                               'IdxOf<Axis<"k",512,16,"exact">>'. Type '"n"' is not assignable to type '"k"'.
K mismatch             TS2345: ... Type '512' is not assignable to type '256'.
ragged, no .pad()      TS2345: ... Type '"this block is ragged: call .pad(identity) to say what
                               out-of-range lanes read"' is not assignable to type '0'.
.pad(1) into mma       TS2345: 'Tile<[64,16],f32,1>' is not assignable to 'Tile<[64,16],f32,0>'.
tiling(f32,64,64,32)   TS2345: Argument of type '32' is not assignable to parameter of type '8 | 16'.
f64                    TS2345: 'f64_does_not_exist_in_WebGPU_use_f32' is not assignable to 'DType'.
a.data[3]              TS2339: Property 'data' does not exist on type 'InputHandle<...>'.
```

Two design rules make these work and must be preserved:

- **`NoInfer` on every non-defining occurrence of a shared dimension.** Without it the K-mismatch
  call compiles clean and infers a confidently wrong type. ~12 signatures to audit.
- **The ragged sentinel is a string, not `never`**, so tsc prints the instruction rather than a
  type name. Failure carriers are intersected into *parameters*, never returned covariantly, which
  makes the `never`-smuggling trap structurally unreachable.

## 3. tsc is the first gate, never the only gate

**Confirmed by measurement, not assumed:** `spike/surface/neg/e11_subset.ts` contains a `while`
loop, an arrow-function closure, and `(a as any).whatever(acc)` — and it **type-checks clean, exit
code 0**. TypeScript's type system has nothing to say about statement forms, and `as any` escapes it
by design.

So the project principle "cut by type, not by syntax" needs its precise form:

- **Gate 1 — tsc.** Shapes, axis identity, dtypes, device features, host-boundary buffer shapes,
  binding counts. Live in the editor, no build step.
- **Gate 2 — the tessera admission pass.** A whitelist over ~35 `ts.SyntaxKind` values with a
  **default-reject** arm. Rejects `any`/`unknown` anywhere in a body expression's checker type
  (reporting the *entry* point, not the victim), computed member access, closures, `while`/`do`/
  C-style `for`, `try`/`throw`/`switch`, higher-order calls, recursion. Every rejection is a hard
  error with a source span, a rule number and a suggested rewrite. Never a slow path.

Ship Gate 2 as a `typescript@6.x` LanguageService plugin so both gates squiggle inline and the
two-error-source seam never appears to the user.

A default-accept walker is how this guarantee becomes a silent hole. Fuzz it against real-world
TypeScript and assert every unlisted kind is rejected.

## 4. Measured checker cost

Reproduced independently on this machine (M-series, 12 cores), `typescript@6.0.3`, `strict: true`:

| Corpus | Result |
|---|---|
| Surface + canonical matmul + ragged twin | clean, 0.83 s |
| 100 complete tiled matmul kernels | clean, 1.23 s |
| 300 complete tiled matmul kernels (4,202 lines) | clean, 1.94 s |
| 11 adversarial negatives | 10 fail as intended; e11 (subset) compiles clean — see §3 |

Declaration emit confirms exact inference with no widening (`Axis<"m",1024,64,"exact">`,
`Axis<"m",1000,64,"ragged">`).

Caveat: these are batch `tsc` runs. Red-squiggle latency in a live tsserver session on a 300-kernel
file is **unmeasured** — open `spike/surface/perf-n300.ts` in VS Code and time it before building
anything on the headline number.

## 4a. Verified: differential testing can be bit-exact

Measured on Apple/metal-3 via `spike/wgsl-baseline/`, against the hand-written
1024×768×512 kernel:

```
vs no-FMA reference   bit-exact 453143 / 786432  (57.62%)   maxAbsDiff 1.144e-5
vs FMA reference      bit-exact 786432 / 786432  (100.00%)  maxAbsDiff 0.000e+0
```

The plan assumed `kernel.reference()` would be bit-exact by applying
`Math.fround` to every f32-typed result. That is *almost* right and would have
failed in practice, because backends contract `acc + a*b` into a fused
multiply-add with one rounding instead of two. Modelling that is a one-line
change — the product of two f32 values is exact in f64 (two 24-bit significands
make at most 48 bits; f64 carries 53), so omitting the inner rounding *is* IEEE
fma semantics:

```js
acc = Math.fround(acc + Math.fround(a * b));   // two roundings — no FMA
acc = Math.fround(acc + a * b);                // one rounding  — FMA
```

**Consequence for the plan:** differential testing against node is a bit-exact
equality check, not a tolerance check. That is a materially stronger oracle than
budgeted for — a tolerance test cannot distinguish a 1-ulp codegen difference
from noise, and this one can. `kernel.reference()` should model contraction by
default and expose the non-contracting mode for backends that do not fuse.

A tolerance is still required for backends whose contraction behaviour is
unknown, and it must be the standard dot-product bound
`|error| <= gamma_K * SUM|a_i*b_i|` with `gamma_K = K*u/(1-K*u)`, `u = 2^-24` —
**not** `atol + rtol*|result|`. Reduction error accumulates against the partial
sums, not the result, so cancellation near zero fails a relative test no matter
how correct the kernel is. This is not academic: the first harness run reported
912 spurious failures out of 786432, every one of them a near-zero ReLU output.

## 5. The backend question is now OPEN, and leaning away from MLIR

The README's standing rule is that MLIR is in the stack **only** to ride existing dialects. Research
found that the dialect we most wanted may not be ridable:

- There is **no `LinalgToSPIRV` pass upstream at all**; the composed `ConvertToSPIRVPass` was
  demoted to test-only (PR #124301). The linalg→tile→fuse→promote→bufferize→map glue is precisely
  what upstream stopped shipping, and only IREE owns it.
- IREE's own `webgpu-spirv` target has been "Experimental" for four-plus years and files
  *"unknown SPIR-V builtin"* bugs against its own emission. **This answers the earlier question:
  IREE did not walk away from web out of disinterest.**
- The dialects with maintained, individually-tested `*ToSPIRV` conversions are exactly the ones a
  block-level surface enters at (`vector`, `scf`, `memref`, `gpu`, `arith`).

But that concession cuts both ways, and the honest statement of it is: using MLIR for
`vector`/`scf`/`memref`/`gpu`/`arith` plus a hand-assembled twelve-pass chain and a custom
fragment-legalization pass is **a convenient textual IR with SPIR-V conversions attached** — not the
structured-codegen infrastructure the MLIR dependency was justified by. A direct
tessera-IR → WGSL printer is ~600 lines and reaches the same v0 milestone far sooner.

**RESOLVED, partially and in favour of MLIR.** The spike ran (`spike/mlir-spirv/`,
reproduce with `./run-l0.sh`) and the highest-variance item did not materialise:
MLIR-emitted SPIR-V — carrying workgroup memory, a barrier and binding
decorations — passes `spirv-val` and is accepted by naga, producing clean WGSL of
the same shape as the hand-written baseline. There is **no ingestion wall** on
this toolchain.

Two things came with it:

- **A codegen rule.** `vector.load` over a scalar-element memref lowers to a
  `spirv.Bitcast` on a *pointer*, which Logical addressing forbids; it fails
  `spirv-val` and naga. Modelling the vector as the memref **element type**
  (`memref<Nxvector<4xf32>>`) is legal all the way to WGSL. So vectorisation is
  available, and `--no-vector` is a fallback rather than the only option.
- **The Homebrew bottle is sufficient.** Every per-dialect `*ToSPIRV` pass is
  present. The *composed* `--convert-to-spirv` is absent, confirming the PR
  #124301 demotion — which is fine, because the per-dialect passes are exactly
  where a block-level surface enters.

**L3 then closed it.** The full tiled matmul — `scf.for` with 16 loop-carried
accumulators, three flat storage buffers, two workgroup staging arrays, real
index arithmetic and a 4×4 register fragment — lowers through the same chain and
runs in Chrome **bit-identical to the hand-written kernel: 786432 / 786432,
maxAbsDiff 0.000e+0**. Since the hand-written kernel is bit-exact against the FMA
oracle, so is the MLIR-derived one.

**Backend decision: MLIR stays in the v0 pipeline.** The lean going in was away
from it; the evidence went the other way.

Two costs are now on the record, and only one of them is established:

- **Compile time is clearly worse — 325.9 ms pipeline creation against ~65 ms for
  the hand-written kernel.** naga emits unstructured `loop`/`continuing` with
  `phi_` variables (374 lines against 118) and Tint must re-structurize it. On a
  compile-in-the-browser story this is on the critical path, and it is a robust
  signal rather than measurement noise.
- **Runtime is ~1.5–1.6× slower**, established by paired within-run comparison
  after working out what the timings actually are. Chrome quantizes
  `timestamp-query` to **2^16 ns = 65.536 µs** — every one of eight samples across
  four sessions is an exact multiple — so that is the resolution floor. Both
  paired observations (17 and 8 quanta apart, far outside ±1 quantum) agree:
  1.61× and 1.50×. The cause is not established; naga's unstructured
  `loop`/`continuing` output is the first suspect. Not urgent while both sides
  are untuned, but no longer an unknown.

A third consequence: the direct WGSL printer scoped as week-4 insurance keeps its
value regardless, now as the *third oracle*. With the MLIR path bit-exact today,
any future divergence localizes to front end vs. conversion chain in one
comparison instead of a twelve-pass bisection.

**The week-0 spike, as originally specified:**

1. Hand-write the target WGSL for `matmul_relu_f32` (64×64×16 tile, `@workgroup_size(16,16,1)`,
   two `array<f32,1024>` workgroup tiles, 4×4 register fragment). Run it in Chrome against a JS oracle.
2. Hand-write the corresponding `spirv.module` MLIR. Run `mlir-translate --serialize-spirv`,
   `spirv-val`, then push through **both** `naga-cli` and a locally built Tint
   (`TINT_BUILD_SPV_READER=ON`). Diff against the hand-written WGSL, re-run in Chrome.
3. Separately: `mlir-opt` on a five-line module with one `memref.alloc()` in
   `#gpu.address_space<workgroup>` and one `vector.load`, through the full conversion chain.
   Two hours. `MemRefToSPIRV` is the least tolerant conversion in the set.

If step 2 does not survive in three days, the direct WGSL printer becomes the v0 product path and
MLIR moves to a later experiment. That decision is cheap in week 0 and expensive in week 8.

## 6. Toolchain notes

- `brew install llvm` (currently 22.1.8) ships `mlir-opt` and `mlir-translate` as a **bottle** —
  the formula lists `mlir` in `LLVM_ENABLE_PROJECTS` and its test block runs `bin/mlir-opt`.
  No source build needed. Pin one exact revision.
- MLIR is driven as a **subprocess over textual IR**. No C++ bindings, no LLVM build inside the
  Node toolchain. Python bindings are not needed and may be absent from the bottle.
- Pin `typescript@6.x`. **TS 7.0 (native Go) shipped to npm 2026-07-08 but has no `typescript.js`
  and no `ts.createProgram`**; the replacement API targets 7.1 (~Oct 2026). 6.x also ships the
  classic `LanguageService`, which is the plugin extension point §3 depends on.
  Keep every checker interaction behind a thin `TesseraProgram` façade (~6 methods) so the eventual
  port is a day, not a rewrite. Do this in week 2, not week 20.

## 7. Deferred, explicitly

Whole-tensor façade (v1, ~2-3 weeks) · dynamic shapes · autotuning (the legal tile space is finite:
~13 tiles × 7 workgroup shapes, exhaustively searchable in seconds) · rank > 2 · f16 end-to-end ·
implicit broadcasting and reshape · Halide-style re-scheduling · user-controlled barrier placement ·
tsc diagnostic cascade suppression (one grid-block mistake currently produces 7 errors).

## 8. The strongest argument against this decision

The whole-tensor surface is the only one that actually uses `linalg`, and riding existing dialects
is the sole stated justification for MLIR being in the stack. This design rides the least
interesting ones. Additionally, the "compiler-synthesizes your masks" advantage is a win over
Triton but **not** over a whole-tensor surface, where the user never writes a load and therefore
never writes a mask — the bugs this surface catches exist largely *because* it exposes tiles. That
is a circular defense and should be discounted accordingly.

It was chosen anyway because the whole-tensor stance puts all risk in the one leg with no surviving
public recipe, its escape hatch is architecturally TF.js, and it layers nowhere — a block layer
added later would bypass its scheduler entirely. This surface layers upward.
