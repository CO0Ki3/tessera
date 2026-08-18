# spike/wgsl-baseline — the target artifact

Week-0, step 1 of the plan in [`docs/001-language-surface.md`](../../docs/001-language-surface.md).

A hand-written tiled GEMM with a fused ReLU, plus the differential harness that
proves it right. **This is what tessera must learn to emit.** Before writing a
compiler it is worth being able to write, by hand, the thing the compiler is
supposed to produce — otherwise the codegen has no target.

```
C[1024,768] = relu(A[1024,512] @ B[512,768])     f32

tile        64 x 64 x 16      workgroup 16x16x1 = 256 invocations
fragment    4 x 4 per invocation   = (64*64) / 256, a constant fold
workgroup   8192 B of the 16384 B guaranteed floor
grid        12 x 16 workgroups
```

Every extent is a compile-time literal and every axis divides its block
exactly, so the shader contains **no bounds checks and no masks at all**. That
is the design's central claim made concrete: staticness buys you the absence of
boundary code. The ragged twin (1000×500×750) is a separate artifact and is
where the masks reappear — that comparison is the headline demo, and it is
deliberately not built yet.

## Files

| | |
|---|---|
| `matmul.wgsl` | the kernel. The artifact everything else exists to validate |
| `oracle.js` | seeded inputs, `Math.fround`-exact CPU reference, `allClose` |
| `sim.mjs` | CPU simulation of the kernel's own index arithmetic — runs without a GPU |
| `run.js` | WebGPU host: device, limits, compile, dispatch, readback, timestamps |
| `index.html` | the index — one page per thing being verified |
| `matmul.html` | hand-written vs both tessera backends, measured and diffed |
| `ragged.html` | the same matmul where no axis divides its block |
| `rowwise.html` | softmax and layernorm, sharing one emitter |
| `measure.js` | warm up, repeat, interleave, report the minimum |

## Running it

**1. Index arithmetic, no GPU needed:**

```bash
node sim.mjs          # 128x128x64, ~50 ms
node sim.mjs --full   # 1024x768x512, ~2 s
```

`sim.mjs` mirrors `matmul.wgsl` statement for statement, modelling
`workgroupBarrier()` as a phase boundary — every invocation finishes staging
before any invocation starts accumulating, which is the exact meaning of a
barrier in a race-free kernel. It must match `oracle.js` **bit-exactly**, since
both accumulate in the same order with the same roundings and JS has no FMA
contraction. Any difference at all is an index bug, not a rounding artefact.

Status: **passing**, 786,432 / 786,432 elements bit-identical at full size.

This catches the most likely class of error — the flattening of a 64×16 tile
across 256 invocations, and the `ty*TM+m` / `tx*TN+n` fragment mapping — without
touching a GPU. It does **not** validate WGSL syntax; that happens in step 2.

**2. On the actual GPU:**

```bash
python3 -m http.server 8080
# then open http://localhost:8080/ in Chrome — the index lists each page
```

Regenerate every shader first with `npm run demo` from the repo root.

A local server is required: `run.js` fetches `matmul.wgsl`, and `file://`
origins block that. The page reports adapter info, the real device limits
against what the tile assumes, WGSL compilation diagnostics, timing, and the
differential result.

## Finding: the GPU contracts, and modelling that costs one line

The first run of this harness reported **912 / 786432 elements outside
tolerance**, with `maxRelDiff` 9.3e-3. That looked like a kernel bug. It was a
harness bug, and chasing it produced the most useful result of the spike.

Two observations resolved it.

**1. `atol + rtol * |result|` is the wrong instrument for a reduction.** Rounding
error in a dot product accumulates against the magnitude of the PARTIAL SUMS, not
of the final result. When cancellation drives a result near zero, the error does
not shrink with it — so near-zero outputs fail a relative test no matter how
correct the kernel is. Roughly half of a ReLU's outputs sit at or near zero,
which is why this bit here specifically. Every one of the 912 failures was a
near-zero element; the worst *absolute* error, 1.14e-5 on a value of 18.11,
passed comfortably.

**2. FMA contraction is exactly modellable, so it does not have to be absorbed
into a fudged tolerance.** The product of two f32 values is *exact* in f64 — two
24-bit significands make at most 48 significant bits and f64 carries 53 — so
omitting the inner rounding computes `round_f32(acc + a*b)` with a single
rounding, which is IEEE fma semantics:

```js
acc = Math.fround(acc + Math.fround(a * b));   // two roundings — no FMA
acc = Math.fround(acc + a * b);                // one rounding  — FMA
```

Running both references and asking which one the GPU matches turns a guess into
a measurement. On Apple/Metal the answer is unambiguous — the two CPU oracles,
differing only in that one rounding, **reproduce the GPU's statistics exactly**:

```
                     GPU vs no-FMA      no-FMA vs FMA (no GPU involved)
maxAbsDiff           1.144e-5           1.144e-5
maxRelDiff           9.306e-3           9.306e-3
naive failures       912 / 786432       912 / 786432
```

Not "consistent with" — identical. The backend fused, and that single rounding
difference accounts for the entire discrepancy.

**So the plan's bit-exact node oracle is achievable after all**, which reverses
this file's earlier conclusion. `kernel.reference()` should model contraction by
default, because backends contract. What remains genuinely necessary is a
principled bound for the cases where you cannot know what the backend did — and
that bound is the standard one,

```
|computed - exact|  <=  gamma_K * SUM |a_i * b_i|,    gamma_K = K*u / (1 - K*u),  u = 2^-24
```

which `withinErrorBound()` checks per element. On this kernel the observed
difference uses **0.29% of the permitted budget** (worst ratio 0.00288, zero
violations). That is a falsifiable check: inside the bound, or the kernel is
wrong. It is not a tolerance anyone tuned until the test went green.

This is `wgsl-numerics` — the ecosystem gap the research turned up — in its
smallest useful form.

| symptom | meaning |
|---|---|
| worst ratio < 1 | inside what f32 permits. Correct |
| worst ratio > 1 | real bug — floating point does not explain it. Suspect barriers or `As`/`Bs` staging |
| bit-exact vs FMA ≫ vs no-FMA | the backend contracted (expected on Metal) |
| output all zeros | the dispatch never ran, or every accumulator stayed 0 |
| `zeros` ≈ half the elements | correct — ReLU on roughly symmetric inputs |

## Measured (Apple, metal-3)

Status: **passing**, 786432 / 786432 elements bit-identical to the FMA reference.

```
                    run 1        run 2
compile             65.2 ms      1.6 ms     <- pipeline cache warm on the second run
gpu (timestamps)     1.311 ms    2.032 ms   <- 614 vs 396 GFLOP/s
wall (submit→done)  18.1 ms      6.9 ms     <- 45 vs 117 GFLOP/s
cpu oracle         924 ms     1393 ms
```

**Read the variance, not the numbers.** Two runs of an identical kernel on an
idle machine disagree by 55% on GPU time, and wall time moved the *opposite*
direction from GPU time. Clock ramping, thermal state and compositor contention
all land inside a single-dispatch measurement, and nothing here separates them.

This is a live demonstration of why a measurement layer is its own piece of work
rather than a helper function — the ecosystem research flagged it as the highest
-leverage support project for exactly this reason, and here it is happening on
the first kernel we ever ran. Treat 400-600 GFLOP/s as an order-of-magnitude
yardstick for what an untuned hand-written kernel achieves, and do not compare
two implementations on numbers gathered this way.

## Timing caveats

`wallMs` spans submit → `onSubmittedWorkDone` and includes queue latency, so it
is a floor on cost, not a kernel measurement. `gpuMs` comes from
`timestamp-query` at pass boundaries — the only available granularity, since
in-pass `writeTimestamp` was removed from the spec for all implementations
(gpuweb#2190).

**Chrome quantizes those timestamps, and this harness measured the quantum.**
Every GPU timing it has produced — eight samples across four sessions — is an
exact multiple of **2^16 ns = 65.536 µs**. The harness now reports the quanta
count next to each timing for exactly that reason: the printed milliseconds imply
a precision that does not exist, and a difference smaller than one quantum is
invisible.

The practical rule that follows: compare kernels **paired within one run**, where
clock and thermal state are shared, and only believe a difference of several
quanta. Cross-session absolute numbers on this harness have ranged 1.05–1.84 ms
for the *same* kernel, so they carry almost no information on their own.

## Next

Step 1 is done when the page shows PASS on real hardware. Then:

- **Step 2** — hand-write the equivalent `spirv.module` MLIR, run
  `mlir-translate --serialize-spirv` → `spirv-val` → both `naga-cli` and a Tint
  built with `TINT_BUILD_SPV_READER=ON`, diff the resulting WGSL against this
  file, and re-run it here. Needs `brew install llvm` (the bottle ships
  `mlir-opt` and `mlir-translate`).
- **Step 3** — two hours: `mlir-opt` on a five-line module with one
  `memref.alloc()` in `#gpu.address_space<workgroup>` and one `vector.load`
  through the full conversion chain. `MemRefToSPIRV` is the least tolerant
  conversion in the set.

If step 2 does not survive within three days, the direct WGSL printer becomes
the v0 product path and MLIR moves to a later experiment. See
`docs/001-language-surface.md` §5.
