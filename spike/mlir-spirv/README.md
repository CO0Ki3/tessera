# spike/mlir-spirv — does MLIR earn its place?

Week-0, steps 2 and 3 of [`docs/001-language-surface.md`](../../docs/001-language-surface.md) §5.

This spike exists to answer one question before any compiler is written:

> Does SPIR-V emitted by MLIR survive a WebGPU consumer?

If it does, MLIR stays in the v0 pipeline. If it does not survive within three
days, the direct WGSL printer becomes the v0 product path and MLIR moves to a
later experiment. **That decision is cheap now and expensive in week 8**, which
is the entire reason this runs first.

## Why this is the risk

The project's standing rule is that MLIR is in the stack *only* to ride existing
dialects. Research found the dialect we most wanted may not be ridable: there is
no `LinalgToSPIRV` pass upstream, the composed `ConvertToSPIRVPass` was demoted
to test-only (PR #124301), and IREE's `webgpu-spirv` target has been
"Experimental" for four-plus years — filing *"unknown SPIR-V builtin"* bugs
against its own emission.

So MLIR emitting *a* SPIR-V binary proves nothing. IREE has been doing that for
years. The question is whether anything on the WebGPU side accepts it.

## Two independent questions

They fail for different reasons and must not be tested together.

| | question | rung |
|---|---|---|
| **Q1 ingestion** | does MLIR-shaped SPIR-V survive `spirv-val` → naga/Tint → Chrome? | L1, L2, L3 |
| **Q2 conversion** | does `mlir-opt` actually lower memref/vector/scf/gpu → spirv? | L0b |

Q1 is the headline risk. Q2 is the one with a known-tender spot:
`MemRefToSPIRV` is the least tolerant conversion in the set, and the pipeline
must map `#gpu.address_space<workgroup>` to `#spirv.storage_class<Workgroup>`
and flatten all subviews to base+offset *before* it runs.

## The ladder

Climb in order. Writing a full tiled matmul in the SPIR-V dialect by hand is
several hundred lines of very verbose IR, and if it fails at the bottom there is
no way to tell a bug in the hand-written IR from a real ingestion wall. Each
rung isolates one variable.

- **L0a — probe.** `./probe.sh`. What does the toolchain actually have? Does the
  Homebrew bottle ship SPIR-V serialization and the per-dialect conversion
  passes at all? Answered by asking the tools, not from memory.
- **L0b — the two-hour conversion test (Q2).** A five-line module: one
  `memref.alloc()` in `#gpu.address_space<workgroup>` and one `vector.load`,
  pushed through the full conversion chain. If this fails, the `--no-vector`
  scalar fallback moves from Plan B to Plan A immediately.
- **L1 — the trivial kernel (Q1).** A compute shader that writes a constant to a
  storage buffer. Serialize → `spirv-val` → naga → WGSL → run in the
  `wgsl-baseline` harness. This is the rung that actually answers the headline
  question, and it is small enough that any failure is unambiguous.
- **L2 — workgroup memory and a barrier.** The first rung that exercises the
  tender part: `Workgroup` storage class, `OpControlBarrier`. Everything real
  needs it and it is where an ingestion wall would plausibly appear.
- **L3 — the matmul.** Only once L1 and L2 are green. Diff the resulting WGSL
  against the hand-written `spike/wgsl-baseline/matmul.wgsl` and re-run it in
  the same harness.

L3's comparison is stronger than originally planned: the baseline spike proved
the GPU/CPU differential can be a **bit-exact equality check** rather than a
tolerance check (see `docs/001-language-surface.md` §4a). So "does the
MLIR-derived kernel compute the same thing" has an unambiguous answer, and any
divergence localizes immediately instead of requiring a twelve-pass bisection.

## RESULTS — the ingestion wall did not materialise

Run `./probe.sh` then `./run-l0.sh` to reproduce. Measured on LLVM 22.1.8
(Homebrew bottle), SPIRV-Tools, naga 30.0.0, Apple arm64.

### Q1 ingestion: **yes**

MLIR-emitted SPIR-V passes `spirv-val --target-env vulkan1.1` and is accepted by
naga, producing clean, idiomatic WGSL — including **workgroup memory, a barrier,
and binding decorations**, which are the parts that would plausibly break:

```wgsl
var<workgroup> _workgroup_mem_0_: type_3;
@group(0) @binding(0) var<storage, read_write> staged_arg_0_: type_6;

fn staged_1() {
    let _e10 = staged_arg_0_.member[0u];
    _workgroup_mem_0_.member[0u] = _e10;
    workgroupBarrier();
    return;
}
@compute @workgroup_size(16, 1, 1) fn staged() { staged_1(); }
```

That is the same shape as the hand-written `spike/wgsl-baseline/matmul.wgsl`.
Since the test kernel already carries workgroup memory, a barrier and bindings,
**L1 and L2 were cleared together**.

### Q2 conversion: **yes, with one concrete codegen rule**

Three ways of expressing the same staged load:

| | form | spirv-val | naga |
|---|---|---|---|
| C | `vector.load` over a **scalar-element** memref | ✗ invalid | ✗ rejected |
| D | scalar `memref.load` (the `--no-vector` fallback) | ✓ | ✓ |
| E | `memref.load` where the **element type is `vector<4xf32>`** | ✓ | ✓ |

C's failure is specific and worth stating precisely, because it is *not* "MLIR
emits bad SPIR-V":

```
spirv-val: OpLoad Pointer '23' is not a logical pointer.
naga:      invalid as type [10]
```

`--convert-vector-to-spirv` lowers a wide load over a scalar memref to a
`spirv.Bitcast` on a **pointer** (`ptr<f32>` → `ptr<vector<4xf32>>`). The Logical
addressing model forbids that — pointers must be derived from `OpAccessChain`.

**So the rule for tessera's codegen is: model vectors as the memref ELEMENT
TYPE, never as a wider load over a scalar memref.** E proves that path is legal
all the way to WGSL, which means vectorisation is available and the `--no-vector`
scalar fallback is not the only option. Finding this in week 0 is exactly what
the spike was for; discovering it in week 7 would have meant rewriting the
staging codegen.

### Toolchain facts worth not rediscovering

- The Homebrew `llvm` bottle ships **every** per-dialect `*ToSPIRV` pass we need,
  plus `--gpu-kernel-outlining`, `--spirv-lower-abi-attrs`, `--spirv-update-vce`,
  `--spirv-attach-target` and `--map-memref-spirv-storage-class`. No source build.
- `--convert-to-spirv`, the *composed* pass, is **absent** — confirming the
  research finding that it was demoted to test-only (PR #124301). The per-dialect
  passes are what remain, and they are what a block-level surface enters at.
- `mlir-translate --serialize-spirv` needs **`--no-implicit-module`**, or it wraps
  the input and fails with *"expected a 'spirv.module' op, got 'builtin.module'"*.
- `--convert-gpu-to-spirv` deliberately **leaves the source `gpu.module` in
  place** for a host pipeline to consume, and `mlir-translate` registers only the
  builtin and spirv dialects — so the leftover module is a parse error. Neither
  `--symbol-dce` nor `--gpu-module-to-binary` removes it; `extract-spirv-module.py`
  slices it out.

### What this does and does not settle

It settles the item flagged as highest-variance: **there is no ingestion wall.**
The fear was that MLIR-shaped SPIR-V would be rejected by WebGPU consumers, on
the evidence that IREE files *"unknown SPIR-V builtin"* bugs against its own
emission. On this toolchain, for a kernel using the constructs the matmul needs,
it is accepted.

It does **not** settle L3. The tested kernel is a few operations; the matmul adds
`scf.for` loops, three buffers, real index arithmetic and a 4×4 register
fragment. Those are more conversion surface and more chances to hit another
Logical-addressing constraint. The next rung is to lower the actual matmul and
diff the resulting WGSL against the hand-written baseline — a comparison that is
now unambiguous, because the baseline spike established a **bit-exact**
differential check rather than a tolerance one.

### L3: the matmul — **bit-identical to the hand-written kernel**

`gen-matmul-mlir.py` emits the MLIR for the same tiled GEMM as
`spike/wgsl-baseline/matmul.wgsl` — constants folded in Python from the tile and
axis literals, MLIR produced as text, no C++ bindings. Exactly the shape of
tessera's eventual emitter.

```
gen-matmul-mlir.py                                  239 lines MLIR
mlir-opt (gpu→spirv, lower-abi-attrs, update-vce)   484 lines spirv.module
mlir-translate --serialize-spirv                   8684 bytes .spv
spirv-val --target-env vulkan1.1                   VALID
naga                                                374 lines WGSL
```

Run in Chrome on Apple/metal-3 against the hand-written kernel, same inputs:

```
── hand-written ──   gpu 1.835 ms   438.9 GFLOP/s   pipeline   0.6 ms (cached)
── MLIR-derived ──   gpu 2.949 ms   273.1 GFLOP/s   pipeline 325.9 ms

bit-exact   786432 / 786432  (100.0000%)
maxAbsDiff  0.000e+0
IDENTICAL
```

Both produce identical output statistics (min 0.000, max 37.139, mean 3.012,
393412 zeros). The hand-written kernel matches the FMA oracle 786432/786432, so
by transitivity **the MLIR-derived kernel is bit-exact against the CPU oracle
too.** Correctness of the whole path is established, not inferred.

A second codegen rule fell out. naga initially rejected the kernel:

```
Unsupported relational function: IsNan
```

`arith.maximumf` — and `arith.maxnumf` — carry IEEE NaN-propagation semantics,
which MLIR lowers using `OpIsNan`, which naga's SPIR-V frontend does not support.
`arith.cmpf ogt` + `arith.select` passes, and is also exactly `v > 0 ? v : 0`,
which is what the CPU oracle computes — so the working form is the faithful one.

### Cost: compile time is clearly worse, runtime is not yet measurable

**Pipeline creation: 325.9 ms for the MLIR-derived WGSL against ~65 ms for the
hand-written one on a cold cache.** That is a large, robust signal. naga emits
unstructured `loop`/`continuing` with `phi_` variables, and Tint has to
re-structurize it — 374 lines against 118 for the same kernel. On a
compile-in-the-browser story that cost is on the critical path.

**Runtime: ~1.5–1.6× slower, and this now survives scrutiny.** An earlier
revision of this file said the gap was not measurable. That was too conservative,
and working out why produced the more useful finding.

Chrome quantizes `timestamp-query` results. Every GPU timing this harness has
produced — eight samples across four sessions — is an **exact multiple of
2^16 ns = 65.536 µs**:

```
printed ms   ns        quanta
     1.311   1310720       20
     2.032   2031616       31
     1.901   1900544       29
     2.949   2949120       45
     1.835   1835008       28
     1.769   1769472       27
     1.573   1572864       24
     1.049   1048576       16
```

Eight for eight is not coincidence. So the resolution floor is 65.536 µs, not
whatever precision the printed milliseconds imply, and the documented "55%
variance between identical runs" was genuine variance rather than quantization
noise.

That reframes the comparison rather than invalidating it. Both kernels run back
to back inside one session, so a **paired** comparison controls for clock and
thermal state in a way cross-session numbers cannot — and both paired
observations agree:

```
hand 28q vs tessera 45q   = 1.61x
hand 16q vs tessera 24q   = 1.50x
```

Separations of 17 and 8 quanta are far outside ±1 quantum, so the gap is real at
this resolution. What is *not* established is why, or whether it survives tuning:
plausible causes are naga's unstructured `loop`/`continuing` output defeating
some optimisation Tint would otherwise apply, or the extra `bitcast<i32>`
comparisons in the generated loop bounds. Both are investigable, neither is
urgent while the kernel is untuned on both sides.

### Decision table, as originally written

| result | decision |
|---|---|
| L1+L2 green in ≤3 days | MLIR stays in the v0 pipeline; proceed to L3 |
| L1 fails at naga *and* Tint | ingestion wall is real. Direct WGSL printer becomes v0; MLIR deferred |
| L1 green, L2 fails | narrower problem. Try scalar-only (`--no-vector`) before abandoning |
| L0b fails | conversion problem, not ingestion. Scalar fallback becomes Plan A |

**Outcome: row 1, in well under a day.** Tint was not needed — naga answered the
question, and building Dawn can stay unbuilt until there is a reason.
