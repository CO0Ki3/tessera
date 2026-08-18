# 002 — Performance: the invariant, what is ruled out, and the plan

Status: **invariant not met** (64% of hand-written; target 80%)
Measurement: `spike/wgsl-baseline/measure.js`

## 1. Why this is not optional

The user's alternative to tessera is writing WGSL by hand. Every percent the
generated kernel gives away is subtracted directly from the reason to use the
compiler at all. A kernel compiler that settles well below hand-written has not
shipped an unfinished feature — it has shipped a failed value proposition.
Triton's credibility rests on reaching roughly 90% of hand-tuned on real shapes;
that is the bar this is measured against, not "correct and somewhat slower".

So this is recorded as an **invariant**, not a task. Tasks get closed. Invariants
keep being checked.

> **The canonical kernel, generated, must reach at least 80% of the hand-written
> kernel's throughput, on the same machine in the same session.**
>
> Current: **64%** — 7 quanta hand-written vs 11 generated. **Not met.**

Re-check on every codegen change. `measure.js` makes this a number rather than an
opinion, so it belongs in CI as soon as there is CI.

## 2. How to measure, and why the obvious way is wrong

The first attempt at this timed each kernel with one dispatch, in a fixed order,
with no warm-up. It produced, for the *same* shader, 45 quanta in one session and
22 in another, and in one run reported that making the hand-written kernel worse
had made it 2.7× faster. A confident "tessera is 1.5× slower" was written into
these docs on that basis, and had to be retracted.

Four requirements, all of them load-bearing:

| | why |
|---|---|
| **warm up** | the first dispatches run at a low clock state. Everything measured before warm-up was ~4× slow |
| **repeat** | one sample per kernel measures the machine, not the kernel |
| **interleave** round-robin | running all of A then all of B lets clock drift masquerade as a difference between A and B |
| **report the minimum** | for fixed work the noise is one-sided — nothing makes a kernel faster than the hardware allows, so the floor converges on the kernel while the mean tracks how busy the machine was |

Report the spread alongside. A wide distribution should be visible, not averaged
into a confident-looking number.

**Chrome quantizes `timestamp-query` to 2^16 ns = 65.536 µs.** Measured, not
assumed: every sample the harness has ever produced is an exact multiple. Two
kernels are only distinguishable if they are several quanta apart, which is why
results are quoted in quanta rather than milliseconds.

## 3. What has been ruled out

Three hypotheses, each probed, all refuted. Recorded so nobody re-runs them.

| # | hypothesis | probe | result |
|---|---|---|---|
| A | **storage access mode** — naga emits `read_write` for inputs the surface knows are `read`; Metal maps read-only to `const device` | two-sided: force the hand kernel to `read_write`, mark tessera's to `read` | **1.00× both ways** |
| B | **integer ALU work** — the emitter recomputed loop-invariant indices, 24 ops per inner iteration | `--no-opt` vs `canonicalize`+`cse`+`licm`, which cuts it to 9 | **1.00×** |
| C | **unstructured control flow** — naga lowers the k loop to `loop{}/continuing{}` with 144 phi variables | `--unroll`, removing the loop (phi 144 → 76) | **0.73× — worse** |

C is the most informative failure: unrolling made it *slower* and grew the SPIR-V
from 8.1 KB to 24.5 KB. Whatever the loop costs, materialising sixteen copies of
its body costs more. Register pressure and instruction footprint are the
suspects. The loop structure is not simply a tax to be removed.

B is **kept despite buying nothing** — 600 fewer SPIR-V bytes, cannot hurt — but
it is no longer described as a performance measure.

Both `--no-opt` and `--unroll` remain in the CLI so any of this is reproducible.

## 4. The shape the problem probably has

All three refuted hypotheses were of the form "tessera's codegen makes a worse
choice". None survived. By elimination, the remaining shape is different: the
generated path crosses **four translation boundaries the hand-written path does
not**, and information can be lost at each.

```
hand      WGSL ------------------------------------------> Tint -> MSL
tessera   IR -> MLIR -> SPIR-V -> naga -> WGSL -----------> Tint -> MSL
```

If that is what the 1.57× is, then **no amount of improving our MLIR will close
it**, and the fix is to shorten the chain rather than to optimise within it.

## 5. The decisive experiment — already scoped

The **direct WGSL printer** was planned as week-4 insurance and as the third
oracle. It is also the A/B that settles §4: same front end, same IR, two
backends, both through `measure.js`.

| outcome | conclusion | consequence |
|---|---|---|
| direct printer ≈ **7 quanta** | the tax is the MLIR chain | the printer stops being insurance and becomes the performance path; MLIR keeps its value for analysis and transformation but stops being the emission path |
| direct printer ≈ **11 quanta** | the tax is our IR and codegen | MLIR is exonerated; optimise inside it, and §6 becomes the work |

Either answer is worth having, and it is a far sharper question than the
WGSL-level guessing that produced three refutations.

**Run it after ragged axes land**, not before: ragged changes codegen — masks
appear in the staging loads and the store — so anything measured first is
measured on a kernel that is about to change, and anything tuned first gets
retuned.

## 6. Levers held in reserve

Not to be pulled before §5 says which half of the pipeline to aim at, and not
before there is a real kernel with masks to measure.

- **Vectorised staging.** The l0b spike established the rule: vectors must be the
  memref *element type* (`memref<Nxvector<4xf32>>`), never a wide load over a
  scalar memref, which lowers to an illegal pointer bitcast. Vec4 staging is
  legal all the way to WGSL and untried.
- **Double buffering.** The tile deliberately uses 8192 B of the 16384 B floor,
  leaving 4096 B of headroom reserved for exactly this. Stage block k+1 while
  computing block k.
- **Tile autotuning.** The legal tile space is finite and small — the `TileFit`
  table is closed — so it is exhaustively searchable against the real adapter in
  seconds. Cheap once `measure.js` is the instrument, which it now is.
- **Workgroup shape.** Currently fixed at 16×16. It is a free parameter with the
  same finite search space.
- **Subgroups.** Chrome-only today. Not portable, so not a v0 lever.

## 7. Method note

Three WGSL-level hypotheses produced three refutations. That is the signal to
change instrument, not to invent a fourth hypothesis at the same level. When §5
runs, if it does not resolve the question, the next instrument is the generated
**MSL** — Chrome can be launched with Dawn's shader-dump flag, which turns "why
is it slower" from inference into reading.

Guessing is cheap to start and expensive to be wrong at. Two of the three probes
above cost under an hour each because they were two-sided and falsifiable; that
is the pattern to keep.
