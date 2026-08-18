# 002 — Performance: the invariant, what is ruled out, and the plan

Status: **invariant met** by the direct backend (117% of hand-written); the MLIR
path remains at 64% and is no longer the default
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
> Current: **met**. Direct backend 6 quanta vs 7 hand-written — at parity, and
> inside one quantum, so "indistinguishable" is the honest reading rather than
> "faster". The MLIR backend remains at 11 quanta (64%) and is no longer the
> default; see §5.

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

**The method is now validated by reproduction**, which is the property the first
one lacked entirely. Two independent sessions, six kernels, quoted as min/med/max
quanta:

```
                                    session 1        session 2
hand-written                        7 /  7 /  8      7 /  7 /  8
tessera                            11 / 11 / 12     11 / 11 / 12
probe A1  hand, read_write          7 /  7 /  7      7 /  7 /  8
probe A2  tessera, read            11 / 11 / 12     11 / 11 / 12
probe B   tessera, --no-opt        11 / 11 / 12     11 / 11 / 12
probe C   tessera, k unrolled      15 / 15 / 16     15 / 15 / 16
```

Identical to the quantum but for one sample's maximum, and every derived ratio
reproduced exactly (1.57× / 1.00× / 0.73× / 1.00× / 1.00×). Compare the method it
replaced, which put the *same shader* at 45q and 22q on different days.

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

## 5. RESOLVED — the tax is the MLIR chain

The direct WGSL printer was built to settle §4: same front end, same IR, same
codegen decisions, no MLIR. It deliberately mirrors the MLIR backend — same tile,
fragment layout, staging loops, clamp-and-select masking, conditional store,
rolled k loop — so that only the emission path differs.

Measured, both kernels, interleaved:

```
aligned  1024x768x512            min    GFLOP/s
  hand-written                    7q     1755
  tessera via MLIR               11q     1117
  tessera direct                  6q     2048

ragged   1000x750x500
  tessera via MLIR               12q      954
  tessera direct                  7q     1635      1.71x
```

**Row 1 of the table above: the tax is the chain.** The direct printer reaches
the hand-written kernel with the same IR behind it, so the 1.57× was never
tessera making worse decisions — it was the cost of
`IR → MLIR → SPIR-V → naga → WGSL`. Both backends are bit-identical to each other
and to the CPU oracle on both kernels (786432/786432 and 750000/750000), so this
is a pure emission-path comparison.

### Consequence: the direct backend is now the default

The project's standing rule, from the README: *MLIR is in the stack only to ride
existing dialects; otherwise it is pure overhead.* We never rode `linalg` — there
is no `LinalgToSPIRV` upstream — so MLIR has been serving as a convenient textual
IR with SPIR-V conversions attached. That convenience now has a measured price,
and it is 1.57×.

`--backend=mlir` opts back in. It stays for three reasons, none of them emission:

- **Second oracle.** Two independent backends agreeing bit-for-bit is a real
  correctness asset; a future divergence localises in one comparison.
- **Analysis and transformation.** Nothing in §6 has been attempted yet, and
  several of those levers are natural in MLIR and awkward by hand.
- It cost a week-0 spike to establish that the path works at all, and that
  knowledge does not expire.

### What this does not settle

The suspected mechanism — naga reconstructing `loop{}/continuing{}` with 144 phi
variables from an unstructured IR, where the direct printer emits WGSL's own
structured loops — is *consistent* with everything measured but is not itself
confirmed. Confirming it needs the MSL dump (§7). Both backends currently handle
exactly one kernel shape, so this is "direct wins at this scale", not a claim
about arbitrary kernels.

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

## 6a. RESOLVED — the cost is naga's reconstruction, not MLIR

The mechanism §5 left open is settled, and it did not need the MSL dump. The direct
WGSL was pushed through naga and back — `wgsl → spv → wgsl` — with MLIR entirely out
of the picture:

```
tessera direct                    8q
probe D: direct, naga round-trip  13q      1.63x
```

The round-trip acquires naga's `loop {} continuing {}` form and **not** the 144 phi
variables or the 4 bitcasts that only the MLIR path has, so it separates the two
candidates cleanly. It reproduces the whole gap on its own. **The loop form is the
cost**, and MLIR's contribution is only that going through SPIR-V means coming back
out through naga.

One confound was removed before measuring: naga inserts a workgroup zero-init that
neither other kernel pays (`if (local_invocation_index == 0u) { … }` plus a barrier).
Left in, the probe would have measured that instead. `make-probes.mjs` strips it and
throws if it cannot find it, so a change in naga's output fails loudly rather than
silently measuring the wrong thing.

## 6b. The unified emitter cost two quanta, and that is the trade

Replacing two hand-shaped emitters with one contraction-driven emitter moved the
direct backend from **6q to 8q** — about 25%. Against the hand-written kernel:

```
hand-written                7q
tessera direct (unified)    8q     1.14x, within one quantum
```

The 80% invariant is still met at 87.5%, and the correctness is unchanged
(786432/786432 bit-identical). But the regression is real and should be named rather
than absorbed: one emitter that derives its schedule costs a little against two that
were shaped by hand for their kernel. Whether that is worth reclaiming is a question
for when there is a reason to tune, and the measurement layer is already there.

## 7. Method note

Three WGSL-level hypotheses produced three refutations. That is the signal to
change instrument, not to invent a fourth hypothesis at the same level. When §5
runs, if it does not resolve the question, the next instrument is the generated
**MSL** — Chrome can be launched with Dawn's shader-dump flag, which turns "why
is it slower" from inference into reading.

Guessing is cheap to start and expensive to be wrong at. Two of the three probes
above cost under an hour each because they were two-sided and falsifiable; that
is the pattern to keep.
