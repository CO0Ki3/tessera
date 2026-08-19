# 004 — The falsification test: is this a compiler or a template?

Status: **complete, and verified on hardware.** Verdict by the rule fixed in §3: **compiler** — with one caveat the rule did not ask about, recorded in R7.

Written *before* attempting the second kernel, on purpose. The failure mode this
guards against is reading whatever happens as a success.

## 1. The claim under test

tessera has exactly one kernel family: matmul + relu, f32, one tiling, three axes, one
schedule. Its 18 boundary checks were derived *for that schedule*. The general claim —
"tessera derives boundary conditions from shape types" — is a claim about kernels that
have not been written.

If every new kernel means hand-extending `emit-wgsl.ts` with a new code path, tessera is
not a compiler. It is a growing library of hand-written schedules with a well-typed
front end, and there are better places to keep hand-written schedules.

## 2. The second kernel: softmax over a ragged axis

Chosen because it breaks the matmul mould in three specific ways, each of which is a
place the design can fail:

- **A different grid shape.** Softmax is one row (or row-block) per workgroup, reducing
  across the row: `grid: [m]`, `reduce: [n]`, output `[m, n]`. The surface currently
  requires `grid` to be *exactly two* axes with a separate reduce axis — a matmul shape.
- **A different reduction.** Not an accumulating product-sum into a register tile, but a
  cross-lane max, then a cross-lane sum of exponentials. There is no `mma` here.
- **A different identity element, and this is the sharp one.** For a masked *sum*, the
  identity is `0` — which is why `.pad(0)` works for matmul, and why the surface's type
  currently constrains `Pad extends 0`. For a masked *max*, the identity is `-inf`.
  Padding a max-reduction with `0` produces a kernel that is correct whenever the row
  contains a positive value and **silently wrong when every value is negative** — and
  only in the ragged tail. That is precisely the bug class this project exists to make
  unrepresentable, and the current surface would let it through.

## 3. Scoring, fixed in advance

New surface vocabulary and new emitter code for the computation are **expected and prove
nothing either way**. A compiler has to know what operations exist; adding `rowMax` and
`rowSum` is not evidence of anything.

The question is narrower. For each item, is it **reused from the axis types**, or
**written again by hand for softmax**?

| # | derived thing | reused = compiler | rewritten = template |
|---|---|---|---|
| 1 | dispatch extents from `ceil(extent/block)` | | |
| 2 | the load masks on the ragged axis | | |
| 3 | the store mask | | |
| 4 | flat index arithmetic from binding axes | | |
| 5 | workgroup staging + barriers | | |
| 6 | the identity element obligation | | |

**Verdict rule, fixed now:**

- **Compiler** — items 1-4 come out of the existing derivation with no softmax-specific
  masking logic, and item 6 generalises (the type demands an identity appropriate to the
  reduction, so `.pad(0)` into a max is a compile error).
- **Template** — softmax needs its own mask placement, or the identity obligation has to
  be special-cased per kernel, or `grid: [m]` requires the derivation to be forked.
- **Partial** — anything else. Say which items fell on which side and do not round up.

## 4. What a *good* failure looks like

If `.pad(0)` into a max-reduction currently type-checks, that is a **hole in the design,
found by the test working as intended**. The right response is to fix the surface so the
identity element is tied to the reduction operator, not to quietly widen the example
until it passes. Record the hole either way.

## 5. Pre-registered prediction

Written before starting, so it can be wrong in public:

- Items 1-4 will be reused. They depend only on axis extents and binding axes, neither of
  which knows what a matmul is.
- Item 5 will be **partially** reused: softmax needs a cross-lane reduction through
  workgroup memory, which is a different staging pattern from the matmul's two operand
  tiles. Expect the barrier placement to be new.
- Item 6 will **fail as currently designed** — `Pad extends 0` is hard-coded, so the
  surface will happily accept the wrong identity for a max. This is the interesting one.
- The `grid: [Axis, Axis]` shape will have to generalise, and that is surface work rather
  than derivation work.

If items 1-4 do not survive, that is the falsification and the recommendation in
`docs/003` §2 applies: keep the type-level idea, drop the emitter, and build it as a
layer on TypeGPU instead.


---

# Results

## R1. The surface was matmul-shaped, and that was one signature

Attempt 1 — softmax written against the unmodified surface — produced exactly two
errors, both about shape, neither about derivation:

```
spec.grid   Source has 1 element(s) but target requires 2
spec.reduce Type '64' is not assignable to type '16'      <- reduce axis bound to T.bk
```

The cause was that `kernel()` tied axes to tile dimensions **positionally**:
`grid[0]` to `T["bm"]`, `grid[1]` to `T["bn"]`, every reduce axis to `T["bk"]`. An
`Axis` already carries its own block, so that binding bought nothing and cost the
ability to express any kernel that is not two-parallel-axes-and-one-reduction.

Relaxing it to `grid: readonly [AnyAxis, ...AnyAxis[]]` and `reduce: readonly AnyAxis[]`,
with `tile` optional, made the softmax spec type-check. **Both matmuls still type-check
and the emitted MLIR is still byte-identical to the verified reference.**

One check moved from Gate 1 to Gate 2: "this axis is blocked at 32 but the tile says 64"
is now `TSA0051` from tessera's pass instead of a tsc cascade. docs/001 §7 listed
suppressing that cascade as outstanding work, so this is a small improvement rather than
a regression.

## R2. The derivation was mostly not matmul-shaped

Inventory of every line in the front end and IR that assumed the grid's shape:

```
frontend.ts:311   grid.length !== 2          a fence, not derivation
frontend.ts:315   reduce.length !== 1        a fence
frontend.ts:321-323  tile coherence, positional
ir.ts:95          dispatch: [grid[1].tiles, grid[0].tiles, 1]
```

Five lines. What is *not* in that list is the part that matters: mask placement
(`raggedNames` × binding axes), literal extraction from types, binding/axis consistency,
element counts, and the pad obligation. None of them know what a matmul is.

Generalising the dispatch to 1–3 grid axes was four lines.

**Item 1 (dispatch extents): reused.** The rest of the scorecard needs the emitter.

## R3. Item 6 failed, and deeper than predicted

The prediction was that `Pad extends 0` is hard-coded so the surface would accept the
wrong identity for a max. Measured:

```ts
a.tile(at.m, k).pad(-Infinity)
// Argument of type 'Tile<readonly [64, 16], f32, number>' is not assignable
// to parameter of type 'Tile<readonly [64, 16], f32, 0>'.
```

Two problems, not one:

1. `mma` demands `Pad extends 0` — as predicted.
2. **`-Infinity` has no literal type in TypeScript.** It is unary minus applied to
   `Infinity`, so it widens to `number`. The identity element for a max cannot be carried
   in the type as a numeric literal *even in principle*.

So the surface cannot express a non-zero identity at all. Padding a masked max with `0`
— the kernel that is right until every value in a tail row is negative — is not merely
allowed, it is the only thing sayable.

**The fix, per §4: change the surface, not the example.** Identities become named symbols
with singleton types rather than numbers, so the obligation attaches to the operator:

```ts
a.tile(at.m, k).pad(identity.zero)     // additive identity — what mma demands
x.tile(at.m, n).pad(identity.negInf)   // max identity — what maxReduce demands
```

That is a better design than the numeric one it replaces: the identity is tied to the
reduction operator by name, it is expressible, and `pad(identity.zero)` reaching a max is
a type error for the same structural reason `pad(1)` reaching a sum already is.

## Scorecard so far

| # | derived thing | verdict |
|---|---|---|
| 1 | dispatch extents | **reused** — 4 lines to generalise, no softmax-specific logic |
| 2 | load masks | not yet tested |
| 3 | store mask | not yet tested |
| 4 | flat index arithmetic | not yet tested |
| 5 | staging + barriers | not yet tested |
| 6 | identity obligation | **failed** — and the surface cannot express the right answer |

## R4. Item 6 fixed, and the identity turned out to travel

Identities became named singletons, so the obligation attaches to the operator.
`rowMax` takes `"exact" | "negInf"` and `mma` takes `"exact" | "zero"`, and the
dangerous mistake is now a compile error:

```
rowMax(x.tile(at.m, n).pad(zero), mx)
  Type '"zero"' is not assignable to type '"exact" | "negInf"'
```

That is the kernel which is right at 1024 and wrong at 750, for tail rows that happen
to be entirely negative — caught before it runs.

**An unregistered finding, and the better one.** Writing softmax produced this error on a
line I believed was correct:

```
rowSum(expTile(subRow(x.tile(at.m, n).pad(negInf), mx)), sm)
  Type '"negInf"' is not assignable to type '"exact" | "zero"'
```

The type system was right and the kernel author was sloppy. `exp(negInf - anything)` is
zero, so a block padded with the *max* identity comes out of `exp` padded with the
*additive* identity — which is precisely what the following `rowSum` needs. The identity
element **transforms through operations**, and that had to be written down:

```ts
expTile(...): Tile<S, D, P extends "negInf" ? "zero" : P>
```

Nobody asserted the two identities were interchangeable; they are not. One operation
converts one into the other, and now the type says which. This was not predicted, was
not in the scorecard, and is the most interesting thing the experiment has produced.

## R5. The access layer is schedule-independent — measured, not asserted

Masking and flat indexing were inline in the matmul staging loops. Extracting them into
`guards` / `emitLoad` / `emitStore`, which know only a binding, two coordinate
expressions, and which axes are ragged, produced **byte-identical WGSL for both matmul
kernels**. The logic was already generic; it simply was not factored.

Whether softmax can use it unchanged is the remaining measurement.

## Scorecard

| # | derived thing | verdict |
|---|---|---|
| 1 | dispatch extents | **reused** — 4 lines to generalise |
| 2 | load masks | **schedule-independent** — extraction left matmul byte-identical; softmax's use not yet emitted |
| 3 | store mask | same |
| 4 | flat index arithmetic | same |
| 5 | staging + barriers | not yet — softmax stages nothing, and needs cross-lane reduction scratch instead |
| 6 | identity obligation | **was broken, now fixed**, and generalised further than predicted |

## R6. The softmax emitter contains no masking logic

Measured, not asserted. The softmax schedule is 73 lines. Inside it:

```
select(   0        min(      0        bound comparisons  0
ragged    0        guards    0        extent             1   (a loop bound)

emitLoad  2 calls  emitStore 1 call
```

Every access goes through the shared access layer. The generated WGSL has **64 bound
checks on the ragged `n` axis and 0 on the exact `m` axis**, the pad is
`-3.4028235e38` because the kernel said `.pad(negInf)`, and the store is guarded:

```wgsl
let off = row * 750u + col;
let v = select(-3.4028235e38, x[min(off, 767999u)], col < 750u);
...
if (col < 750u) { y[row * 750u + col] = exp(v - mx[0]) / sm[0]; }
```

naga validates it, and the matmul suite is unchanged.

## R7. Verdict, and the caveat the rule did not ask about

| # | derived thing | verdict |
|---|---|---|
| 1 | dispatch extents | **reused** — 4 lines to generalise from 2 axes to 1–3 |
| 2 | load masks | **reused** — 0 lines of masking in the softmax path |
| 3 | store mask | **reused** — same |
| 4 | flat index arithmetic | **reused** — same |
| 5 | staging + barriers | **schedule-specific**, as predicted. softmax stages nothing and needs a rows×lanes scratch and two reduction barriers instead |
| 6 | identity obligation | **was broken, now fixed** and generalised further than predicted |

By §3's rule that is **compiler**: items 1–4 came out of the existing derivation with no
softmax-specific masking, and item 6 generalised — `rowMax` takes `"exact" | "negInf"`, so
padding a masked max with zero does not compile.

**The caveat, stated plainly because §3 did not think to ask.** tessera *recognises*
schedules; it does not compile an arbitrary body. `checkCanonicalBody` now matches two
patterns — mma-into-a-Frag, and rowMax-then-rowSum-then-store — and refuses anything else
rather than approximating it. A third kernel family will need a third recogniser.

So the honest statement is narrower than "compiler" and wider than "template":

> The **derivation** is general — masks, indexing, dispatch and the identity obligation
> are computed from the axis types and are shared across schedules, measured at zero
> lines of duplication. The **schedule vocabulary** is a fixed set, and grows one entry
> per kernel family.

That is roughly where Triton sits: `tl.load(..., mask=...)` is general, but the set of
things the language knows how to lower is finite and grows deliberately. The difference
that matters is which half had to be rewritten for kernel number two, and the answer
measured here is: not the derivation.

## R8. Verified on hardware

Apple/metal-3, 1024 x 750 with the ragged 46-column tail:

```
row sums      [0.999999700, 1.000000246]        every row normalised
bit-exact     317754 / 768000   (41.37%)
max ULP       6      at [59][242]
maxRelDiff    4.742e-7
```

Two checks, deliberately independent.

**The row sums need no reference at all.** A correct softmax normalises each row, so a
wrong tail mask changes that row's denominator and shows up here regardless of what the
oracle does. It matters that this is separate: the oracle and the kernel could in
principle share a misunderstanding, and this check cannot.

**Against the CPU oracle, 41% bit-exact is the expected number, not a poor one.** WGSL
does not require `exp` to be correctly rounded, so the GPU's `exp` and JS's `Math.exp` may
differ in the last place; that difference then propagates through the sum and the divide,
and most outputs end up one or two ULP apart. The meaningful figure is the maximum, and
6 ULP is comfortably inside what a permitted `exp` explains. A mask error would not be
subtle at this resolution — it would move a whole row's denominator and both checks would
fail by O(1).

This is a real difference from the matmul, where bit-exactness *was* the right bar because
`+`, `*` and fma are all correctly rounded. Which bar applies is a property of the
operations in the kernel, not a matter of how strict one feels like being.

### One more validator gap, found the hard way

The first hardware run failed to compile: `-3.4028235e38` looks like the f32 minimum and
is routinely written as one, but it exceeds f32::MAX (3.4028234663852886e38) and does not
round-trip. **naga validated all 57 occurrences and reported success**; only Tint rejected
it. A validator that accepts what the target rejects converts a build-time error into a
runtime one, so the check moved into the compiler (`assertF32Literals`) and into the suite.
Reported upstream as [gfx-rs/wgpu#10106](https://github.com/gfx-rs/wgpu/issues/10106).

---

# Part 2 — the third kernel family, and what it costs

Written before starting, same as part 1.

## The question

Part 1 established that the *derivation* is shared and the *schedule vocabulary* is not.
The open question that follows is about the second half: **is the marginal cost of a
schedule falling?** If schedule N costs the same as schedule N-1 forever, the design is a
template collection with an unusually good front end. If the cost falls, the vocabulary is
converging on something general.

## The kernel: layernorm over a ragged axis

```
y[m,n] = (x[m,n] - mean_n) / sqrt(var_n + eps),   var = E[x^2] - E[x]^2
```

Chosen because it is structurally unlike both existing schedules in one specific way, and
routine enough that the answer generalises:

| | matmul | softmax | layernorm |
|---|---|---|---|
| accumulators | 1 (a 2-D register fragment) | 1 (a row vector) | **2** (sum and sum-of-squares) |
| passes over the reduce axis | 1 | 3 | **1, carrying both** |

Neither existing schedule carries two accumulators through one loop. If the recogniser
generalises over "N accumulators, M passes" rather than gaining a third hand-written case,
that is the marginal cost falling.

Note that a masked lane must contribute **zero to both** sums, so `.pad(zero)` is correct
here — unlike softmax, where the max demanded `negInf`. That the two kernels need
different identities for the same axis is the point of naming them.

## Scoring, fixed in advance

Reusing the access layer is no longer in question — part 1 settled it. What is measured:

| | |
|---|---|
| **A. emitter** | how many lines does the layernorm schedule add, against softmax's 73? |
| **B. recogniser** | a third hand-written pattern, or does the second one generalise? |
| **C. derivation** | anything at all in `derive()` / the front end that needs a third case? |
| **D. surface** | new ops are expected; does anything *structural* have to change? |

**Verdict rule:** the vocabulary is converging if B generalises and C is empty. It is a
template collection with good ergonomics if B gains a third case and C needs a third
branch — which would be a fine thing to be, but should be said plainly rather than
discovered later by someone else.

## Prediction

- **A**: fewer lines than softmax. Layernorm is one pass where softmax is three, and the
  cross-lane reduction helper already exists — but it reduces two accumulators, so
  `acrossLanes` will need to take a list rather than one name.
- **B**: **will need a third case as written.** `checkCanonicalBody` counts specific op
  names (`mma`, `rowMax`, `rowSum`), so a new op means a new branch. The honest move if
  that happens is to generalise the recogniser rather than add the branch — the
  pre-registered response, same as part 1 §4.
- **C**: empty. Nothing in `derive()` knows about the reduction being performed.
- **D**: `rowSum` already exists and should serve both accumulators unchanged; `sqrt` and
  an `eps` constant are new. No structural change expected.


## Results, part 2

**Prediction B was right, and the pre-registered response was taken.** Adding layernorm
produced exactly the expected failure:

```
the softmax schedule is 2 rowFill, 3 loops, 1 rowMax, 1 rowSum, 1 store;
this body is 2 rowFill / 2 loops / 0 rowMax / 2 rowSum / 1 store
```

A third template. Rather than add the branch, the row-wise family is now **read** instead
of matched. `src/body.ts` parses a body into accumulators, an ordered list of steps, and a
small expression IR over the closed operator vocabulary. How many accumulators exist, how
many passes there are and what each does all come out of the statements the author wrote.

| | measured |
|---|---|
| **A. emitter** | **0 new lines.** One `emitRowwise`, 100 lines, serves softmax *and* layernorm |
| **B. recogniser** | **generalised**, not branched. Two schedules remain — `matmul` and `rowwise` — and `rowwise` is a parser, not a pattern |
| **C. derivation** | **empty**, as predicted. Nothing in `derive()` or the mask logic needed a third case |
| **D. surface** | 4 new operators (`sqTile`, `meanRow`, `rstdRow`, `mulRow`); `rowFill`, `rowSum` and `subRow` reused unchanged. No structural change |

Zero non-comment mentions of "layernorm" anywhere in `src/`.

**Verdict by the rule fixed in advance: the vocabulary is converging.** B generalised and
C is empty. The second kernel family cost a new emitter path; the third cost none.

### Two bugs the experiment found in the work it was testing

Both were caught by tools rather than by review, which is the point of having them.

1. **Source order was lost.** `RowBody` kept derived row values and passes in separate
   lists, so the emitter referenced `mu` before declaring it. In layernorm the two
   interleave — reduce, then derive a mean and a reciprocal standard deviation, then
   store — and softmax simply has no derived values, so nothing had exercised it. naga
   caught it: *"no definition in scope for identifier: `mu`"*. Steps are now one ordered
   list.

2. **The f32 literal guard added last round was too strict.** It required an exact round
   trip, and rejected `0.00001`. Almost no decimal literal is exactly representable in
   f32 — `0.1` would have failed too — and rounding to the nearest is normal and
   permitted. What Tint actually rejects is a magnitude beyond f32's range, so that is
   what the guard checks now. A guard that fires on correct input is worse than none,
   because the next person deletes it.

### What is still not general

`matmul` is still recognised by shape rather than read: `mma` into a `Frag`, one loop, one
store. Generalising it means giving the body IR a register-fragment accumulator alongside
the row vector, which is the same move again and is now the obvious next one. Until that
happens the honest statement stays two-part — the derivation is general, and the schedule
vocabulary is two entries, one of which is a parser and one of which is a template.


## R9. Verified on hardware — and the error-scale lesson recurred a third time

```
softmax    invariant  rows sum to 1        [0.999999700, 1.000000246]   PASS
           oracle     41.37% bit-exact, max 6 ULP                       PASS

layernorm  invariant  mean 0, variance 1   mean [-3.9e-8, 3.3e-8]
                                           var  [0.999967, 0.999973]    PASS
           oracle     61.71% bit-exact, max 1783 ULP                    initially FAILED
```

The invariant passed and the ULP check failed, which is the shape of a measurement
problem rather than a kernel problem. The worst element was `[748][249]`, where the GPU
gave `1.2533184e-5` against a reference of `1.2531563e-5` — an absolute difference of
**1.6e-9** on an output whose range is `[-1.89, 1.87]`.

layernorm computes `(x - mu) * inv`. Where `x` happens to sit near `mu` the result cancels
to near zero, while its error is inherited from operands of order one. ULP on the result
then counts thousands of them for an error that is 1.6e-9 in absolute terms.

**This is the third appearance of the same lesson**, and it is worth naming as a rule
rather than rediscovering:

| kernel | wrong scale | right scale |
|---|---|---|
| matmul | `rtol * \|result\|` — 912 spurious failures on near-zero ReLU outputs | `gamma_K * SUM \|a·b\|`, the partial sums |
| softmax | bit-exactness — `exp` is not correctly rounded | ULP on the result |
| layernorm | ULP on the result — cancellation makes it tiny | absolute error against the operand magnitude |

The scale is a property of the operations in the kernel. Choosing it correctly is not the
same as loosening a tolerance until the test passes, and the difference between those two
is whether you can say in advance which one applies.

With the right scale, measured on hardware:

```
softmax    41.37% bit-exact   max 6 ULP        maxAbsDiff 1.16e-9   PASS
layernorm  61.71% bit-exact   max 1783 ULP     maxAbsDiff 3.58e-7   PASS
                              3.2 ULP at a scale of 1.888, bound 9.00e-7
```

A correction to an earlier draft of this line: it claimed a margin of about 550×, computed
from the absolute difference at the worst-*ULP* element (1.6e-9) rather than at the worst-
*absolute* one (3.58e-7). The real margin is **2.5×** — comfortable, and the kernel uses
about 40% of an 8-ULP budget. The 8 is a choice; it should be revisited if a kernel ever
sits near it rather than being quietly widened.


## R10. matmul joins the parser; the emitters stay two, honestly

The last template was `matmul`, still recognised by counting operator names — "1 zeros,
N loops, 1 mma, 1 store". That is gone. `parseBody` now reads both kinds of body:

```
matmul     accKind frag   accs [{acc, frag, zero}]        steps  reduce -> storeFrag
softmax    accKind row    accs [{mx, row, negInf}, ...]   steps  reduce -> reduce -> store
layernorm  accKind row    accs [{s, row, zero}, ...]      steps  reduce -> derived -> derived -> store
```

The schedule now *falls out* of what the body declares:

```ts
const schedule = parsed.accKind === "frag" ? "matmul" : "rowwise";
```

Zero operator-name counting remains in the front end. The matmul MLIR is still
byte-identical to the verified reference, so the refactor changed nothing that is emitted.

**The emitters are still two, and that is the honest result rather than a shortfall.**
A fragment accumulates an outer product from two operands staged into workgroup memory; a
row accumulator folds one cell at a time from an operand read straight from global memory.
Those are different memory schedules, not different spellings of one. Unifying them means
expressing both as a contraction over named axes — accumulate over `(m,n)` contracting `k`,
versus accumulate over `(m)` contracting `n` — which is the linalg/einsum formulation, and
a design step rather than a refactor.

So the state, stated precisely:

| | count | shared by |
|---|---|---|
| access layer (masks, flat indexing) | 1 | every schedule |
| derivation (dispatch, fragment, masks-where) | 1 | every schedule |
| body parser | 1 | every schedule |
| emitter | **2** | fragment vs row-wise |

Three of four are one. The fourth is two because the memory schedules genuinely differ,
and the next honest step for it is named above rather than left as "more work".


## R11. The unified emitter, verified

All four kernels through one contraction-driven emitter, on hardware:

```
aligned matmul    786432/786432 bit-identical to the hand-written kernel
ragged matmul     750000/750000 bit-exact, tails included
softmax           rows sum to 1; 6 ULP against the oracle
layernorm         mean 0 variance 1; 3.2 ULP at scale against the oracle
```

Sizes: matmul 113 → 107 lines, ragged 145 → 139, softmax 477 → 163, layernorm
390 → 160. softmax's bound checks fell 64 → 16 because reads are now once per
accumulate slice rather than once per fragment cell — the old emitter re-read the
same row four times per block, which the unification removed without anyone
aiming at it.

**Cost: none, after two codegen fixes.** The unification first cost two quanta
(6q → 8q); comparing the generated inner loop against the hand-written one found a
cancelling `ci - cb` in every staged read and a loop-invariant address recomputed
per step. Both removed, the direct backend is back to **6q** — at parity with the
hand-written kernel, inside one quantum. The derivation generalises, and it turned
out to be free; see docs/002 §6b, including why the same class of fix was worth
1.00× on the other backend.

**So R10 was wrong twice over.** It said the split was honest and that unifying it
needed a contraction abstraction rather than a refactor. The abstraction was real;
the split was not.

## R12. The fourth family: attention, and what it moved

Written as a probe — the kernel first, before finding out what is admitted — so
the type checker enumerated the gap instead of argument doing it. Four things,
three vocabulary and one structural:

| | | |
|---|---|---|
| G1 | `mma` wants `[bk, bn]`; attention needs `Kᵀ`, and `k.tile(n, d)` is `[bn, bk]` | vocabulary — **closed** |
| G2 | `rowMax(s, mx)` where `s` is a computed `Frag`, not a loaded `Tile` | **closed** |
| G3 | `subRow` / `divRow` / `mma` over a computed fragment | **closed** (row ops; `mma` needs G4) |
| G4 | `at.d` does not exist | **structural** |

G1 is the surface working as designed: a transposed axis *is* a type error here,
so attention has to **say** the transpose rather than have it inferred.

**G4 is the one worth having found.** Attention contracts the head dimension in
`S = Q·Kᵀ` and leaves it free in `O = P·V`, with the key axis as the mirror image
— the same axis on both sides of the split, in one kernel. But
`planContraction(accumulate, contract, …)` was already parameterised per
contraction and assumed nothing global. What assumed a global split was the
**surface**: `spec.grid` and `spec.reduce` partitioned the axes once per kernel,
and `at.*` was built from grid alone.

So R11's prediction — derivation general, vocabulary finite — survived contact in
a specific form, and named its own next move.

### The partition is gone; the roles are read

`spec.grid` and `spec.reduce` became `spec.axes`, one list, and `Ctx` opens `at`
and `reduce` over every declared axis. The roles come from the body:

```
contracted   the axis of a `for (const n of reduce.n)` pass
parallel     an axis used as `at.<name>` in the store's `.tile(...)`
```

Declaring them was asking the author for what the compiler could already see —
the same shape of mistake as a host recomputing `manifest.dispatch`.

**"Parallel = the output binding's axes" would have been wrong**, and softmax is
the counterexample that caught it before any code was written: `y` is `[m, n]`
but `n` is reduced, and the body already says so, by writing `y.tile(at.m, n)`
rather than `y.tile(at.m, at.n)`.

Three things the parser had been discarding, each redundant only because of the
global partition: the loop axis, the tile coordinates, and the store coordinates.
The second is the notable one — **transposition was caught by the type checker
alone, and nothing downstream could see which axis went where.**

Verified behaviour-preserving: all four kernels emit **byte-identical WGSL**
before and after. The type surface came out simpler as well as more general —
one list instead of two — and two new admission errors exist that could not have
existed before, both with negatives:

```
unusedaxis   an axis in spec.axes the body never mentions
bothroles    an axis reduced and stored along within one contraction
```

### A correction, made immediately after

`bothroles` was first written up as "attention's shape, sayable now and not yet
schedulable". That was wrong, and conflated two different things:

| | |
|---|---|
| **reduced and stored along in ONE contraction** | `c[m,n] = sum_n …` sums over the axis that indexes the output. Not unimplemented — **undefined**, and an error however far the emitter gets |
| **one axis in two roles across TWO contractions** | attention: the head dimension is contracted in `S = Q·Kᵀ` and free in `O = P·V`. Legal, unimplemented |

`bothroles.kernel.ts` has one contraction, so it is the first. It cannot be the
second, and neither can anything else yet: a body reducing over two axes is
refused earlier by "this build supports exactly 1 reduction axis", so attention's
case never reaches that check at all.

The mistake is worth recording because of its direction. It dressed an
ill-defined kernel up as a milestone, which would have made the next step look
closer than it is — the check that "already handles attention's shape" handles
nothing of the kind. **Composing two contractions in one body is untouched work**,
and it is what G4 actually needs.

### G1 closed: `.tileT()`

A transposed read, said rather than inferred — the surface refusing
`k.tile(d, n)` is it working as designed, so the kernel has to name what it
wants and naming it wrong stays an error.

**The access layer needed nothing.** `emitLoad` already took both indices from
its caller instead of deriving them, so a transposed read is the same load with
the indices passed the other way. `logical(bd)` swaps the axes for everything
that reasons about the tile's shape; `emitLoad` is the one place that swaps back,
because it is the one place that indexes the buffer.

Verified twice, and the two are different kinds of evidence:

*Structurally.* The generated WGSL for `A·Bᵀ` differs from plain matmul by
exactly one line —

```
-  let off1 = t_gr * 768u + t_gc;     b  is [512, 768]  ->  b[gr][gc]
+  let off1 = t_gc * 512u + t_gr;     bT is [768, 512]  ->  bT[gc][gr]
```

— and `bT[gc][gr] == b[gr][gc]` *is* the statement that `bT` is `b` transposed.
Everything else is byte-identical, so those values feed identical arithmetic.
That argument covers every input rather than one, and it is a test rather than a
paragraph. Writing the test caught the entry point's continuation line differing
in **alignment** only, because parameters are padded to the opening paren and the
two kernel names are different lengths; whitespace is collapsed before comparing.

*Numerically.* The harness runs both on the same logical matrices with `B` stored
two ways:

```
A·Bᵀ vs A·B    bit-exact 786432 / 786432   IDENTICAL
```

The structural argument was strong and had never been executed. Given how often
this project has had a confident inference turned over by running it, that was
worth closing.

**Cost: nothing measurable, now at sixteen times the resolution.** This was first
recorded as "not measurably worse, which is weaker than free" — 6q against 6q
with a whole quantum of noise cannot resolve a 15% difference, and a transposed
read changes which lanes touch which addresses, so the question was fair.
Batched, `A·Bᵀ` is 6.63q against 6.69q with 0.31q of noise. Still not measurable,
and now that means something. See `docs/002` §2a.

### G2 and G3 closed: reducing a computed fragment

`softmax_n(A·B)`, one contraction and then a row reduction over the fragment it
produced. N is one block wide so a row fits a workgroup, which keeps this clear
of G4.

**Four layers moved, and only the first is what "vocabulary" described.**

| | |
|---|---|
| surface | `Frag` overloads for `rowMax`/`rowSum`/`subRow`/`divRow`/`mulRow`/`expTile` |
| parser | a fragment reduction as a derived value outside any loop; a fragment-valued intermediate; `FragExpr` grew the unary and row-scaling forms `Expr` already had |
| emitter | `emitReduceFrag` — a **cross-lane** fold |
| `rowExpr` | a row value is indexed by the row cell, not the whole 2-D cell |

The emitter half is the reclassification. `accumulate = (m, n)` assigns n to the
x lane, so one row of the fragment lives in all sixteen of them: fold the four
columns this invocation holds, write the partial to scratch at its own lane,
barrier, fold across lanes. That is a new capability rather than a new name — the
same shape as `emitCombine` and not the same code, since there the accumulator is
one-dimensional and here only one axis of a two-dimensional fragment is folded.

The overloads are deliberately **unguarded**. A fragment's out-of-range lanes hold
the zero `mma` guarantees, which is the identity a sum wants and the wrong one for
a max — but whether that matters depends on whether the reduced axis is ragged,
which is a property of the axis and lives in the spec rather than in the value. So
the check belongs to the pass that has the axis table. Same split as everywhere
else.

Verified: `fused.html`, 264-line kernel, six barriers, 12288 B of workgroup
storage against the 16384 B floor. Every row sums to 1 — an invariant needing no
reference, which a wrong cross-lane fold would break outright — and ULP against a
CPU `softmax(A·B)`.

Measured:

```
every row sums to 1   [0.999999754, 1.000000402]        PASS
softmax(A·B) oracle   45 ULP against a bound of 64      PASS
TypeGPU vs raw        65536 / 65536 bit-exact           IDENTICAL
```

**Both oracle failures on the way were the reference, not the kernel, and both
are the same mistake this project has now made three times.**

*189 ULP.* The reference did not contract the multiply-add, where the adapter
fuses `acc + a*b` into one rounding — which the plain matmul had already
established at 786432/786432 bit-exact. Two roundings leave the scores off by
~1.14e-5, and `exp` turns an absolute score error into a relative one, so ~190
ULP comes out the far side. Predicted 192, measured 189.

*45 ULP.* The bound was flat at 8, copied from `rowwise.html`'s softmax. Measured
in two parts rather than argued: the cross-lane fold order against a linear sum
costs **4 ULP** and is not the cause; and a GPU computes `exp(x)` as
`exp2(x · log2e)`, so rounding that product to f32 costs about **|x| ULP** of the
result — `|x| = 20` gives ~1, `47` gives ~22, `60` gives ~28. This kernel feeds
`exp` a 512-term dot product, so the argument reaches −55.6 where softmax over
data read from memory keeps it small. The bound is now
`ceil(max |s − rowmax|) + 8`, with the oracle returning that maximum because it
*is* the error scale.

Same class as the 912 spurious matmul failures in `docs/002` §2 both times: **a
criterion that is easy to state, standing in for one that is true.** The sharp
check here is the invariant — rows summing to 1 needs no reference at all, and a
wrong cross-lane fold breaks it outright.

### G4's two prerequisites, both closed

G4 needs three things and only the third is the hard one. The first two are done.

**One identity per BINDING, not per kernel.** The front end used to collapse every
named identity and refuse anything but a singleton. `examples/maxsum.kernel.ts` is
the case its own error message named:

```
y[m,n] = (p[m,n] - max_n p[m,:]) / sum_n q[m,:]
```

`p` feeds a MAX and `q` a SUM through the same ragged axis, so a masked lane of
`p` must read −∞ and one of `q` must read 0. Emitted: 12 masked reads of `p` carry
`-3.4028234663852886e38`, 12 of `q` carry `0.0`. Every existing kernel's WGSL is
byte-identical afterwards except its header comment, which now names which binding
got which identity. The MLIR backend carries one and refuses a two-identity body
by name rather than picking one.

**More than one reduction axis.** `examples/twoaxis.kernel.ts` reduces over `n`
(768) and `k` (512) independently, and the generated kernel walks both extents.
The emitter builds one plan per axis, and this is cheaper than it sounds because
**the lane geometry does not depend on the contracted axis at all** —
`planContraction` derives `perAxis`, `contractLanes` and `fragment` from
`accumulate` and the contiguous axis alone. So the plans differ only in what they
stage and how far they walk, which is checked rather than assumed.

The one thing beyond bookkeeping: a plan is built from the bindings its **own**
pass reads. With a single contraction every read binding took part in it; with
two, handing a pass an operand whose axes it does not walk asks for the global
coordinate of an axis that is neither accumulated nor contracted there. That was
the only crash the change produced, and it names the assumption that had been
invisible.

Again byte-identical: all six existing kernels unchanged.

### Four checks that were matmul-shaped

Getting attention as far as the redistribution meant correcting four things, each
of which had been right while every kernel had one contraction:

**`accumulate = grid ∪ enclosing`** was my rule for a nested pass, and it is
wrong. Attention's inner contraction runs over `d`, which is *also* a grid axis
because the output is indexed by it, so the rule asked for three lanes from a
workgroup that has two. The correct rule excludes the axis being contracted:
`(grid ∪ enclosing) \ {contract}`. It gives the same answer as before for
nested-softmax and the right one for attention.

**A body may hold both kinds of accumulator.** `parseBody` refused a mix because
`accKind` picked the schedule and there was one schedule per kernel. The emitter
has not branched on schedule since R11, and each pass now derives its own
geometry, so what was left of `accKind` is bookkeeping. Attention needs the mix:
a running max and a running sum, which are rows, alongside the output fragment.
Accumulators are sized by kind now — those were the same number while a body had
only one.

**"Reduced and also stored along" is about DEPTH.** The check refused any axis
that was both, which is right at the top level — `c[m,n] = sum_n …` sums over the
axis indexing the output, and that is undefined rather than unimplemented — and
wrong when the contraction is nested. Attention's `d` is contracted inside the
score loop and free at the store, which is two contractions, not one.

**Tile coherence was positional.** It checked grid[0] against `bm`, grid[1]
against `bn` and the reduction against `bk` — matmul's shape. Attention's second
parallel axis is the head dimension, blocked at `bk` because that is also what the
score contraction walks. Now every axis must be blocked at one of the tile's three
sizes, which still catches the typo it was for without assuming a position.

### The redistribution, measured before implementing it

`mma` now takes a computed fragment as its first operand, and the parser reads it,
so `spike/attention/attention.kernel.ts` type-checks and parses end to end. The
emitter refuses one thing, by name:

```
contracting a fragment along an axis it holds on a lane
```

`P` comes out of `S = Q·Kᵀ` laid out with `n` on the x lane — four of the
sixty-four `n` values per invocation — and `O = P·V` sums along `n`. Read in
place, each invocation would sum a quarter of its row.

The mechanism is not the hard part: **stage the fragment through workgroup memory,
exactly as an operand read from a buffer is staged, with registers as the source.**
The budget is:

```
                     q       k       v       P (staged)     total
bm=64 bn=64 bd=16    4096  + 4096  + 4096  + 16384      =  28672 B   OVER
bm=64 bn=32 bd=16    4096  + 2048  + 2048  +  8192      =  16384 B   exactly at it
bm=32 bn=32 bd=16    2048  + 2048  + 2048  +  4096      =  10240 B   ok
```

Against the 16384 B floor every WebGPU implementation guarantees. **Attention is
not expressible at this project's usual 64×64×16 tile at all** — not because of
anything in the emitter, but because the staged fragment is `bm × bn` and that is
the whole budget on its own. It needs 32×32×16.

Knowing that before implementing is the point of computing it first: the
alternative was to build the staging and then discover the kernel cannot be
dispatched.

The emitter now also **refuses a kernel over the floor** rather than reporting the
number and emitting it. The harness had always checked at run time; a kernel that
cannot be dispatched anywhere should not compile.

### G4 closed: attention

```
attention_f32   256 lines, 12 barriers, 12288 B of the 16384 B floor
dispatch [4, 32, 1]   workgroup 16x16   fragment 2x2

V = 1              every output 1   [0.999998391, 1.000001550]   PASS
TypeGPU vs raw     65536 / 65536 bit-exact                       IDENTICAL
error bound        0 violations / 65536, worst ratio 0.0289      PASS
```

`P` comes out of `S = Q·Kᵀ` with `n` on the x lane and `O = P·V` sums along `n`,
so the fragment goes through workgroup memory — exactly as an operand read from a
buffer is staged, with registers as the source. After that the block is an
ordinary matmul.

The tile is 32×32×16 because a `bm × bn` fragment at 64×64 is 16384 B on its own.
That was computed before any of it was built; the alternative order was to
implement the staging and then discover the kernel undispatchable.

**The check that found the truth needed no reference at all.** With every `V`
entry 1, every output must be exactly 1, because the softmax weights sum to 1 and
a convex combination of ones is one. It tests the normalisation, the
redistribution and the `P·V` contraction together and does not care what the
scores are — so it separates "the weights are wrong" from "V is read wrong". It
passed, which said the kernel was right while the ULP check was still saying FAIL.

**And the ULP check was the fourth time this project picked a criterion that is
easy to state over one that is true.** It reported 120658 ULP against a bound of
39, on a `maxAbsDiff` of **1.01e-6**. `O` is a convex combination of zero-mean
data, so a great many outputs land near zero, where relative error is meaningless
and one part in a million reads as a hundred thousand ULP. Precisely the 912
spurious matmul failures of `docs/002` §2, with a different operator.

What f32 actually permits has two terms, and the page now checks it:

```
|gpu - cpu|  <=  gamma_N · SUM|p·v|  +  |s - rowmax|·u · |O|
                 ─────────────────      ──────────────────────
                 N products accumulated  exp's own error, which grows
                 (the matmul's bound)    with its argument, carried
                                         through a weighted average
```

Worst ratio 0.0289 — the kernel uses 3% of what the arithmetic allows.

### One diagnosis on the way was wrong

`collectTransposed` did skip nested passes, and fixing it was right: the logical
and memory axis orders are now genuinely separate. But it was **not** the bug.
The old code staged `k` as `[n][d]` and read it as `[n][d]`; the new code stages
`[d][n]` and reads `[d][n]`. Both index the same element. The fix was a numerical
no-op, which is why three runs produced byte-identical output and why "the fix
did nothing" was the correct reading of it.

Two of those three runs were also lost to caching — the page answered from cache
after a change and reported the identical wrong number, which reads as a failed
fix rather than a stale file. Artifact fetches now pass `{ cache: "no-store" }`,
and the page prints a fingerprint of the shader it actually ran, comparable
against `npm run fingerprint`. A number the reader can check ends that class of
round trip.

### What is left

The kernel recomputes the scores for every block of `d`, which is wasteful and
correct: each workgroup owns one `(m, d)` block of the output and needs the full
softmax over `n` to produce it. Removing the recomputation is flash attention's
online rescaling, and it is a scheduling question rather than a missing
capability.

### The original G4 statement, for the record

Composing two contractions where the second **consumes the first's fragment**.
`plan.lanes` decides which invocation owns which output element, and a handoff
needs that fragment redistributed when the two layouts disagree — the
flash-attention register-layout problem. Nothing in G1–G3 or the two
prerequisites touched it; the emitter now refuses a mismatch by name rather than
emitting for the wrong layout.

Attention also needs a **nested** reduction — `for n { for d { … } … }` — which
the parser does not admit inside a pass. That nesting *is* the handoff, so it is
part of G4 rather than another prerequisite.
