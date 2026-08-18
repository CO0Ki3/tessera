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
Written up for reporting in `spike/upstream/naga-f32-literal-range.md`.

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

**Cost: two quanta.** The direct backend went 6q → 8q, still 87.5% of the
hand-written kernel and still inside the 80% invariant. One emitter that derives
its schedule is slightly slower than two shaped by hand. That is the trade, and it
is worth stating plainly since the whole argument of this document is that the
derivation generalises — it generalises, and it is not free.

**So R10 was wrong twice over.** It said the split was honest and that unifying it
needed a contraction abstraction rather than a refactor. The abstraction was real;
the split was not.
