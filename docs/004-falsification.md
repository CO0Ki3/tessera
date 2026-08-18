# 004 — The falsification test: is this a compiler or a template?

Status: **criteria fixed, experiment not yet run.**

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
