# naga and Tint disagree on the upper bound of a float literal

- **Repo**: [gfx-rs/wgpu](https://github.com/gfx-rs/wgpu) (naga)
- **Affects**: naga-cli 30.0.0 (`cargo install naga-cli`)
- **Duplicate check**: nearest is #4568 `[glsl-out] Incorrect rounding for max-float
  constant expression`, which is a backend rounding issue, not front-end validation.
  No issue found on the accepted-range question.
- **Status**: premise checked and holds; **one confirmation left** — see below
- **Not verified**: which behaviour WGSL §15.7.6 mandates — see *Open question*.

## Premise check (do this before the details, not after)

The companion TypeGPU report was retracted because every verified detail sat on a
premise nobody had tested: *does this project intend to support what I am doing?*
Same question here, answered before filing:

| | |
|---|---|
| Does naga claim to validate WGSL? | Yes — WGSL support is listed as *"Fully validated"* |
| Is it a dev tool, or a real implementation? | *"It serves as the core of the WebGPU integration in **Firefox**, Servo, and Deno."* So naga's front-end validation **is** a browser's |
| Do they aim at the spec? | Yes, and they run the CTS — while noting the implementation *"will likely differ from what is specified, as the implementation catches up"*, which is what makes a specific divergence worth naming rather than noise |

So unlike the TypeGPU case the premise holds, and it holds harder than the
original framing: this is not "a CLI disagrees with Chrome", it is potentially
**Firefox and Chrome disagreeing about whether a shader compiles**.

### The one thing still to confirm

Everything below was measured against `naga-cli 30.0.0`. Firefox ships its own
version of wgpu/naga, and that is what makes the interop claim real, so **open
`spike/wgsl-baseline/f32-literal-check.html` in Firefox and in Chrome** before
filing and paste both results into the issue. If Firefox accepts `3.4028235e38`
and Chrome rejects it, the issue is about browser interop. If Firefox already
rejects it, the divergence is confined to the CLI's release and the issue is
smaller — still real, but it should say so.

Everything between the markers is the issue body. The section after it is ours.

<!-- ─────────────────── BEGIN ISSUE BODY ─────────────────── -->

### Summary

naga accepts a float literal that Chrome (Tint) rejects. The two implementations
draw the line in different places:

| | accepts a literal when |
|---|---|
| **naga** | it rounds to a **finite** value — i.e. it is below the round-to-nearest midpoint between `f32::MAX` and `2^128` |
| **Tint** | it is **≤ `f32::MAX`** |

Everything in between is accepted by one and rejected by the other. For f32 that
band is `(3.4028234663852886e38, 3.4028235677973366e38)`, and it contains
`3.4028235e38` — the decimal people actually write when they mean "the f32
maximum".

### Repro

```wgsl
@group(0) @binding(0) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(1)
fn main() {
  out[0] = -3.4028235e38;   // f32::MAX is 3.4028234663852886e38
}
```

```
$ naga f32lit.wgsl
Validation successful
```

Chrome rejects the same source at `createShaderModule`:

```
value -340282349999999991754788743781432688640.0 cannot be represented as 'f32'
```

### The boundary, bisected

naga's cut-off is exactly the round-to-nearest midpoint, which is the signature
of converting and then testing for finiteness rather than range-checking first:

```
3.402823567e38   ACCEPTED    (midpoint is 3.4028235677973366e38)
3.40282366e38    rejected    error: the concrete type `f32` cannot represent the abstract value
```

The check is present and works for values that round to infinity — `1e39` is
rejected — so this is where the boundary sits, not a missing check.

### It is not an f32-specific slip

f16 behaves the same way, so naga is internally consistent. `f16::MAX` is 65504
and the midpoint to `2^16` is 65520:

```
65503h   ACCEPTED   (in range, not exactly representable — correctly fine)
65504h   ACCEPTED   (exact max)
65505h   ACCEPTED   (above max, rounds down to max)
65519h   ACCEPTED   (still below the midpoint)
65520h   rejected   error: numeric literal not representable by target type
```

### Every position, including the suffixed one

The out-of-range f32 value is accepted in all of these:

```
o[0] = 3.4028235e38;              ACCEPTED
let v = 3.4028235e38;             ACCEPTED
const C : f32 = 3.4028235e38;     ACCEPTED
override C : f32 = 3.4028235e38;  ACCEPTED
o[0] = 3.4028235e38f;             ACCEPTED   <-- explicit suffix
vec3f(3.4028235e38).x             ACCEPTED
```

The suffixed one is worth calling out because §3.5.2 names it directly:

> A shader-creation error results if: A decimal floating point literal with an
> `f` or `h` suffix overflows the target type.

Whether `3.4028235e38` "overflows" f32 is exactly the question below.

### Why this has gone unnoticed

The CTS does not cover the band. `src/webgpu/shader/validation/parse/literal.spec.ts`
tests overflow with `1.0e+999999999999f` and `0x1.0p+999999999999f` — values that
round to infinity, which both implementations already reject. Nothing exercises
the region between `f32::MAX` and the midpoint, so the disagreement is invisible
to conformance.

### Open question — which one is right?

I could not get a definitive reading of §15.7.6 Floating Point Conversion, so
this is reported as a divergence rather than as a naga bug:

- If **rounding to nearest** is the specified behaviour for an out-of-range
  AbstractFloat, naga is correct and this belongs in Dawn.
- If the value must be **within range before rounding**, the fix is to compare
  against `f32::MAX` / `f16::MAX` before converting, which would also make the
  §3.5.2 suffixed-literal rule hold.

Either way the current state is that a shader validated by naga fails in a
browser, and I am happy to send the CTS cases for the boundary once the intended
answer is settled — and the naga fix too, if it is this side.

### Why it is worth fixing rather than documenting

Two reasons, and the second is the bigger one.

**As a toolchain problem.** This was found by a compiler that emits WGSL. The
generated shader passed naga — 57 occurrences of the literal, "Validation
successful" — and failed only when it reached Chrome. A validator that accepts
what the target rejects is worse than no validator at that step, because it turns
a build-time error into a runtime one.

**As an interop problem.** naga is the core of the WebGPU integration in Firefox,
so this is not a tool disagreeing with a browser but two browsers potentially
disagreeing about whether a shader compiles at all — for a literal that a
generator has every reason to emit, since it is what "the f32 maximum" looks like
when written as a decimal.

`f32-literal-check.html` in this report's repo runs the five boundary cases
through `createShaderModule` and prints what the current browser did, so both
results can be pasted in rather than described.

<!-- ──────────────────── END ISSUE BODY ───────────────────── -->

## Not part of the issue — what tessera does about it

`src/ir.ts` emits the exact f32 maximum, `-3.4028234663852886e38`, as the `negInf`
pad identity, and `assertF32Literals` range-checks every literal the emitter
produces.

The guard checks **range, not exact representability**. An earlier version
required round-tripping through `Math.fround`, which rejected `0.00001` — a
perfectly good f32 literal that simply is not exact in binary. Rejecting inexact
literals would have been the same mistake as the 912 spurious matmul failures:
picking a criterion that is easy to state instead of the one that is true.

## How this was investigated

The first version of this file listed three things as not done, and all three
changed the report:

- **f16 and other positions.** Checking them was expected to show f32 was a
  one-off. It showed the opposite — naga is consistent, which turned "naga has an
  f32 bug" into "naga and Tint use different rules". The first framing would have
  been wrong.
- **A boundary bisect** turned "naga accepts something too large" into "naga's
  boundary is the round-to-nearest midpoint", which names the likely line of code.
- **The duplicate search** found #4568, close enough to link and different enough
  to file separately.

The spec question stayed open, so the issue asks it instead of assuming an
answer. An issue that asserts the wrong side is worse than one that reports a
divergence precisely.
