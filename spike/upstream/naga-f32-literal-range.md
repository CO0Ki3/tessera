# naga's `wgsl-out` emits a float literal that Chrome rejects

- **Repo**: [gfx-rs/wgpu](https://github.com/gfx-rs/wgpu) — `gfx-rs/naga` was
  **archived 2025-01-29** and is read-only; development moved into `wgpu/naga/`
- **Affects**: naga 30.0.0 — which is also wgpu 30.0.0 and naga-cli 30.0.0, one
  release train, published 2026-07-02. This is current, not an old CLI.
- **Related**: [#4568](https://github.com/gfx-rs/wgpu/issues/4568) — same root
  cause (float serialization near `f32::MAX`), different backend and symptom
- **Label**: `naga (Shader Translator)`
- **Status**: ready to file

## Premise check (before the details, not after)

The companion TypeGPU report was retracted because every verified detail rested
on a premise nobody had tested: *does this project intend to support what I am
doing?* Asking it here gives the opposite answer.

| | |
|---|---|
| Does naga claim to validate WGSL? | Yes — WGSL support is listed as *"Fully validated"* |
| Dev tool, or a real implementation? | *"It serves as the core of the WebGPU integration in **Firefox**, Servo, and Deno."* naga's validation **is** a browser's |
| Do they aim at the spec? | Yes, and they run the CTS, while noting the implementation *"will likely differ from what is specified, as the implementation catches up"* — which makes naming a specific divergence useful rather than noisy |

And the headline finding below does not depend on any of that, or on reading the
spec: naga emits WGSL that another implementation refuses to compile, from input
that was exactly representable.

<!-- ─────────────────── BEGIN ISSUE BODY ─────────────────── -->

### Summary

Given a shader containing `f32::MAX` written exactly, naga's `wgsl-out` emits a
literal whose value is **strictly greater than `f32::MAX`**, and Chrome (Tint)
rejects the result at `createShaderModule`.

```wgsl
// in.wgsl
@group(0) @binding(0) var<storage, read_write> o : array<f32>;
@compute @workgroup_size(1)
fn main() {
  o[0] = 0x1.fffffep+127;      // exactly f32::MAX, unambiguously spelled
}
```

```
$ naga in.wgsl out.wgsl
$ grep 'o\[0\]' out.wgsl
    o[0] = 340282350000000000000000000000000000000f;
```

```
340282350000000000000000000000000000000  =  3.4028235e38
f32::MAX                                 =  3.4028234663852886e38
```

The emitted decimal round-trips *as an f32*, so it is a correct shortest
round-trip spelling. But WGSL parses an unsuffixed or `f`-suffixed decimal as an
AbstractFloat first, and as a real number `3.4028235e38 > f32::MAX`. Chrome
rejects it:

```
value 340282349999999991754788743781432688640.0 cannot be represented as 'f32'
```

The same literal comes out of the SPIR-V round trip (`naga in.wgsl mid.spv` then
`naga mid.spv rt.wgsl`), so it is the shared float formatting, not one backend.

This is the WGSL sibling of #4568, which reports `glsl-out` emitting
`3.4028235e38` where it used to emit `3.4028234663852886e38`.

**An exact spelling is available and shorter to reason about**: `0x1.fffffep+127`
is already what the input used, and `3.4028234663852886e38` also works.

### The front-end accepts what it emits, which is why this is invisible here

naga's validator accepts the literal it just produced, so a round trip looks
clean and only fails at a browser. The accepted range extends to the
round-to-nearest midpoint — the signature of converting first and then testing
for finiteness, rather than range-checking before conversion:

```
3.402823567e38   ACCEPTED    (midpoint is 3.4028235677973366e38)
3.40282366e38    rejected    error: the concrete type `f32` cannot represent the abstract value
1e39             rejected
```

So the check exists; this is where the boundary sits. It holds in every position,
including the `f`-suffixed one that §3.5.2 names directly:

```
o[0] = 3.4028235e38;              ACCEPTED
let v = 3.4028235e38;             ACCEPTED
const C : f32 = 3.4028235e38;     ACCEPTED
override C : f32 = 3.4028235e38;  ACCEPTED
o[0] = 3.4028235e38f;             ACCEPTED   <-- and this is what wgsl-out writes
vec3f(3.4028235e38).x             ACCEPTED
```

> §3.5.2 — A shader-creation error results if: A decimal floating point literal
> with an `f` or `h` suffix overflows the target type.

### f16 is fine, which locates the problem

`f16::MAX` is 65504, which is exactly representable as a decimal, so `wgsl-out`
writes `65504h` and nothing goes wrong. The f32 maximum is the case where the
shortest round-trip decimal happens to exceed the value it denotes.

naga's f16 *front-end* uses the same permissive rule (`65519h` accepted, `65520h`
— the midpoint — rejected), so the two are consistent; f16 simply never has to
serialize a value that lands in the band.

### Why the CTS has not caught it

`src/webgpu/shader/validation/parse/literal.spec.ts` tests overflow with
`1.0e+999999999999f` and `0x1.0p+999999999999f` — values that round to infinity,
which both implementations already reject. Nothing exercises the region between
`f32::MAX` and the midpoint.

### Two things here, one of them spec-independent

1. **`wgsl-out` should not emit a literal outside the target type's range.**
   This holds whatever §15.7.6 says: the input value was exactly representable,
   an exact spelling exists, and the output is refused by another implementation.
2. **Whether the front-end should accept such a literal** depends on §15.7.6
   Floating Point Conversion, which I could not get a definitive reading of. If
   rounding an out-of-range AbstractFloat is specified, naga is right and Tint is
   wrong; if the value must be in range before rounding, the fix is to compare
   against `f32::MAX` before converting. Either way (1) stands, and (2) is what
   keeps (1) from being caught in-tree.

Happy to send the CTS cases for the band, and the naga fix, once you say which
answer (2) should have.

### Why it is worth fixing rather than documenting

This surfaced from a compiler that emits WGSL. Its shader passed naga — 57
occurrences of the literal, "Validation successful" — and failed only in Chrome,
turning a build-time error into a runtime one. Since naga is the core of the
WebGPU integration in Firefox, the same literal also decides whether two browsers
agree that a shader compiles.

<!-- ──────────────────── END ISSUE BODY ───────────────────── -->

## Before filing

Open `spike/wgsl-baseline/f32-literal-check.html` in **Chrome and Firefox** and
paste both results in. Row 2 is naga's actual `wgsl-out` output, so it answers
the interop question directly. Everything above was measured against naga-cli
30.0.0, not against a shipping Firefox.

## Not part of the issue — what tessera does about it

`src/ir.ts` emits the exact f32 maximum, `-3.4028234663852886e38`, as the
`negInf` pad identity, and `assertF32Literals` range-checks every literal the
emitter produces.

The guard checks **range, not exact representability**. An earlier version
required round-tripping through `Math.fround`, which rejected `0.00001` — a
perfectly good f32 literal that simply is not exact in binary. Rejecting inexact
literals would have been the same mistake as the 912 spurious matmul failures:
picking a criterion that is easy to state instead of the one that is true.

## How the report changed under checking

Three passes, each of which changed the claim:

1. **"naga accepts a literal larger than f32::MAX."** Checking f16 and the other
   literal positions was expected to show f32 was a one-off. It showed naga is
   consistent — so the finding was a *divergence* between naga and Tint, not a
   naga slip. First framing wrong.
2. **Bisecting** moved "accepts something too large" to "the boundary is the
   round-to-nearest midpoint", which names the likely implementation.
3. **Reading the nearest existing issue** (#4568, found while checking for
   duplicates) turned the whole thing around: it reports naga *emitting*
   `3.4028235e38` for `f32::MAX`. Testing whether `wgsl-out` does the same found
   the real bug — naga produces WGSL that Chrome will not compile — which needs
   no spec interpretation at all, and demoted the original claim to a secondary
   point.

The duplicate search was worth more than a duplicate check.
