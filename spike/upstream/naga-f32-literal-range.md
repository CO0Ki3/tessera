# naga's `wgsl-out` emits a float literal that Chrome rejects

- **Repo**: [gfx-rs/wgpu](https://github.com/gfx-rs/wgpu) — `gfx-rs/naga` was
  **archived 2025-01-29** and is read-only; development moved into `wgpu/naga/`
- **Affects**: naga 30.0.0 — which is also wgpu 30.0.0 and naga-cli 30.0.0, one
  release train, published 2026-07-02. This is current, not an old CLI.
- **Related**: [#4568](https://github.com/gfx-rs/wgpu/issues/4568) — same root
  cause (float serialization near `f32::MAX`), different backend and symptom
- **Label**: `naga (Shader Translator)`
- **Confirmed end to end on shipping browsers**: Firefox 153.0 accepts the exact
  string naga emits; Chrome 151.0 rejects it (macOS). Nothing here is inferred.
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

**Description**

`wgsl-out` emits a float literal whose value is greater than `f32::MAX`, and
Chrome (Tint) rejects the result. Given a source containing `f32::MAX` written
exactly as a hex float, naga writes it back as a decimal that round-trips *as an
f32* but that, read as WGSL's `AbstractFloat`, is out of range.

The output is accepted by Firefox (naga) and rejected by Chrome (Tint), so the
same generated shader compiles in one browser and not the other.

naga's front-end accepts the literal it just produced, which is why a round trip
looks clean in-tree and only fails at a browser. Its accepted range extends to
the round-to-nearest midpoint rather than to `f32::MAX` — the signature of
converting first and testing for finiteness, rather than range-checking before
conversion.

This is the WGSL sibling of #4568, which reports `glsl-out` emitting
`3.4028235e38` where it used to emit `3.4028234663852886e38`. Same root cause,
different backend, and this one produces a shader a browser refuses.

**Repro steps**

```wgsl
// in.wgsl
@group(0) @binding(0) var<storage, read_write> o : array<f32>;
@compute @workgroup_size(1)
fn main() {
  o[0] = 0x1.fffffep+127;      // exactly f32::MAX, unambiguously spelled
}
```

```console
$ naga in.wgsl out.wgsl
$ grep 'o\[0\]' out.wgsl
    o[0] = 340282350000000000000000000000000000000f;
```

```
emitted    340282350000000000000000000000000000000  =  3.4028235e38
f32::MAX                                               3.4028234663852886e38
```

The SPIR-V round trip produces the same literal, so this is the shared float
formatting rather than one backend:

```console
$ naga in.wgsl mid.spv && naga mid.spv rt.wgsl
$ grep '= 3' rt.wgsl
    global.member[0u] = 340282350000000000000000000000000000000f;
```

Paste `out.wgsl` into `device.createShaderModule()` in Chrome to see it refused.
A self-contained page that runs the boundary cases through `createShaderModule`
and prints each verdict is attached below.

**Expected vs observed behavior**

*Expected*: a WGSL value that is exactly representable in `f32` round-trips
through `wgsl-out` as a literal every WGSL implementation accepts. An exact
spelling is available and is what the input used — `0x1.fffffep+127`, or
`3.4028234663852886e38`.

*Observed*: the emitted literal is greater than `f32::MAX` as a real number, and
is refused by Tint.

Running the boundary through `createShaderModule` on both browsers:

| literal | Firefox 153 (naga) | Chrome 151 (Tint) |
|---|---|---|
| `3.4028234663852886e38` — exactly `f32::MAX` | accepted | accepted |
| `340282350000000000000000000000000000000f` — **what `wgsl-out` writes** | **accepted** | **rejected** |
| `3.4028235e38` — the same value | **accepted** | **rejected** |
| `3.402823567e38` | **accepted** | **rejected** |
| `3.4028235677973366e38` — the midpoint | rejected | rejected |
| `1e39` | rejected | rejected |

Tint:

```
value 340282349999999991754788743781432688640.0 cannot be represented as 'f32'
```

Firefox, for the values it does reject — note *accurately* rather than a range
complaint, which is consistent with convert-then-test-finiteness:

```
the concrete type `f32` cannot represent the abstract value `3.4028235677973366e38` accurately
```

There are two separable things here, and the first does not depend on how the
spec reads:

1. **`wgsl-out` should not emit a literal outside the target type's range.** The
   input value was exactly representable, an exact spelling exists, and the
   output is refused by another implementation.
2. **Whether the front-end should accept such a literal** depends on §15.7.6
   Floating Point Conversion, and I could not get a definitive reading of it. If
   rounding an out-of-range `AbstractFloat` is specified, naga is right here and
   Tint is wrong; if the value must be in range before rounding, the fix is to
   compare against `f32::MAX` before converting. Either way (1) stands, and (2)
   is what keeps (1) from being caught in-tree.

Happy to send the CTS cases for the band, and the naga fix, once you say which
answer (2) should have.

**Extra materials**

*The front-end boundary, bisected.* It sits exactly at the round-to-nearest
midpoint between `f32::MAX` and `2^128`:

```
3.402823567e38   accepted     (midpoint is 3.4028235677973366e38)
3.40282366e38    rejected     error: the concrete type `f32` cannot represent the abstract value
1e39             rejected
```

So the check exists and works for values that round to infinity; this is where
the boundary is.

*Every literal position accepts it*, including the `f`-suffixed one that §3.5.2
names directly — and the suffixed form is what `wgsl-out` writes:

```
o[0] = 3.4028235e38;              accepted
let v = 3.4028235e38;             accepted
const C : f32 = 3.4028235e38;     accepted
override C : f32 = 3.4028235e38;  accepted
o[0] = 3.4028235e38f;             accepted
vec3f(3.4028235e38).x             accepted
```

> §3.5.2 — A shader-creation error results if: A decimal floating point literal
> with an `f` or `h` suffix overflows the target type.

*f16 is unaffected, which locates the defect.* `f16::MAX` is 65504, exactly
representable as a decimal, so `wgsl-out` writes `65504h` and nothing goes wrong.
`f32::MAX` is the case where the shortest round-trip decimal exceeds the value it
denotes. naga's f16 front-end uses the same permissive rule (`65519h` accepted,
`65520h` — the midpoint — rejected), so the two are consistent; f16 simply never
has to serialize a value in the band.

*Why the CTS has not caught it.*
`src/webgpu/shader/validation/parse/literal.spec.ts` tests overflow with
`1.0e+999999999999f` and `0x1.0p+999999999999f` — values that round to infinity,
which both implementations already reject. Nothing exercises the region between
`f32::MAX` and the midpoint.

*Repro page.* A single self-contained HTML file that runs the cases through
`createShaderModule` and prints each browser's verdict, for pasting results
without describing them. It covers the **browser half only** — it cannot run
naga, so the `naga in.wgsl out.wgsl` step above is what supplies row 2's string:
<!-- attach f32-literal.html, or inline it here -->

Its first two rows are the input/output pair: `0x1.fffffep+127` as given to naga,
and `340282350000000000000000000000000000000f` as naga wrote it. Both browsers
take the input; only Firefox takes the output.

**Platform**

```
naga / naga-cli   30.0.0   (= wgpu 30.0.0, same release train, 2026-07-02)
rustc             1.97.1
OS                macOS 14.1 (23B2073), arm64, Apple M3 Pro
Firefox           153.0     accepts the emitted literal
Chrome            151.0     rejects it
```

The GPU and backend are not involved — `naga in.wgsl out.wgsl` never touches a
device, and the browser half fails at `createShaderModule`, before any pipeline
is created.

Found by a compiler that emits WGSL: the generated shader passed naga — 57
occurrences of the literal, "Validation successful" — and failed only once it
reached Chrome, turning a build-time error into a runtime one.

<!-- ──────────────────── END ISSUE BODY ───────────────────── -->

## Verification status — complete

Both browsers run (Firefox 153.0, Chrome 151.0, macOS) via
`spike/wgsl-baseline/f32-literal.html`. Firefox's boundary matches naga-cli
30.0.0 exactly, midpoint rejection included, which confirms the CLI behaviour
measured here is what Firefox ships.

Two small observations from the runs, both worth keeping in the issue:

- Tint's diagnostic for the suffixed form is shorter — `value cannot be
  represented as 'f32'`, with no expanded decimal, where the unsuffixed forms all
  print one. Different validation path, consistent with §3.5.2 treating suffixed
  literals as their own rule.
- Firefox says the value cannot be represented *"accurately"*. Accuracy rather
  than range, which is what convert-then-test-finiteness would say.

### Possible follow-up at gpuweb

Two implementations that both aim at the spec disagree, so §15.7.6 is either
ambiguous or one of them is wrong. A clarification issue on `gpuweb/gpuweb` would
settle point (2) below, and the CTS gap is theirs to fill. Not drafted — the
`wgsl-out` defect is naga-side regardless of how the spec question lands, so that
one goes first.

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
