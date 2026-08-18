# naga accepts a float literal larger than f32::MAX

**Project:** wgpu / naga
**Found:** 2026-08-18, naga 30.0.0 (`cargo install naga-cli`)
**Impact:** WGSL that naga validates fails to compile in Chrome.

## Repro

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

Chrome (Tint) rejects the same shader at `createShaderModule`:

```
value -340282349999999991754788743781432688640.0 cannot be represented as 'f32'
```

`3.4028235e38` is a decimal that looks like the f32 maximum and is often written
as one, but it is strictly greater than it and does not round-trip:

```js
Math.fround(3.4028235e38) !== 3.4028235e38          // true
Math.fround(3.4028234663852886e38) === 3.4028234663852886e38   // the real max
```

## Why it matters

This was found by a compiler that emits WGSL. The generated shader passed naga —
57 occurrences of the literal, "Validation successful" — and only failed once it
reached a browser. A validator that accepts what the target rejects is worse than
no validator at that step, because it converts a build-time error into a runtime
one.

## Not yet done

- Not checked against a current naga `main`, only the released 30.0.0.
- Not checked whether the same applies to f16 literals, or to `abstract-float`
  conversion in other positions (a `const` initialiser, an override default).
- No existing issue search performed. Do that before filing.
