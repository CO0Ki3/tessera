# Comment to post on gfx-rs/wgpu#9907

Paste everything between the markers. Nothing else needs to be filed.

<!-- ─────────────────── BEGIN COMMENT ─────────────────── -->

There is a second problem in the same literal path, and since both land on the
one `write!` for `Literal::F32` it may be worth fixing together.

For `f32::MAX`, `Display` emits `340282350000000000000000000000000000000f`. That
round-trips *as an f32*, but WGSL parses a literal as `AbstractFloat` first,
where it is strictly greater than `f32::MAX`. Chrome rejects it:

```wgsl
// in.wgsl — the input spells f32::MAX exactly
o[0] = 0x1.fffffep+127;
```
```console
$ naga in.wgsl out.wgsl && grep 'o\[0\]' out.wgsl
    o[0] = 340282350000000000000000000000000000000f;
```

Through `createShaderModule` on both browsers:

| literal | Firefox 153 | Chrome 151 |
|---|---|---|
| `0x1.fffffep+127` — the input | accepted | accepted |
| `340282350000000000000000000000000000000f` — what `wgsl-out` produced | accepted | **rejected** |
| `3.4028235e38` — the same value, exponent form | accepted | **rejected** |

**This is a range problem rather than a length one**, so exponent notation alone
would not cover it — `3.4028235e38f` is short and still rejected, as the third
row shows.

The `i32` arm just below already handles exactly this shape of bug:

```rust
crate::Literal::I32(value) => {
    // `-2147483648i` is not valid WGSL. The most negative `i32`
    // value can only be expressed in WGSL using AbstractInt and
    // a unary negation operator.
    if value == i32::MIN {
```

The same treatment for `f32`:

```diff
-                crate::Literal::F32(value) => write!(self.out, "{value}f")?,
+                crate::Literal::F32(value) => {
+                    // `340282350000000000000000000000000000000f` is not valid
+                    // WGSL. `Display` for `f32` emits the shortest decimal that
+                    // round-trips as an `f32`, and for `f32::MAX` that decimal
+                    // denotes a larger value than `f32::MAX` itself. WGSL reads
+                    // a literal as `AbstractFloat` first, where it is out of
+                    // range, so the extremes must be spelled exactly.
+                    if value == f32::MAX || value == f32::MIN {
+                        write!(self.out, "{:e}f", value as f64)?;
+                    } else {
+                        write!(self.out, "{value}f")?;
+                    }
+                }
```

which gives `3.4028234663852886e38f`, accepted by both browsers above.

Scope: scanning the two million `f32` values below the maximum found exactly one
whose `Display` output exceeds `f32::MAX`, and it is `f32::MAX`. So this touches
two literals and no snapshot that does not already contain one, and `0.1f` and
friends are unchanged. `f16` needs nothing — `f16::MAX` is 65504, exactly
representable as a decimal.

It is orthogonal to the formatting change this issue asks for: after the above,
`3e38` still prints as its 39-digit expansion.

Happy to open a PR for this if you would like it done this way — or for both
changes together, if you would rather they land at once.

<!-- ──────────────────── END COMMENT ───────────────────── -->

## Note

Posting here instead of filing separately means the range defect rides in a
thread whose title and requested fix are about notation. If it gets resolved as
"emit exponent form", this is still broken — `3.4028235e38f` is rejected. The
comment says so explicitly, which is the mitigation. If a maintainer replies
treating them as one thing, that is the moment to file the standalone issue,
which is written and ready in `naga-f32-literal-range.md`.
