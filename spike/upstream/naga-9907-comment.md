# Comment to post on gfx-rs/wgpu#9907

Paste everything between the markers. Nothing else needs to be filed.

<!-- ─────────────────── BEGIN COMMENT ─────────────────── -->

`wgsl-out` has a second problem in this same `write!`, which may be worth fixing
alongside. For `f32::MAX` it emits `340282350000000000000000000000000000000f`:

```console
$ cat in.wgsl
@group(0) @binding(0) var<storage, read_write> o : array<f32>;
@compute @workgroup_size(1) fn main() { o[0] = 0x1.fffffep+127; }

$ naga in.wgsl out.wgsl && grep 'o\[0\]' out.wgsl
    o[0] = 340282350000000000000000000000000000000f;
```

That round-trips *as an f32*, but WGSL reads a literal as `AbstractFloat` first,
where it is greater than `f32::MAX`. Chrome rejects it, Firefox accepts it. It is
a **range** problem rather than a length one, so exponent form alone would not
cover it — `3.4028235e38f` is short and still rejected by Chrome.

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

which gives `3.4028234663852886e38f`, accepted by both browsers. `f32::MAX` is
the only `f32` whose `Display` output exceeds it, so this touches two literals
and leaves `0.1f` and the snapshots alone. `f16` needs nothing — 65504 is exactly
representable.

Happy to open a PR for this piece if you would like it done this way.

<!-- ──────────────────── END COMMENT ───────────────────── -->

## Note

Posting here instead of filing separately means the range defect rides in a
thread whose title and requested fix are about notation. If it gets resolved as
"emit exponent form", this is still broken — `3.4028235e38f` is rejected. The
comment says so explicitly, which is the mitigation. If a maintainer replies
treating them as one thing, that is the moment to file the standalone issue,
which is written and ready in `naga-f32-literal-range.md`.
