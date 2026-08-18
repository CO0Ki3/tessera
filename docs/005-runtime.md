# 005 — The runtime: what TypeGPU absorbs, and what it structurally cannot

Status: **adapter built and statically verified; the first browser run found an
upstream blocker and a bundler assumption, both worked around; the resource
layer measured free and bit-identical.**
Code: `spike/wgsl-baseline/typegpu-runner.js`, `vendor.mjs`,
`check-typegpu-layout.mjs`.

## 1. tessera has no host-side story, and that is where its bugs came from

The compiler emits WGSL and a manifest. Everything after that — allocating
buffers, uploading, wiring a bind group, dispatching, reading back — the harness
does by hand, and that hand-written host code is where this project has actually
been wrong:

- the entry point was hardcoded, and naga renamed it (`matmul_relu_f32` →
  `matmul_relu_f32_`, because the original ends in a reserved word);
- the dispatch was recomputed host-side as `[N/64, M/64, 1]`, which is neither
  integer division nor a ceiling, so for 1000×750 the last block in each
  dimension was never dispatched and the ragged kernel looked like it had a mask
  bug. **The masks were correct. The host ignored a manifest that said the right
  thing.**

Both are the same failure: the host re-deriving something the compiler already
decided. The rule that came out of it — *anything the compiler already decided is
read, never re-derived* — is a rule about a layer tessera does not have.

[TypeGPU](https://github.com/software-mansion/TypeGPU) is that layer, and it is
better at it than anything invented here would be. So the question is not whether
tessera should compete with it, but whether tessera can sit on it.

## 2. What fits: the resource half

| | owner |
|---|---|
| bind group layout | **TypeGPU** — `tgpu.bindGroupLayout({ a: { storage: arrayOf(f32, n), access: "readonly" } })` |
| buffers | **TypeGPU** — `root.createBuffer(schema).$usage("storage")`, flags derived |
| upload | **TypeGPU** — `buf.write(src.buffer)` |
| bind group | **TypeGPU** — `root.createBindGroup(layout, bufs)`, by **name**, not index |
| readback | **TypeGPU** — `await buf.read()`, staging buffer and mapping handled |
| shader module, pipeline | raw WebGPU |
| dispatch | raw WebGPU |

The bind group is the part worth having. The raw path writes

```js
entries: bindings.map((bd, i) => ({ binding: i, resource: { buffer: bufs[i] } }))
```

— an index-matching between two arrays, with nothing checking that they are the
same two arrays. The TypeGPU path passes `{ a, b, c }` and a wrong name does not
compile.

## 3. What does not fit, and why it is structural

**Pipeline creation cannot go through TypeGPU.** `TgpuComputePipeline`'s
descriptor is

```ts
{ compute: TgpuComputeFn<Input> }
```

and nothing else. A `TgpuComputeFn` only exists if the body went through
TypeGPU's own TS→WGSL path, so there is no way to hand it a string. Buffers have
a wrap-an-existing-`GPUBuffer` overload; pipelines have no equivalent.

This is not a gap to file an issue about. TypeGPU's whole value is that it knows
what a resource *contains*, and it cannot know that about WGSL somebody else
emitted. Asking for `createComputePipeline(someWgslString)` is asking it to give
up the property that makes it worth using.

So the seam sits exactly there — and it is a *clean* seam only because
`root.unwrap(layout)` returns a real `GPUBindGroupLayout`. That lets the pipeline
be built with an **explicit** layout:

```js
device.createPipelineLayout({ bindGroupLayouts: [root.unwrap(layout)] })
```

With `layout: "auto"` the two halves would each invent their own binding
assignment and the bind group would be rejected. `unwrap` is what makes the
adapter possible at all.

## 4. The risk it moves, and the test that pins it

The adapter does not remove risk; it *moves* it. The layout now comes from the
manifest while the pipeline comes from the WGSL, and WebGPU validates the bind
group against the **layout** — so if those two disagree about which buffer is
binding 0, every kernel still runs, with `a` and `b` swapped, and produces a
plausible wrong answer.

That agreement is decidable on the CPU, so `check-typegpu-layout.mjs` decides it:
it builds the layout exactly as the adapter does, greps the emitted WGSL for its
`@group/@binding` declarations, and compares index, group, element type, access
mode and name. It runs in `npm test` and in `npm run demo`.

Both negatives were exercised before it was trusted:

```
swap two bindings in the manifest
  binding 0: TypeGPU would put "b" there, the WGSL declares "a"        FAIL
flip the output binding to read
  c: WGSL is <storage, read_write>, the manifest says "read"           FAIL
```

The first is precisely the silent-wrong-answer case.

## 5. The dependency stays optional

A compiler that hard-requires a pre-1.0 runtime is worse than one that does not.
So the adapter is a **peer** of the raw runner, not a replacement:
`matmul.html` imports it with a dynamic `import()` in a `try`, and if
`npm run vendor` was never run the page reports `skipped` and measures everything
else. `check-typegpu-layout.mjs` skips itself the same way.

The harness is served by `python3 -m http.server` rooted at
`spike/wgsl-baseline/`, which cannot see the repo root's `node_modules`, so
TypeGPU is vendored into the harness directory (gitignored). See §6.

## 6. Vendoring took three attempts, and the third one is the method

The plan was `cp -R node_modules/typegpu` and nothing else. It took two browser
runs to find out why that does not work, and the interesting part is not either
defect — it is that **both were invisible from Node**, and that the check which
was supposed to catch the second one reported success.

### Attempt 1 — `process.env` at module top level

`shared/env.js` evaluates:

```js
export const DEV = process.env.NODE_ENV === 'development';
```

`env.js` is imported by `errors.js`, so it is in everything. In a browser
`process` is undefined and the entire graph dies with a `ReferenceError`. The
file's own comment names the assumption — *"pretty much every bundler replaces
the expression below"* — which is true of the documented setups and means the
package as published is not loadable by the runtime it targets. This is a real
upstream defect, because it fails even with a correct import map:
[`spike/upstream/typegpu-process-env-esm.md`](../spike/upstream/typegpu-process-env-esm.md).
The fix upstream is one line.

### Attempt 2 — three dependencies imported by bare specifier

TypeGPU imports `tsover-runtime`, `typed-binary` and `tinyest` by name. That is
normal and not a defect; resolving it is the consumer's job. Without a bundler
the options are an import map or rewriting the specifiers, and vendoring rewrites
them, so `typegpu-runner.js` stays importable from any page and from Node without
either knowing about this.

### The part worth keeping: a check that returns zero proves nothing

Before the first browser run, this was checked. The check reported
**"bare imports across the ESM: 0"**, and that number was believed.

The grep matched `from "..."`. TypeGPU writes `from 'tsover-runtime'`.

The check could not have returned anything but zero, and nothing in it would ever
have said so. It was not a weak check; it was a check whose failure mode was
indistinguishable from success. Same for the module-graph walk that ran next: it
followed relative specifiers and *silently skipped* bare ones, so it confirmed
that 166 files exist — true, and not the problem.

So `vendor.mjs` requires a **positive control**. It scans the unmodified tree
first and refuses to continue if it finds no bare specifiers:

```
the bare-specifier scanner found nothing in an UNMODIFIED tree.
That is not good news — typegpu imports tsover-runtime, typed-binary and
tinyest by bare specifier. The scanner is broken; fix it before trusting it.
```

That message is not hypothetical — reintroducing the double-quotes-only regex
produces it. Every guard was falsified before being trusted:

| perturbation | caught by |
|---|---|
| restore `from 'tsover-runtime'` (single quotes) | bare-specifier scan |
| restore `process.env.NODE_ENV` | host-only scan |
| delete a reachable leaf module | module-graph walk |
| break the scanner to double-quotes-only | **positive control** |

`vendor.mjs --check` runs the verification half against an existing tree and is in
`npm test`, so a TypeGPU upgrade that reintroduces either problem fails there
rather than in a browser console nobody is reading.

### The harness was also lying about why

The skip message said `run npm run vendor` for *any* load failure — and TypeGPU
*had* been vendored. A page that guesses at why it skipped something costs more
than one that says it does not know, so the error is now kept and printed
verbatim. The second failure was diagnosed from the page in one reading;
the first cost a round trip.

## 7. RESOLVED — the resource layer is free, and the output is identical

`once()` is **not** reimplemented in the adapter. Both runners call
`createDispatcher` from `measure.js`, which owns the query set and the compute
pass, so the timed path is the same code and the only difference between the two
variants is who allocated the resources.

Measured on the same WGSL, the same manifest and the same dispatch, interleaved
round-robin, 8 warm-up + 25 timed:

```
tessera direct, raw resources            7q     1755 GFLOP/s
tessera direct, TypeGPU resources        6q     2048 GFLOP/s
hand-written                             6q     2048 GFLOP/s

TypeGPU resources vs raw   bit-exact 786432 / 786432   IDENTICAL
```

**The layer costs nothing.** It cannot: the WGSL is byte-identical, the dispatch
comes from the same manifest, and the timed code is the same function. The ±1q is
the instrument.

That is worth stating carefully, because the harness first printed it as
`0.86x` — which reads as TypeGPU making the kernel *faster*, a thing a buffer
allocator cannot do. Chrome quantises timestamps to 65.536 µs, so 6q against 7q
is a rounding boundary rather than a result. The page no longer prints a ratio it
cannot resolve; a difference of one quantum or less now says `indistinguishable`.
This is the same failure mode as the retracted 1.5× claim in
[`docs/002`](002-performance.md) §2, caught earlier this time.

The bit-identity is the load-bearing check. It confirms the adapter binds
`a`, `b` and `c` to the same buffers in the same order as the raw path — which is
exactly the thing `check-typegpu-layout.mjs` verifies statically, now confirmed
dynamically on real data.

### The unplanned control this produced

Running the identical shader through two resource layers is, incidentally, a
direct measurement of the harness's own resolution: **identical work, measured
twice, differs by one quantum.** Every other comparison on the page should be
read against that floor. Recorded in `docs/002` §2a.

## 8. Coverage: all three harness pages, all four kernels

The adapter runs alongside the raw path everywhere, not just on the kernel it was
written against:

| page | kernel | bindings | why this one matters |
|---|---|---|---|
| `matmul.html` | 1024×768×512 aligned | 3 | the timing comparison — same WGSL, same dispatch, same timed code |
| `ragged.html` | 1000×750×500 | 3 | **18 synthesised masks.** A swapped operand still produces plausible finite numbers in the tails rather than an obvious crash, so bit-identity against the raw path is the check with teeth |
| `rowwise.html` | softmax, layernorm | **2** | the output binding is at index 1, not 2 — the case that would catch an adapter which only worked for the shape it was written against |

`check-typegpu-layout.mjs` covers the same four statically (manifest ↔ emitted
WGSL ↔ TypeGPU layout: index, group, element type, access mode, name).

Measured:

```
matmul      raw 7q    TypeGPU 6q     bit-exact 786432 / 786432   IDENTICAL
ragged      raw 7q    TypeGPU 7q     bit-exact 750000 / 750000   IDENTICAL
softmax                              bit-exact 768000 / 768000   IDENTICAL
layernorm                            bit-exact 768000 / 768000   IDENTICAL
```

**The bit-identity is the result; the timings on the rowwise kernels were not.**
softmax and layernorm first measured `min 1q` and `min 0q`, which is not a fast
kernel but an unmeasured one — the whole timed region fit inside one quantum. The
page now batches 64 dispatches per timed pass and prints fractional quanta, and
`measureInterleaved` returns `belowResolution` so a row like that says so instead
of looking like data. See [`docs/002`](002-performance.md) §2a.

### One constructor, because three would rot

The branch lives in `runners.js`, not in each page. Three copies of
`if (runtime === "typegpu")` is the duplication this project spent a week
removing from the emitter, and it would put the optional-dependency handling in
three places as well — which is where it rots, because only one of the three gets
read after a TypeGPU upgrade. Two structural tests hold the line, and both were
falsified before being trusted:

```
createRunner called directly in a page   ✗ these pages build a runner by hand instead of via runners.js
adapter dropped from one page            ✗ these pages do not exercise the TypeGPU adapter
```

## 9. What this does not settle

Only the compute path is covered. Textures, samplers, vertex layouts, multiple
bind groups and uniform buffers are all untouched, because tessera emits none of
them. `d.arrayOf(d.f32, n)` is the only schema the adapter constructs, which is
also the only thing tessera's bindings are today; a kernel taking an `i32` index
buffer or a struct would need this widened.

Nor is TypeGPU a dependency of the compiler. It is a dependency of the harness,
behind `runners.js`, and every page still runs without it.
