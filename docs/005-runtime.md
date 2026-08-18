# 005 — The runtime: what TypeGPU absorbs, and what it structurally cannot

Status: **adapter built and statically verified; the timing number is pending a
browser run.** Code: `spike/wgsl-baseline/typegpu-runner.js`,
`spike/wgsl-baseline/check-typegpu-layout.mjs`.

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

TypeGPU ships as fully self-contained relative-path ESM (166 modules, zero bare
specifiers), so `npm run vendor` is a plain `cp -R` into the harness directory —
no bundler and no import map. The harness is served by `python3 -m http.server`
rooted at `spike/wgsl-baseline/`, which cannot see the repo root's
`node_modules`; vendoring is what makes it reachable, and the directory is
gitignored.

## 6. Measuring it honestly

`once()` is **not** reimplemented in the adapter. Both runners call
`createDispatcher` from `measure.js`, which owns the query set and the compute
pass, so the timed path is the same code and the only difference between the two
variants is who allocated the resources. `matmul.html` runs
`tessera direct, TypeGPU resources` against `tessera direct` on the same WGSL,
the same manifest and the same dispatch, interleaved, and checks the outputs are
bit-identical.

The expected result is *no difference* — a resource layer that costs quanta
would be a finding worth chasing before adopting it. The number goes here once
the harness has been run.

## 7. What this does not settle

Only the compute path is covered. Textures, samplers, vertex layouts, multiple
bind groups and uniform buffers are all untouched, because tessera emits none of
them. `d.arrayOf(d.f32, n)` is the only schema the adapter constructs, which is
also the only thing tessera's bindings are today; a kernel taking an `i32` index
buffer or a struct would need this widened.
