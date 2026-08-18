# NOT AN ISSUE — TypeGPU's `process.env` read is by design

**Verdict: do not file.** This was written up as an upstream defect and it is not
one. Kept because the reasoning cost two browser runs and a false start, and
because the retraction is the useful part.

## The claim that was going to be filed

`typegpu@0.12.0`'s `shared/env.js` reads `process.env.NODE_ENV` at module top
level:

```js
export const DEV = process.env.NODE_ENV === 'development';
export const TEST = process.env.NODE_ENV === 'test';
```

`env.js` is imported by `errors.js`, so it is in everything. Serving the package
directory over a plain HTTP server and loading `index.js` from a browser fails
with `ReferenceError: process is not defined`, taking the whole 166-module graph
with it. The draft issue argued that `"exports": { ".": "./index.js" }` with no
`browser` condition made that a broken promise.

## Why it is wrong

**The premise was that TypeGPU intends its published ESM to be served raw to a
browser. It does not, and nothing ever said it did.**

- The [getting-started docs](https://docs.swmansion.com/TypeGPU/getting-started)
  document exactly two paths, the CLI and a manual npm install, both ending in a
  build step. No CDN link, no `<script type="module">`, no import map, no
  unbundled usage anywhere.
- The headline feature — writing shaders in TypeScript — *requires* a build
  plugin: *"TypeGPU features related to writing shaders in JavaScript/TypeScript
  rely on an additional build tool that hooks into existing bundlers called
  unplugin-typegpu"*. A bundler is not an implementation detail of the
  recommended setup; it is the setup.
- `package.json` carries `engines: { node: ">=12.20.0" }` and no `browser`,
  `unpkg`, or `jsdelivr` field, and the tarball ships no UMD/IIFE/minified
  bundle. Every signal says "consumed by a build tool".
- **And the no-bundler path works.** esm.sh serves a transformed build with
  `process.env` substituted — 0 occurrences — and no bare specifiers:

  ```
  $ curl -sL https://esm.sh/typegpu@0.12.0/es2022/common.mjs | grep -c process.env
  0
  ```

  A transforming CDN is the ecosystem's answer for "no bundler", and it is
  already answered.

`process.env.NODE_ENV` guarded by dead-code elimination is a long-standing
convention — React ships the same thing. Filing it as a defect would be reporting
a package for following a convention, against a use case its documentation never
offers.

## What was actually true, and still is

Nothing about TypeGPU. Two facts about **our** harness:

1. It is served by `python3 -m http.server` rooted at `spike/wgsl-baseline/`,
   with no build step, because a compiler's test harness should not need one.
2. That is an unsupported way to consume TypeGPU, so we do the transform
   ourselves in `vendor.mjs`.

Vendoring is a *choice* — esm.sh would also work and needs no script — made
because a local harness that requires the network to run its own kernels is
worse than one that does not. The framing "we work around an upstream defect" was
wrong; the framing is "we consume it in an unsupported way on purpose, and pay
for that ourselves".

## The lesson worth keeping

The defect report was three sections of verified detail — reproduction, boundary,
scope, suggested one-line fix — built on a premise nobody had checked: *does this
project intend to support the thing I am doing?* Everything downstream of that
was correct and irrelevant.

The question that caught it was "typegpu가 실제로 돌리는 환경이 맞는지 확인해줄래,
애초에 의도하지 않은 걸 수도 있잖아" — asked before filing, not after. Check the
premise before the details, because the details cannot fail in a way that reveals
a wrong premise. Compare `naga-f32-literal-range.md`, where the equivalent
question — which behaviour does the spec mandate — could not be answered, so the
issue asks it instead of assuming.
