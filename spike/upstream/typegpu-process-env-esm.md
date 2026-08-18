# TypeGPU: published ESM reads `process.env` at module top level, so it cannot be loaded by a browser without a bundler

- **Project**: [software-mansion/TypeGPU](https://github.com/software-mansion/TypeGPU)
- **Version**: 0.12.0
- **File**: `shared/env.js` (published, in the npm tarball)
- **Status**: not yet filed
- **Impact on tessera**: worked around in `spike/wgsl-baseline/vendor.mjs`; guarded by `npm test`

## What happens

`package.json` advertises the package as plain ESM:

```json
"exports": { ".": "./index.js", "./data": "./data/index.js", ... }
```

There is no `browser` field and no bundled build. Loading `./index.js` in a browser
with a plain `<script type="module">` — no bundler, no import map beyond a relative
path — fails immediately:

```
Uncaught ReferenceError: process is not defined
    at shared/env.js:9
```

because `shared/env.js` evaluates, at module top level:

```js
export const DEV = process.env.NODE_ENV === 'development';
export const TEST = process.env.NODE_ENV === 'test';
```

`env.js` is imported by the core of the library, so the whole graph — 166 modules
reachable from `index.js` + `data/index.js` — fails to evaluate. Nothing loads.

## Why it is easy to miss

The file's own comment names the assumption:

> Even though the value of this constant uses Node.js specific APIs, pretty much
> every bundler replaces the expression below with either `development` or
> `production`

That is true of the documented setups (Vite, Next, Bun). But it means the package
as published is not loadable by the runtime it targets, and the failure is
invisible from Node: `await import("typegpu")` in a Node script succeeds, because
Node has `process`. So a test suite that imports the package passes while the
browser build is broken.

It also makes the `exports` map misleading. A consumer reading it has every
reason to expect `./index.js` to be a loadable ES module.

## Suggested fixes, cheapest first

1. **Guard the read.** One line, no build change, no behavioural difference under
   any bundler, since the expression still folds:

   ```js
   const env = typeof process !== 'undefined' ? process.env : {};
   export const DEV = env.NODE_ENV === 'development';
   export const TEST = env.NODE_ENV === 'test';
   ```

2. Ship a `browser` condition in `exports` pointing at a variant with the
   constants inlined to `false`.

3. Document that a bundler is required, and that the `exports` entries are not
   directly loadable.

(1) is enough and costs nothing.

## Reproduction

```
npm i typegpu@0.12.0
mkdir -p www && cp -R node_modules/typegpu www/typegpu
printf '<script type="module">import tgpu from "./typegpu/index.js"; console.log(tgpu);</script>' > www/i.html
python3 -m http.server -d www 8080
# open http://localhost:8080/i.html -> ReferenceError: process is not defined
```

Contrast with `node -e 'import("typegpu").then(m => console.log(!!m.default))'`, which
prints `true`.

## What tessera does about it

`spike/wgsl-baseline/vendor.mjs` performs the substitution a bundler would
(`DEV = false`, `TEST = false`), then rescans the whole vendored tree for any
remaining `process.` / `require(` / `node:` / `__dirname` reference and walks the
module graph for missing files. Both checks run in `npm test` via
`vendor.mjs --check`, and both were falsified before being trusted, so a TypeGPU
upgrade that reintroduces a host-only reference fails there rather than in a
browser console.
