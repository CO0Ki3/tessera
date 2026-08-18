# TypeGPU: published ESM is not loadable by a browser (`process.env` at module top level)

- **Repo**: [software-mansion/TypeGPU](https://github.com/software-mansion/TypeGPU)
- **Affects**: 0.12.0 (also the latest published version, checked 0.12.0)
- **Duplicate check**: no matching issue — `is:issue process.env` returns no results
- **Status**: ready to file, not yet filed
- **In tessera**: worked around in `spike/wgsl-baseline/vendor.mjs`, guarded by `npm test`

Everything between the markers below is the issue body, verbatim. The section
after it is ours and does not belong upstream.

<!-- ─────────────────── BEGIN ISSUE BODY ─────────────────── -->

### Summary

`shared/env.js` reads `process.env` at module top level, so the published ESM
throws `ReferenceError: process is not defined` when loaded directly in a
browser. Because `env.js` is imported by `errors.js`, this takes down the whole
module graph — nothing loads.

```js
// shared/env.js
export const DEV = process.env.NODE_ENV === 'development';
export const TEST = process.env.NODE_ENV === 'test';
```

The file's comment states the assumption:

> Even though the value of this constant uses Node.js specific APIs, pretty much
> every bundler replaces the expression below with either `development` or
> `production`

That holds for the documented setups. It also means the package as published is
not loadable by the runtime it targets, while `package.json` advertises

```json
"exports": { ".": "./index.js", "./data": "./data/index.js", ... }
```

with no `browser` condition and no bundled build. A consumer reading that has
every reason to expect `./index.js` to be a loadable ES module.

### Why this is easy to miss

**It is invisible from Node.** `await import('typegpu')` in a Node script
succeeds, because Node has `process`. A test suite or CI job that imports the
package passes while direct browser loading is broken.

### Reproduction

TypeGPU's three runtime dependencies have to be resolved first or the page fails
on those instead, so the import map below covers them. With that in place,
`process` is the only thing left:

```sh
npm i typegpu@0.12.0
mkdir -p www
for p in typegpu tsover-runtime typed-binary tinyest; do cp -R node_modules/$p www/$p; done
cat > www/index.html <<'HTML'
<script type="importmap">{"imports":{
  "typegpu":        "./typegpu/index.js",
  "tsover-runtime": "./tsover-runtime/dist/index.js",
  "typed-binary":   "./typed-binary/dist/index.js",
  "tinyest":        "./tinyest/index.js"
}}</script>
<script type="module">import tgpu from "typegpu"; console.log(tgpu);</script>
HTML
python3 -m http.server -d www 8080
```

Open `http://localhost:8080/`:

```
Uncaught ReferenceError: process is not defined
    at typegpu/shared/env.js:9
```

Contrast with `node -e "import('typegpu').then(m => console.log(!!m.default))"`,
which prints `true`.

### Scope

This is only about `process.env`. Shipping bare specifiers for real dependencies
is normal, and resolving them is the consumer's job — the import map above is not
a complaint. The `process.env` read is different in kind, because it fails **even
with** a correct import map: there is no `process` to read.

### Suggested fix

Guarding the read is one line, changes nothing under any bundler (the expression
still folds), and makes the published `exports` entries honest:

```js
const env = typeof process !== 'undefined' ? process.env : {};
export const DEV = env.NODE_ENV === 'development';
export const TEST = env.NODE_ENV === 'test';
```

Alternatives, if preferred: ship a `browser` condition in `exports` pointing at a
variant with the constants inlined to `false`, or document that a bundler is
required and that the `exports` entries are not directly loadable.

Happy to open a PR for the one-line version if that is welcome.

<!-- ──────────────────── END ISSUE BODY ───────────────────── -->

## Not part of the issue — what tessera does about it

`spike/wgsl-baseline/vendor.mjs` performs the substitution a bundler would
(`DEV = false`, `TEST = false`) and rewrites the three bare specifiers to
relative paths, so the vendored tree loads from a plain `python3 -m http.server`
with no bundler and no import map.

It then refuses to trust itself. It rescans the tree for any remaining
`process.` / `require(` / `node:` / `__dirname` reference and walks the module
graph the browser will fetch. `vendor.mjs --check` runs both against an existing
tree and is wired into `npm test`, so a TypeGPU upgrade that reintroduces either
problem fails there rather than in a browser console.

Every guard was falsified before being trusted, including the scanner itself:

| perturbation | caught by |
|---|---|
| restore `from 'tsover-runtime'` (single quotes) | bare-specifier scan |
| restore `process.env.NODE_ENV` | host-only scan |
| delete a reachable leaf module | module-graph walk |
| break the scanner to double-quotes-only | **positive control** |

The positive control exists because the first version of this check reported
"0 bare imports" from a grep that only matched double quotes, while TypeGPU
writes `from 'tsover-runtime'`. A check that returns zero is worthless until it
has been shown capable of returning non-zero.
