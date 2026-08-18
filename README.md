# tessera

A compiler that lowers kernels written in TypeScript to WebGPU.

```
kernel.ts ──▶ tsc typed AST ──▶ tessera IR ──▶ WGSL          (default)
                                          └──▶ MLIR ──▶ SPIR-V ──▶ WGSL   (--backend=mlir)
```

## Why

- Python has Triton and JAX. The TypeScript/web ecosystem has no option for WebGPU kernels other than writing WGSL by hand.
- TypeScript's type system (literal types, conditional types) can check **tensor shapes at the type level**. That's an advantage no Python frontend has.
- There was exactly one reason to use MLIR: **to ride the existing dialects** — `linalg`, `vector`, `gpu`, SPIR-V — and otherwise it is pure overhead. That rule got tested. `LinalgToSPIRV` does not exist upstream, so we never rode the dialect we wanted, and an A/B against a direct WGSL printer measured what the detour costs: **1.57×**. Direct emission is now the default; `--backend=mlir` remains as the second oracle. See [`docs/002`](docs/002-performance.md) §5.

## Scope principles

- Cut the language **by type, not by syntax**. The checker is the admission gate.
- `any`, dynamic indexing, prototype manipulation and friends must produce a **clear error** — never a silent fall-off into a slow path. (This is what AssemblyScript got right.)
- General-purpose TS→native migration is an explicit non-goal.

## v0 plan

1. Get a type-annotated AST via the TypeScript Compiler API.
2. No custom dialect yet — **emit MLIR text directly** and pipe it into an `mlir-opt` pipeline, so we have something end-to-end without writing any C++.
3. Prove the differentiation in the backend: one `matmul` going TS → WGSL → actually running in a browser.
4. Differential testing from day one — a CI job comparing against the same TS run through tsc + node.

Promote to IRDL or a real C++ dialect only when the abstraction starts paying for itself.

## Prior art to survey

- `ASDAlexander77/TypeScriptCompiler` — `ts` dialect → LLVM. Aims at the general-purpose case rather than ours, but worth seeing where it hit walls.
- AssemblyScript, Static TypeScript (MakeCode), Porffor — reference points for where to draw the subset line.
