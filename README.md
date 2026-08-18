# tessera

A compiler that lowers kernels written in TypeScript to the GPU by way of MLIR.

```
kernel.ts ──▶ tsc typed AST ──▶ MLIR (linalg/vector/scf) ──▶ SPIR-V / WGSL
```

## Why

- Python has Triton and JAX. The TypeScript/web ecosystem has no option for WebGPU kernels other than writing WGSL by hand.
- TypeScript's type system (literal types, conditional types) can check **tensor shapes at the type level**. That's an advantage no Python frontend has.
- There is exactly one reason to use MLIR here: **to ride the existing dialects** — `linalg`, `vector`, `gpu`, SPIR-V. If we were going straight to LLVM, MLIR would be pure overhead.

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
