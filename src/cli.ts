#!/usr/bin/env -S npx tsx
/**
 * tessera build <kernel.ts> [-o outdir]
 *
 * The walking skeleton: one TypeScript file in, one WGSL file out, through MLIR.
 *
 *   kernel.ts --> tsc checker --> tessera IR --> MLIR
 *             --> mlir-opt --> spirv.module --> mlir-translate --> .spv
 *             --> naga --> .wgsl
 *
 * Every stage is a real tool driven as a subprocess over text. No C++ bindings,
 * no LLVM inside the Node toolchain.
 */

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { compileToIR, FrontendError } from "./frontend.ts";
import { emitMLIR } from "./emit-mlir.ts";
import type { KernelIR } from "./ir.ts";

const HERE = dirname(fileURLToPath(import.meta.url));
const LLVM = process.env.LLVM_PREFIX ?? "/opt/homebrew/opt/llvm";
const MLIR_OPT = join(LLVM, "bin", "mlir-opt");
const MLIR_TRANSLATE = join(LLVM, "bin", "mlir-translate");
const EXTRACT = resolve(HERE, "..", "spike", "mlir-spirv", "extract-spirv-module.py");

const ATTACH =
  "--spirv-attach-target=ver=v1.3 caps=Shader exts=SPV_KHR_storage_buffer_storage_class";

function run(cmd: string, args: string[], input?: string): string {
  try {
    return execFileSync(cmd, args, { input, encoding: "utf8", maxBuffer: 64 << 20 });
  } catch (err) {
    const e = err as { stderr?: string; message: string };
    throw new Error(`${cmd} failed:\n${(e.stderr ?? e.message).trim()}`);
  }
}

function report(ir: KernelIR): void {
  const [wgx, wgy, wgz] = ir.workgroup;
  const [tm, tn] = ir.fragment;
  console.log(`  kernel      ${ir.name}  (${ir.dtype})`);
  console.log(`  tile        ${ir.tile.bm} x ${ir.tile.bn} x ${ir.tile.bk}`);
  console.log(`  grid        ${ir.grid.map((a) => `${a.name}=${a.extent}/${a.block}`).join("  ")}`);
  console.log(`  reduce      ${ir.reduce.map((a) => `${a.name}=${a.extent}/${a.block}`).join("  ")}`);
  console.log(`  bindings    ${ir.bindings.map((b) => `${b.name}[${b.axes.join(",")}]:${b.mode}`).join("  ")}`);
  console.log(`  workgroup   ${wgx}x${wgy}x${wgz}   fragment ${tm}x${tn}   ` +
              `wgBytes ${ir.workgroupBytes}   dispatch ${ir.dispatch.join("x")}`);
  console.log(`  masks       none  (every axis divides its block exactly)`);
}

function main(): void {
  const argv = process.argv.slice(2);
  const args = argv[0] === "build" ? argv.slice(1) : argv;
  const oIdx = args.indexOf("-o");
  const outDir = oIdx >= 0 ? args[oIdx + 1] : "build";
  const entry = args.find((x, i) => !x.startsWith("-") && i !== oIdx + 1);

  if (!entry) {
    console.error("usage: tessera build <kernel.ts> [-o outdir]");
    process.exit(2);
  }
  for (const t of [MLIR_OPT, MLIR_TRANSLATE]) {
    if (!existsSync(t)) {
      console.error(`missing ${t}\n  brew install llvm   (or set LLVM_PREFIX)`);
      process.exit(2);
    }
  }

  const entryPath = resolve(entry);
  mkdirSync(outDir, { recursive: true });
  const at = (ext: string) => join(outDir, `${base}${ext}`);
  let base = "kernel";

  // ---- front end ---------------------------------------------------------
  console.log(`── front end ──────────────────────────────────────`);
  const ir = compileToIR(entryPath);
  base = ir.name;
  report(ir);

  // ---- MLIR --------------------------------------------------------------
  console.log(`\n── emit ───────────────────────────────────────────`);
  const mlir = emitMLIR(ir);
  writeFileSync(at(".mlir"), mlir);
  console.log(`  ${at(".mlir")}  ${mlir.split("\n").length} lines`);

  // ---- lower + serialize -------------------------------------------------
  const lowered = run(MLIR_OPT, [
    ATTACH, "--convert-gpu-to-spirv", "--spirv-lower-abi-attrs", "--spirv-update-vce",
    at(".mlir"),
  ]);
  const spirvMlir = run("python3", [EXTRACT], lowered);
  writeFileSync(at(".spirv.mlir"), spirvMlir);
  console.log(`  ${at(".spirv.mlir")}  ${spirvMlir.split("\n").length} lines`);

  run(MLIR_TRANSLATE, [
    "--no-implicit-module", "--serialize-spirv", at(".spirv.mlir"), "-o", at(".spv"),
  ]);
  console.log(`  ${at(".spv")}  ${readFileSync(at(".spv")).length} bytes`);

  // ---- validate + ingest -------------------------------------------------
  console.log(`\n── validate ───────────────────────────────────────`);
  try {
    run("spirv-val", ["--target-env", "vulkan1.1", at(".spv")]);
    console.log(`  spirv-val   ✓ VALID`);
  } catch (e) {
    console.log(`  spirv-val   ✗ ${(e as Error).message.split("\n")[1] ?? ""}`);
    process.exitCode = 1;
  }

  run("naga", [at(".spv"), at(".wgsl")]);
  const wgsl = readFileSync(at(".wgsl"), "utf8");
  console.log(`  naga        ✓ ${at(".wgsl")}  ${wgsl.split("\n").length} lines`);
  console.log(`\n  entry point: ${ir.name}   @workgroup_size(${ir.workgroup.join(", ")})`);
}

try {
  main();
} catch (err) {
  if (err instanceof FrontendError) {
    console.error(`\ntessera: ${err.message}`);
  } else {
    console.error(`\ntessera: ${(err as Error).message}`);
  }
  process.exit(1);
}
