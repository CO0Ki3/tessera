#!/usr/bin/env python3
"""Emit the MLIR for the tiled matmul in spike/wgsl-baseline/matmul.wgsl.

This is a stand-in for tessera's eventual emitter, written the same way tessera
will have to write it: constants folded in Python from the tile and axis
literals, MLIR produced as text, no C++ bindings anywhere.

The structure mirrors the hand-written WGSL statement for statement so the two
can be diffed:

    for kk in 0..K step BK:
        stage A block, stage B block      (each 1024 elements over 256 threads)
        barrier
        for k in 0..BK:
            aFrag[TM], bFrag[TN], TM*TN fused multiply-adds
        barrier
    relu + store the TM*TN register fragment

Buffers are FLAT 1-D memrefs, exactly as the WGSL sees `array<f32>`. That is not
a simplification -- it sidesteps strided memrefs and subviews entirely, which is
where MemRefToSPIRV is least tolerant.

    usage: gen-matmul-mlir.py [--scalar|--vector] > matmul.mlir
"""
import sys

M, N, K = 1024, 768, 512
BM, BN, BK = 64, 64, 16
WGX, WGY = 16, 16
TM, TN = BM // WGY, BN // WGX      # 4, 4 -- the constant fold
THREADS = WGX * WGY                # 256
NACC = TM * TN                     # 16

SC = "#spirv.storage_class<StorageBuffer>"
WG = "#spirv.storage_class<Workgroup>"

assert M % BM == 0 and N % BN == 0 and K % BK == 0, "exact axes only"
assert (BM * BK) % THREADS == 0 and (BK * BN) % THREADS == 0
assert (BM * BN) == THREADS * NACC


class Emit:
    def __init__(self):
        self.lines = []
        self.n = 0
        self.consts = {}

    def __call__(self, s="", indent=0):
        self.lines.append("  " * indent + s if s else "")

    def tmp(self, base="v"):
        self.n += 1
        return f"%{base}{self.n}"

    def idx(self, k, ind):
        """An index constant, emitted once and reused."""
        if k not in self.consts:
            name = f"%c{k}"
            self.consts[k] = name
            self.pending.append((name, k, ind))
        return self.consts[k]


def gen(vectorised=False):
    e = Emit()
    e.pending = []

    e("module attributes {gpu.container_module} {")
    e("gpu.module @kernels {", 1)
    e(f"gpu.func @matmul(%a: memref<{M*K}xf32, {SC}>,", 2)
    e(f"                 %b: memref<{K*N}xf32, {SC}>,", 2)
    e(f"                 %c: memref<{M*N}xf32, {SC}>) kernel", 2)
    e(f"    attributes {{spirv.entry_point_abi = "
      f"#spirv.entry_point_abi<workgroup_size = [{WGX}, {WGY}, 1]>}} {{", 2)

    body = []
    B = lambda s="", i=3: body.append("  " * i + s if s else "")

    # ---- constants ---------------------------------------------------------
    needed = sorted({0, 1, BK, BM, BN, K, N, THREADS, TM, TN, BM * BK, BK * BN}
                    | set(range(max(TM, TN))))   # fragment offsets 0..TM-1 / 0..TN-1
    for v in needed:
        B(f"%c{v} = arith.constant {v} : index")
    B("%f0 = arith.constant 0.0 : f32")
    B()

    # ---- workgroup staging -------------------------------------------------
    B(f"%As = memref.alloc() : memref<{BM*BK}xf32, {WG}>")
    B(f"%Bs = memref.alloc() : memref<{BK*BN}xf32, {WG}>")
    B()

    # ---- ids ---------------------------------------------------------------
    B("%bx = gpu.block_id x")
    B("%by = gpu.block_id y")
    B("%tx = gpu.thread_id x")
    B("%ty = gpu.thread_id y")
    B(f"%tid0 = arith.muli %ty, %c{WGX} : index")
    B("%tid = arith.addi %tid0, %tx : index")
    B(f"%ty4 = arith.muli %ty, %c{TM} : index")
    B(f"%tx4 = arith.muli %tx, %c{TN} : index")
    B(f"%rowBase = arith.muli %by, %c{BM} : index")
    B(f"%colBase = arith.muli %bx, %c{BN} : index")
    B()

    # ---- outer reduction loop ---------------------------------------------
    accs = [f"%acc{i}" for i in range(NACC)]
    init = ", ".join(f"{a} = %f0" for a in accs)
    types = ", ".join(["f32"] * NACC)
    B(f"%out:{NACC} = scf.for %kk = %c0 to %c{K} step %c{BK} "
      f"iter_args({init}) -> ({types}) {{")

    # stage A: As[i] = a[(rowBase + i/BK)*K + (kk + i%BK)]
    B(f"scf.for %i = %tid to %c{BM*BK} step %c{THREADS} {{", 4)
    B(f"%r = arith.divui %i, %c{BK} : index", 5)
    B(f"%cc = arith.remui %i, %c{BK} : index", 5)
    B("%gr = arith.addi %rowBase, %r : index", 5)
    B(f"%gr2 = arith.muli %gr, %c{K} : index", 5)
    B("%gc = arith.addi %kk, %cc : index", 5)
    B("%off = arith.addi %gr2, %gc : index", 5)
    B(f"%val = memref.load %a[%off] : memref<{M*K}xf32, {SC}>", 5)
    B(f"memref.store %val, %As[%i] : memref<{BM*BK}xf32, {WG}>", 5)
    B("}", 4)

    # stage B: Bs[i] = b[(kk + i/BN)*N + (colBase + i%BN)]
    B(f"scf.for %i = %tid to %c{BK*BN} step %c{THREADS} {{", 4)
    B(f"%r = arith.divui %i, %c{BN} : index", 5)
    B(f"%cc = arith.remui %i, %c{BN} : index", 5)
    B("%gr = arith.addi %kk, %r : index", 5)
    B(f"%gr2 = arith.muli %gr, %c{N} : index", 5)
    B("%gc = arith.addi %colBase, %cc : index", 5)
    B("%off = arith.addi %gr2, %gc : index", 5)
    B(f"%val = memref.load %b[%off] : memref<{K*N}xf32, {SC}>", 5)
    B(f"memref.store %val, %Bs[%i] : memref<{BK*BN}xf32, {WG}>", 5)
    B("}", 4)

    B("gpu.barrier", 4)

    # ---- inner accumulate --------------------------------------------------
    inner_init = ", ".join(f"%i{i} = {accs[i]}" for i in range(NACC))
    B(f"%acc:{NACC} = scf.for %k = %c0 to %c{BK} step %c1 "
      f"iter_args({inner_init}) -> ({types}) {{", 4)

    for m in range(TM):
        B(f"%am{m}a = arith.addi %ty4, %c{m} : index", 5)
        B(f"%am{m}b = arith.muli %am{m}a, %c{BK} : index", 5)
        B(f"%am{m}c = arith.addi %am{m}b, %k : index", 5)
        B(f"%af{m} = memref.load %As[%am{m}c] : memref<{BM*BK}xf32, {WG}>", 5)
    for n in range(TN):
        B(f"%bn{n}a = arith.muli %k, %c{BN} : index", 5)
        B(f"%bn{n}b = arith.addi %tx4, %c{n} : index", 5)
        B(f"%bn{n}c = arith.addi %bn{n}a, %bn{n}b : index", 5)
        B(f"%bf{n} = memref.load %Bs[%bn{n}c] : memref<{BK*BN}xf32, {WG}>", 5)

    outs = []
    for m in range(TM):
        for n in range(TN):
            i = m * TN + n
            B(f"%p{i} = arith.mulf %af{m}, %bf{n} : f32", 5)
            B(f"%s{i} = arith.addf %i{i}, %p{i} : f32", 5)
            outs.append(f"%s{i}")
    B(f"scf.yield {', '.join(outs)} : {types}", 5)
    B("}", 4)

    B("gpu.barrier", 4)
    B(f"scf.yield {', '.join(f'%acc#{i}' for i in range(NACC))} : {types}", 4)
    B("}", 3)
    B()

    # ---- relu + store ------------------------------------------------------
    for m in range(TM):
        B(f"%sr{m}a = arith.addi %ty4, %c{m} : index")
        B(f"%sr{m} = arith.addi %rowBase, %sr{m}a : index")
        B(f"%sr{m}n = arith.muli %sr{m}, %c{N} : index")
        for n in range(TN):
            i = m * TN + n
            B(f"%sc{i}a = arith.addi %tx4, %c{n} : index")
            B(f"%sc{i} = arith.addi %colBase, %sc{i}a : index")
            B(f"%so{i} = arith.addi %sr{m}n, %sc{i} : index")
            # relu as cmpf+select, NOT arith.maximumf/maxnumf: MLIR lowers those
            # with IEEE NaN-propagation semantics, which emits OpIsNan, which
            # naga's SPIR-V frontend rejects ("Unsupported relational function:
            # IsNan"). cmpf ogt + select is also exactly `v > 0 ? v : 0`, which
            # is what the CPU oracle computes -- so this is the faithful form too.
            B(f"%rc{i} = arith.cmpf ogt, %out#{i}, %f0 : f32")
            B(f"%rl{i} = arith.select %rc{i}, %out#{i}, %f0 : f32")
            B(f"memref.store %rl{i}, %c[%so{i}] : memref<{M*N}xf32, {SC}>")
    B()
    B("gpu.return")

    e.lines.extend(body)
    e("}", 2)
    e("}", 1)
    e("}")
    return "\n".join(e.lines) + "\n"


if __name__ == "__main__":
    sys.stdout.write(gen(vectorised="--vector" in sys.argv))
