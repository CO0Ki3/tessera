#!/usr/bin/env python3
"""Extract the spirv.module block from mlir-opt output.

`--convert-gpu-to-spirv` deliberately LEAVES the original gpu.module in place —
the design intent is that a host pipeline (gpu-to-llvm) consumes it afterwards to
build the launch code. For offline shader compilation there is no host pipeline,
and `mlir-translate --serialize-spirv` registers only the builtin and spirv
dialects, so the leftover gpu.module is a parse error:

    error: Dialect `gpu' not found for custom op 'gpu.module'

Neither --symbol-dce nor --gpu-module-to-binary removes it. So we slice it out.

This is a harness concern, not a design one: tessera controls its own emission
and can produce a module containing only the spirv.module. Keeping the step
explicit means the spike measures the real toolchain rather than a convenience
wrapper hiding a rough edge.

    usage: extract-spirv-module.py < lowered.mlir > spirv-only.mlir
"""
import sys


def extract(text):
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.lstrip().startswith("spirv.module")), None)
    if start is None:
        sys.exit("no spirv.module found — did --convert-gpu-to-spirv actually run?")

    # Brace-match from the opening line. Safe here: MLIR string literals in this
    # output ("None", "GLCompute", "LocalSize") contain no braces.
    depth = 0
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth == 0 and i > start:
            return "\n".join(lines[start:i + 1]) + "\n"
    sys.exit("unbalanced braces while slicing spirv.module")


if __name__ == "__main__":
    sys.stdout.write(extract(sys.stdin.read()))
