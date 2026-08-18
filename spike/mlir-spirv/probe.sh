#!/usr/bin/env bash
# probe.sh — what does this toolchain actually have?
#
# Run this before writing any MLIR. The week-0 question is whether the
# MLIR -> SPIR-V -> WGSL path exists at all on a stock install; finding out by
# hitting errors halfway through a hand-written matmul wastes the cheap part of
# the schedule. Nothing here is guessed — every line asks the tool.

set -uo pipefail

PREFIX="${LLVM_PREFIX:-$(brew --prefix llvm 2>/dev/null || echo /opt/homebrew/opt/llvm)}"
OPT="$PREFIX/bin/mlir-opt"
XLATE="$PREFIX/bin/mlir-translate"

hr() { printf '\n── %s %s\n' "$1" "$(printf '─%.0s' $(seq 1 $((46 - ${#1}))))"; }
have() { command -v "$1" >/dev/null 2>&1 && echo "  ✓ $1  ($(command -v "$1"))" || echo "  ✗ $1  MISSING"; }

hr "llvm / mlir"
if [ -x "$OPT" ]; then
  echo "  ✓ mlir-opt        $OPT"
  echo "    $("$OPT" --version 2>&1 | head -2 | tail -1 | sed 's/^ *//')"
else
  echo "  ✗ mlir-opt        not at $OPT"
  echo "    brew install llvm"
  exit 1
fi
[ -x "$XLATE" ] && echo "  ✓ mlir-translate  $XLATE" || echo "  ✗ mlir-translate  MISSING"

hr "spirv serialization (the emit leg)"
"$XLATE" --help 2>&1 | grep -iE '^\s+--(de)?serialize-spirv' | sed 's/^/  /' \
  || echo "  ✗ no spirv (de)serialization in this build"

HELP="$("$OPT" --help-list 2>&1)"

hr "conversion passes (the lowering leg)"
# These are the passes a block-level surface actually needs. Each is reported
# individually because the composed ConvertToSPIRVPass was demoted to test-only
# upstream (PR #124301) — the per-dialect ones are what remain maintained.
for p in convert-to-spirv convert-func-to-spirv convert-arith-to-spirv \
         convert-memref-to-spirv convert-vector-to-spirv convert-scf-to-spirv \
         convert-cf-to-spirv convert-math-to-spirv convert-gpu-to-spirv \
         gpu-kernel-outlining spirv-lower-abi-attrs spirv-update-vce \
         map-memref-spirv-storage-class spirv-attach-target; do
  # NB: use --help-list (--help omits the pass list) and avoid `grep -q`, whose
  # early exit SIGPIPEs mlir-opt and, under `set -o pipefail`, is reported as a
  # failed pipeline -- which made an earlier version of this script report every
  # pass as missing when all of them were present.
  if [ -n "$(HELP_CACHE_HIT=1; echo "$HELP" | grep -E "^\\s+--$p\\b")" ]; then
    echo "  ✓ --$p"
  else
    echo "  ✗ --$p"
  fi
done

hr "spirv-tools (validation)"
have spirv-val
have spirv-dis
have spirv-cross
[ -x "$(command -v spirv-val 2>/dev/null)" ] || echo "    brew install spirv-tools"

hr "spirv -> wgsl (the ingestion leg — the actual risk)"
have naga
have naga-cli
have cargo
have tint
echo
echo "  naga:  cargo install naga-cli --features spv-in,wgsl-out"
echo "  tint:  build Dawn with TINT_BUILD_SPV_READER=ON (heavy; naga first)"

hr "browser side"
have node
have python3

hr "verdict"
echo "  The critical unknown is the ingestion leg. MLIR emitting *a* SPIR-V"
echo "  binary proves nothing on its own — IREE has been doing that for four"
echo "  years and still files 'unknown SPIR-V builtin' bugs against its own"
echo "  output. The question is whether a WebGPU consumer accepts it."
