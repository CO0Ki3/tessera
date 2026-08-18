#!/usr/bin/env bash
# run-l0.sh — the week-0 conversion + ingestion test, reproducible.
#
#   ./run-l0.sh
#
# Answers two questions that fail for different reasons and must not be tested
# together:
#   Q1 ingestion  — does MLIR-emitted SPIR-V survive spirv-val -> naga -> WGSL?
#   Q2 conversion — does mlir-opt lower memref/vector/gpu -> spirv at all?

set -uo pipefail
cd "$(dirname "$0")"

PREFIX="${LLVM_PREFIX:-$(brew --prefix llvm 2>/dev/null || echo /opt/homebrew/opt/llvm)}"
OPT="$PREFIX/bin/mlir-opt"
XLATE="$PREFIX/bin/mlir-translate"
ATTACH="--spirv-attach-target=ver=v1.3 caps=Shader exts=SPV_KHR_storage_buffer_storage_class"
OUT=l0b
mkdir -p "$OUT"

for t in "$OPT" "$XLATE" "$(command -v spirv-val)" "$(command -v naga)"; do
  [ -x "$t" ] || { echo "missing tool: $t — run ./probe.sh"; exit 1; }
done

# Lower, slice out the spirv.module, serialize, validate, ingest.
#
# --no-implicit-module is required: mlir-translate otherwise wraps the file in a
# builtin.module and --serialize-spirv rejects that ("expected a 'spirv.module'
# op, got 'builtin.module'").
#
# The python slice is required because --convert-gpu-to-spirv deliberately LEAVES
# the source gpu.module in place for a host pipeline to consume, and
# mlir-translate registers only the builtin and spirv dialects. Neither
# --symbol-dce nor --gpu-module-to-binary removes it.
check() {
  local stem="$1" label="$2"
  printf '\n── %s\n' "$label"

  "$OPT" "$ATTACH" --convert-gpu-to-spirv --spirv-lower-abi-attrs --spirv-update-vce \
      "$OUT/$stem.mlir" 2>/dev/null \
    | python3 ./extract-spirv-module.py > "$OUT/$stem-spirv.mlir" 2>/dev/null

  if [ ! -s "$OUT/$stem-spirv.mlir" ]; then printf '  lower      ✗ no spirv.module produced\n'; return 1; fi
  printf '  lower      ✓\n'

  "$XLATE" --no-implicit-module --serialize-spirv "$OUT/$stem-spirv.mlir" -o "$OUT/$stem.spv" 2>/dev/null
  if [ ! -s "$OUT/$stem.spv" ]; then printf '  serialize  ✗\n'; return 1; fi
  printf '  serialize  ✓ %s bytes\n' "$(wc -c < "$OUT/$stem.spv" | tr -d ' ')"

  local valout
  valout=$(spirv-val --target-env vulkan1.1 "$OUT/$stem.spv" 2>&1)
  if [ -z "$valout" ]; then printf '  spirv-val  ✓ VALID\n'
  else printf '  spirv-val  ✗ %s\n' "$(echo "$valout" | head -1)"; fi

  local nagaout
  nagaout=$(naga "$OUT/$stem.spv" "$OUT/$stem.wgsl" 2>&1)
  if [ -z "$nagaout" ] && [ -s "$OUT/$stem.wgsl" ]; then printf '  naga       ✓ ACCEPTED -> %s\n' "$OUT/$stem.wgsl"
  else printf '  naga       ✗ %s\n' "$(echo "$nagaout" | head -1)"; fi
}

# ---------------------------------------------------------------------------
# C — vector.load/store over a SCALAR-element memref.
# The natural way to write a wide load, and it does not survive.
# ---------------------------------------------------------------------------
cat > "$OUT/c-gpu.mlir" <<'EOF'
module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @staged(%arg: memref<64xf32, #spirv.storage_class<StorageBuffer>>) kernel
        attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      %smem = memref.alloc() : memref<16xf32, #spirv.storage_class<Workgroup>>
      %c0 = arith.constant 0 : index
      %v = vector.load %arg[%c0] : memref<64xf32, #spirv.storage_class<StorageBuffer>>, vector<4xf32>
      vector.store %v, %smem[%c0] : memref<16xf32, #spirv.storage_class<Workgroup>>, vector<4xf32>
      gpu.barrier
      gpu.return
    }
  }
}
EOF

# ---------------------------------------------------------------------------
# D — scalar memref.load/store. The `--no-vector` fallback from the plan.
# ---------------------------------------------------------------------------
cat > "$OUT/d-scalar.mlir" <<'EOF'
module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @staged(%arg: memref<64xf32, #spirv.storage_class<StorageBuffer>>) kernel
        attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      %smem = memref.alloc() : memref<16xf32, #spirv.storage_class<Workgroup>>
      %c0 = arith.constant 0 : index
      %v = memref.load %arg[%c0] : memref<64xf32, #spirv.storage_class<StorageBuffer>>
      memref.store %v, %smem[%c0] : memref<16xf32, #spirv.storage_class<Workgroup>>
      gpu.barrier
      gpu.return
    }
  }
}
EOF

# ---------------------------------------------------------------------------
# E — vector as the memref ELEMENT TYPE. OpAccessChain lands directly on a vec4,
# so no pointer bitcast is needed and Logical addressing is satisfied.
# ---------------------------------------------------------------------------
cat > "$OUT/e-vecmemref.mlir" <<'EOF'
module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @staged(%arg: memref<16xvector<4xf32>, #spirv.storage_class<StorageBuffer>>) kernel
        attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      %smem = memref.alloc() : memref<4xvector<4xf32>, #spirv.storage_class<Workgroup>>
      %c0 = arith.constant 0 : index
      %v = memref.load %arg[%c0] : memref<16xvector<4xf32>, #spirv.storage_class<StorageBuffer>>
      %w = arith.mulf %v, %v : vector<4xf32>
      memref.store %w, %smem[%c0] : memref<4xvector<4xf32>, #spirv.storage_class<Workgroup>>
      gpu.barrier
      gpu.return
    }
  }
}
EOF

echo "MLIR -> SPIR-V -> WGSL, three ways of expressing the same staged load"
check c-gpu      "C  vector.load over a scalar-element memref"
check d-scalar   "D  scalar memref.load                        (the --no-vector fallback)"
check e-vecmemref "E  memref whose ELEMENT is vector<4xf32>     (the legal vectorised path)"

cat <<'EOF'

── conclusion
  C fails and D/E pass. The failure is not MLIR emitting bad SPIR-V in general:
  --convert-vector-to-spirv lowers a wide load over a scalar memref to a
  spirv.Bitcast on a POINTER (ptr<f32> -> ptr<vector<4xf32>>), which the Logical
  addressing model forbids -- pointers must be derived from OpAccessChain.

  So the codegen rule for tessera is: model vectors as the memref ELEMENT TYPE,
  never as a wider load over a scalar memref. E proves that path is legal all the
  way to WGSL, which means vectorisation is available and the scalar fallback is
  not the only option.
EOF
