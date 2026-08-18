// matmul.wgsl — tiled f32 GEMM with a fused ReLU epilogue.
//
//   C[M,N] = relu(A[M,K] @ B[K,N])
//
// This file is the TARGET. It is what tessera must eventually emit for:
//
//   const T = tiling(f32, 64, 64, 16);
//   const M = axis("m", 1024, T.bm);   // 1024 % 64 == 0 -> "exact", no masks
//   const N = axis("n",  768, T.bn);   //  768 % 64 == 0 -> "exact"
//   const K = axis("k",  512, T.bk);   //  512 % 16 == 0 -> "exact"
//
// Every extent is a compile-time literal, exactly as the surface promises — so
// this shader is specialised to one shape and carries no bounds checks at all.
// The ragged variant (1000 x 500 x 750) is a separate artifact; that is the
// point of the design, and week 9 of the plan is where it gets built.

const M : u32 = 1024u;
const N : u32 =  768u;
const K : u32 =  512u;

const BM : u32 = 64u;   // T.bm — block rows
const BN : u32 = 64u;   // T.bn — block cols
const BK : u32 = 16u;   // T.bk — reduction block

// Per-invocation register fragment. This is a constant fold from two literals:
//   (BM * BN) / (wgX * wgY) = (64 * 64) / (16 * 16) = 16 = TM * TN
// tessera computes it; here it is written out by hand so the arithmetic the
// compiler has to reproduce is visible.
const TM : u32 = 4u;
const TN : u32 = 4u;

@group(0) @binding(0) var<storage, read_write> a : array<f32>;   // M x K, row-major
@group(0) @binding(1) var<storage, read_write> b : array<f32>;   // K x N, row-major
@group(0) @binding(2) var<storage, read_write> c : array<f32>;   // M x N, row-major

// Workgroup staging. (BM*BK + BK*BN) * 4 B = (1024 + 1024) * 4 = 8192 B,
// exactly half of WebGPU's guaranteed maxComputeWorkgroupStorageSize floor of
// 16384 B. The 4096 B of headroom the plan reserves is what keeps the door open
// for double-buffering later without changing the tile.
var<workgroup> As : array<f32, 1024>;   // BM x BK
var<workgroup> Bs : array<f32, 1024>;   // BK x BN

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id)        wg  : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {

  let blockRow = wg.y;              // dispatch.y = M / BM = 16
  let blockCol = wg.x;              // dispatch.x = N / BN = 12

  let tx  = lid.x;                  // 0..15
  let ty  = lid.y;                  // 0..15
  let tid = ty * 16u + tx;          // 0..255, the linear invocation index

  // Function-scope `var` is zero-initialised by WGSL, so this is the accumulator
  // the surface writes as `let acc = zeros(T.bm, T.bn, f32)` — except that each
  // invocation owns only its own 4x4 slice of that 64x64 logical accumulator.
  var acc : array<f32, 16>;         // TM * TN

  for (var kk : u32 = 0u; kk < K; kk = kk + BK) {

    // ---- stage the A block: BM*BK = 1024 elements across 256 invocations.
    // Consecutive tid maps to consecutive column, so the global reads coalesce.
    for (var i : u32 = tid; i < BM * BK; i = i + 256u) {
      let r  = i / BK;              // 0..63
      let cc = i % BK;              // 0..15
      As[i] = a[(blockRow * BM + r) * K + (kk + cc)];
    }

    // ---- stage the B block: BK*BN = 1024 elements.
    for (var i : u32 = tid; i < BK * BN; i = i + 256u) {
      let r  = i / BN;              // 0..15
      let cc = i % BN;              // 0..63
      Bs[i] = b[(kk + r) * N + (blockCol * BN + cc)];
    }

    workgroupBarrier();

    // ---- accumulate this block. Outer product formulation: each invocation
    // reads TM values down a column of As and TN values across a row of Bs,
    // then does TM*TN = 16 FMAs. That is the whole reason for the register
    // fragment — 8 loads feed 16 multiply-adds.
    for (var k : u32 = 0u; k < BK; k = k + 1u) {

      var aFrag : array<f32, 4>;
      for (var m : u32 = 0u; m < TM; m = m + 1u) {
        aFrag[m] = As[(ty * TM + m) * BK + k];
      }

      var bFrag : array<f32, 4>;
      for (var n : u32 = 0u; n < TN; n = n + 1u) {
        bFrag[n] = Bs[k * BN + (tx * TN + n)];
      }

      for (var m : u32 = 0u; m < TM; m = m + 1u) {
        for (var n : u32 = 0u; n < TN; n = n + 1u) {
          acc[m * TN + n] = acc[m * TN + n] + aFrag[m] * bFrag[n];
        }
      }
    }

    // Guard the next iteration's overwrite of As/Bs against invocations still
    // reading them. Without this the kernel is racy and intermittently wrong.
    workgroupBarrier();
  }

  // ---- fused epilogue + store. The surface writes this as
  //   c.tile(at.m, at.n).store(relu(acc))
  // and it is one statement precisely because the store is edge-clipped by
  // construction on an "exact" axis.
  for (var m : u32 = 0u; m < TM; m = m + 1u) {
    let row = blockRow * BM + ty * TM + m;
    for (var n : u32 = 0u; n < TN; n = n + 1u) {
      let col = blockCol * BN + tx * TN + n;
      c[row * N + col] = max(acc[m * TN + n], 0.0);
    }
  }
}
