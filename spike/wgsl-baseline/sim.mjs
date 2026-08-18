// sim.mjs — a CPU simulation of matmul.wgsl, mirroring its index arithmetic
// statement for statement.
//
//   node sim.mjs          # small shape, fast
//   node sim.mjs --full   # the real 1024x768x512 the shader is specialised to
//
// Why this exists: the likeliest way the hand-written kernel is wrong is not the
// math, it is the STAGING INDEX ARITHMETIC — the flattening of a 64x16 tile
// across 256 invocations, and the ty*TM+m / tx*TN+n fragment mapping. Those are
// pure integer bugs and they do not need a GPU to find.
//
// Barriers are modelled as loop boundaries: every invocation completes the
// staging phase before any invocation begins the accumulate phase. That is the
// exact meaning of workgroupBarrier() for a race-free kernel, which is also the
// modelling choice tessera's CPU backend will make.
//
// The result must match oracle.js BIT-EXACTLY, not merely within tolerance:
// both accumulate in the same order with the same f32 roundings, and JS has no
// FMA contraction. Any difference at all is an index bug.

import { randomF32, matmulReluF32, allClose } from "./oracle.js";

const BM = 64, BN = 64, BK = 16;
const TM = 4, TN = 4;
const WGX = 16, WGY = 16;
const THREADS = WGX * WGY;   // 256

const fr = Math.fround;

function simulate(a, b, { M, N, K }) {
  const c = new Float32Array(M * N);
  const As = new Float32Array(BM * BK);
  const Bs = new Float32Array(BK * BN);
  const acc = new Float32Array(THREADS * TM * TN);

  const gridY = M / BM, gridX = N / BN;

  for (let blockRow = 0; blockRow < gridY; blockRow++) {
    for (let blockCol = 0; blockCol < gridX; blockCol++) {
      acc.fill(0);

      for (let kk = 0; kk < K; kk += BK) {

        // ---- phase 1: stage. All invocations, then a barrier.
        for (let tid = 0; tid < THREADS; tid++) {
          for (let i = tid; i < BM * BK; i += THREADS) {
            const r = (i / BK) | 0, cc = i % BK;
            As[i] = a[(blockRow * BM + r) * K + (kk + cc)];
          }
          for (let i = tid; i < BK * BN; i += THREADS) {
            const r = (i / BN) | 0, cc = i % BN;
            Bs[i] = b[(kk + r) * N + (blockCol * BN + cc)];
          }
        }

        // ---- phase 2: accumulate. All invocations, then a barrier.
        for (let tid = 0; tid < THREADS; tid++) {
          const tx = tid % WGX, ty = (tid / WGX) | 0;
          const base = tid * TM * TN;

          for (let k = 0; k < BK; k++) {
            const aFrag = [0, 0, 0, 0], bFrag = [0, 0, 0, 0];
            for (let m = 0; m < TM; m++) aFrag[m] = As[(ty * TM + m) * BK + k];
            for (let n = 0; n < TN; n++) bFrag[n] = Bs[k * BN + (tx * TN + n)];

            for (let m = 0; m < TM; m++) {
              for (let n = 0; n < TN; n++) {
                const idx = base + m * TN + n;
                acc[idx] = fr(acc[idx] + fr(aFrag[m] * bFrag[n]));
              }
            }
          }
        }
      }

      // ---- epilogue: relu + store
      for (let tid = 0; tid < THREADS; tid++) {
        const tx = tid % WGX, ty = (tid / WGX) | 0;
        const base = tid * TM * TN;
        for (let m = 0; m < TM; m++) {
          const row = blockRow * BM + ty * TM + m;
          for (let n = 0; n < TN; n++) {
            const col = blockCol * BN + tx * TN + n;
            const v = acc[base + m * TN + n];
            c[row * N + col] = v > 0 ? v : 0;
          }
        }
      }
    }
  }
  return c;
}

// ---------------------------------------------------------------------------

const full = process.argv.includes("--full");
const dims = full
  ? { M: 1024, N: 768, K: 512 }   // what matmul.wgsl is specialised to
  : { M: 128, N: 128, K: 64 };    // same tile, smaller grid

for (const [name, v] of Object.entries(dims)) {
  const block = name === "K" ? BK : name === "M" ? BM : BN;
  if (v % block !== 0) throw new Error(`${name}=${v} is not a multiple of ${block}`);
}

console.log(`dims        ${dims.M} x ${dims.N} x ${dims.K}`);
console.log(`tile        ${BM} x ${BN} x ${BK}   workgroup ${WGX}x${WGY}   fragment ${TM}x${TN}`);
console.log(`grid        ${dims.N / BN} x ${dims.M / BM} workgroups`);
console.log(`workgroup   ${(BM * BK + BK * BN) * 4} B of 16384 B floor`);
console.log("");

const a = randomF32(dims.M * dims.K, 0x1234);
const b = randomF32(dims.K * dims.N, 0x5678);

let t = performance.now();
const got = simulate(a, b, dims);
console.log(`simulate    ${(performance.now() - t).toFixed(0)} ms`);

t = performance.now();
const want = matmulReluF32(a, b, dims);
console.log(`oracle      ${(performance.now() - t).toFixed(0)} ms`);
console.log("");

// Bit-exact, not tolerance-based. See the header.
let mismatches = 0, firstIdx = -1;
for (let i = 0; i < got.length; i++) {
  if (!Object.is(got[i], want[i])) {
    mismatches++;
    if (firstIdx < 0) firstIdx = i;
  }
}

const cmp = allClose(got, want);
console.log(`maxAbsDiff  ${cmp.maxAbsDiff.toExponential(3)}`);
console.log(`maxRelDiff  ${cmp.maxRelDiff.toExponential(3)}`);
console.log("");

if (mismatches === 0) {
  console.log(`PASS  ${got.length} / ${got.length} elements bit-identical to the oracle.`);
  console.log(`      The staging and fragment index arithmetic in matmul.wgsl is correct.`);
} else {
  const r = Math.floor(firstIdx / dims.N), c = firstIdx % dims.N;
  console.log(`FAIL  ${mismatches} / ${got.length} elements differ.`);
  console.log(`      first at c[${r}][${c}]  sim ${got[firstIdx]}  oracle ${want[firstIdx]}`);
  console.log(`      Bit-exactness is expected here, so ANY difference is an index bug,`);
  console.log(`      not a rounding artefact.`);
  process.exitCode = 1;
}
