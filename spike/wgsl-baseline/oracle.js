// oracle.js — the CPU reference, plus the comparison machinery.
//
// This module is deliberately environment-free (no DOM, no WebGPU) so the exact
// same code runs in the browser harness and under plain node. In tessera proper
// this becomes `kernel.reference()`, executing the SAME kernel body on a CPU
// backend rather than a second hand-written implementation that can drift.

export const DIMS = { M: 1024, N: 768, K: 512 };

// ---------------------------------------------------------------------------
// Seeded PRNG. Browser and node must generate byte-identical inputs, so
// Math.random() is unusable here.
// ---------------------------------------------------------------------------
export function mulberry32(seed) {
  let t = seed >>> 0;
  return function () {
    t = (t + 0x6d2b79f5) >>> 0;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

/** Uniform in [-1, 1). Float32Array storage means every element is already an
 *  exact f32 value, so no rounding is needed on the way in. */
export function randomF32(n, seed) {
  const rnd = mulberry32(seed);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = rnd() * 2 - 1;
  return out;
}

// ---------------------------------------------------------------------------
// The reference kernel.
//
// Two properties make this a real oracle rather than an approximation:
//
//   1. f32 SEMANTICS. A JS `number` is always float64. Reading out of a
//      Float32Array widens exactly, but every arithmetic result must be pushed
//      back down with Math.fround or the reference silently computes in higher
//      precision than the GPU and the diff becomes meaningless.
//
//   2. ACCUMULATION ORDER. The GPU walks kk = 0,16,...,496 and k = 0..15 inside
//      each block, i.e. global k strictly ascending 0..K-1. This loop uses the
//      same order, so summation-order differences are eliminated as a source of
//      divergence — which is what makes a tight tolerance meaningful.
//
// FMA CONTRACTION is modelled explicitly rather than treated as noise. WGSL
// permits an implementation to fuse `acc + a*b` into a single fused multiply-add
// with one rounding instead of two, and there is no portable way to forbid it.
// Rather than absorbing that into a fudged tolerance, `contract: true` models it
// exactly:
//
//   the product of two f32 values is EXACT in f64 — two 24-bit significands make
//   at most 48 significant bits, and f64 carries 53 — so simply omitting the
//   inner fround computes `round_f32(acc + a*b)` with a single rounding, which
//   is precisely IEEE fma semantics.
//
// The one residual imprecision is double rounding: `acc + a*b` rounds to f64
// first and to f32 second, where a true fma rounds once. That diverges with
// probability ~2^-29 per operation, so over a few hundred million operations a
// small handful of elements may still differ by one ulp. That is a known,
// bounded artefact of the emulation, not of the GPU.
// ---------------------------------------------------------------------------
export function matmulReluF32(a, b, { M, N, K } = DIMS, { contract = false } = {}) {
  const c = new Float32Array(M * N);
  const fround = Math.fround;

  for (let i = 0; i < M; i++) {
    const aRow = i * K;
    const cRow = i * N;
    for (let j = 0; j < N; j++) {
      let acc = 0;
      if (contract) {
        for (let k = 0; k < K; k++) {
          acc = fround(acc + a[aRow + k] * b[k * N + j]);        // one rounding
        }
      } else {
        for (let k = 0; k < K; k++) {
          acc = fround(acc + fround(a[aRow + k] * b[k * N + j])); // two roundings
        }
      }
      c[cRow + j] = acc > 0 ? acc : 0;   // relu
    }
  }
  return c;
}

// ---------------------------------------------------------------------------
// The error scale for a dot product.
//
// This is the correction to the harness's original mistake. A tolerance of the
// form `atol + rtol * |result|` is structurally the wrong instrument for a
// reduction: rounding error accumulates against the magnitude of the PARTIAL
// SUMS, not of the final result. When cancellation drives a result near zero the
// error does not shrink with it, so near-zero outputs fail a relative test no
// matter how correct the kernel is. Roughly half of a ReLU's outputs sit at or
// near zero, which is why this bit us here specifically.
//
// The standard bound for a length-K dot product in floating point is
//
//     |computed - exact|  <=  gamma_K * SUM |a_i * b_i|,     gamma_K = K*u/(1 - K*u)
//
// with u = 2^-24 the f32 unit roundoff. So the correct per-element scale is the
// absolute sum, which this computes. Comparing observed error against this bound
// turns "loosen it until it passes" into a falsifiable check: we are either
// inside the bound or the kernel is wrong.
// ---------------------------------------------------------------------------
export const F32_UNIT_ROUNDOFF = Math.pow(2, -24);

export function gammaK(K) {
  const ku = K * F32_UNIT_ROUNDOFF;
  return ku / (1 - ku);
}

/** Per-output-element SUM |a_ik * b_kj| — the scale the error lives on. */
export function dotAbsSum(a, b, { M, N, K } = DIMS) {
  const s = new Float32Array(M * N);
  const fround = Math.fround;
  for (let i = 0; i < M; i++) {
    const aRow = i * K, sRow = i * N;
    for (let j = 0; j < N; j++) {
      let acc = 0;
      for (let k = 0; k < K; k++) {
        acc = fround(acc + Math.abs(fround(a[aRow + k] * b[k * N + j])));
      }
      s[sRow + j] = acc;
    }
  }
  return s;
}

/**
 * Check every element against its own theoretical bound.
 * Reports the worst ratio observed/bound — a value < 1 means every element is
 * within what floating point permits, and how far below 1 says how much slack
 * the kernel is actually using.
 */
export function withinErrorBound(got, want, absSum, K) {
  const g = gammaK(K);
  let violations = 0, worstRatio = 0, worstIndex = -1, worstDiff = 0, worstBound = 0;

  for (let i = 0; i < got.length; i++) {
    const diff = Math.abs(got[i] - want[i]);
    // Guard the degenerate all-zero row; the bound is 0 there and so is the diff.
    const bound = g * absSum[i];
    const ratio = bound > 0 ? diff / bound : (diff > 0 ? Infinity : 0);
    if (ratio > worstRatio) {
      worstRatio = ratio; worstIndex = i; worstDiff = diff; worstBound = bound;
    }
    if (ratio > 1) violations++;
  }

  return { ok: violations === 0, violations, worstRatio, worstIndex, worstDiff, worstBound, gammaK: g };
}

/** Count of elements that are bit-for-bit identical. */
export function bitExactCount(x, y) {
  let n = 0;
  for (let i = 0; i < x.length; i++) if (Object.is(x[i], y[i])) n++;
  return n;
}

// ---------------------------------------------------------------------------
// Comparison. Reports the worst offender rather than just a boolean, because
// "which element, by how much" is the only useful thing when it fails.
// ---------------------------------------------------------------------------
export function allClose(got, want, { rtol = 1e-5, atol = 1e-6 } = {}) {
  let failures = 0;
  let maxAbsDiff = 0;
  let maxRelDiff = 0;
  let worstIndex = -1;
  let worstGot = 0;
  let worstWant = 0;

  for (let i = 0; i < got.length; i++) {
    const g = got[i];
    const w = want[i];
    const diff = Math.abs(g - w);
    const rel = Math.abs(w) > 0 ? diff / Math.abs(w) : diff;

    if (rel > maxRelDiff) maxRelDiff = rel;
    if (diff > maxAbsDiff) {
      maxAbsDiff = diff;
      worstIndex = i;
      worstGot = g;
      worstWant = w;
    }
    if (diff > atol + rtol * Math.abs(w)) failures++;
  }

  return {
    ok: failures === 0,
    failures,
    total: got.length,
    maxAbsDiff,
    maxRelDiff,
    worstIndex,
    worstGot,
    worstWant,
    rtol,
    atol,
  };
}

/** Cheap structural check: catches an all-zeros or all-NaN result before the
 *  tolerance report makes it look like a subtle numerical problem. */
export function summarize(x) {
  let zeros = 0, nans = 0, min = Infinity, max = -Infinity, sum = 0;
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    if (Number.isNaN(v)) { nans++; continue; }
    if (v === 0) zeros++;
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
  }
  return { zeros, nans, min, max, mean: sum / x.length, length: x.length };
}
