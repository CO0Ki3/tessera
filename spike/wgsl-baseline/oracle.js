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

// ---------------------------------------------------------------------------
// Row softmax, mirroring the emitted kernel's traversal exactly.
//
//   y[m,n] = exp(x[m,n] - max_n x[m,:]) / sum_n exp(x[m,:] - max)
//
// The order matters and is not incidental. The kernel gives each lane `tx` the
// columns `nn + tx*TN + n`, accumulates within a lane over ascending nn then
// ascending n, and only then combines the WGX lane partials in ascending order.
// A reference that summed the row left to right would differ in the last bits
// for a reason that has nothing to do with correctness.
//
// UNLIKE the matmul, this cannot be bit-exact. WGSL does not require `exp` to be
// correctly rounded — it permits a few ULP — so the GPU's exp and JS's
// `Math.exp` are allowed to disagree in the last place, and that disagreement
// then propagates through the sum and the divide. Bit-exactness is the right bar
// for +, * and fma; for a transcendental it is the wrong question.
// ---------------------------------------------------------------------------

/** What the emitter writes for `negInf`: the largest finite f32 magnitude. */
export const NEG_INF_F32 = Math.fround(-3.4028235e38);

export function softmaxF32(x, { M, N }, geom = {}) {
  const { BM = 64, BN = 64, WGX = 16, WGY = 16, TM = 4, TN = 4 } = geom;
  const fr = Math.fround;
  const y = new Float32Array(M * N);

  for (let blockRow = 0; blockRow * BM < M; blockRow++) {
    for (let ty = 0; ty < WGY; ty++) {
      for (let m = 0; m < TM; m++) {
        const row = blockRow * BM + ty * TM + m;
        if (row >= M) continue;
        const base = row * N;

        // pass 1 — per-lane max over that lane's columns, then across lanes
        const laneMax = new Float32Array(WGX).fill(NEG_INF_F32);
        for (let nn = 0; nn < N; nn += BN) {
          for (let tx = 0; tx < WGX; tx++) {
            for (let n = 0; n < TN; n++) {
              const col = nn + tx * TN + n;
              const v = col < N ? x[base + col] : NEG_INF_F32;
              if (v > laneMax[tx]) laneMax[tx] = v;
            }
          }
        }
        let mx = NEG_INF_F32;
        for (let j = 0; j < WGX; j++) if (laneMax[j] > mx) mx = laneMax[j];

        // pass 2 — per-lane sum of exp(x - max), then across lanes
        const laneSum = new Float32Array(WGX);
        for (let nn = 0; nn < N; nn += BN) {
          for (let tx = 0; tx < WGX; tx++) {
            for (let n = 0; n < TN; n++) {
              const col = nn + tx * TN + n;
              const v = col < N ? x[base + col] : NEG_INF_F32;
              laneSum[tx] = fr(laneSum[tx] + fr(Math.exp(fr(v - mx))));
            }
          }
        }
        let sm = 0;
        for (let j = 0; j < WGX; j++) sm = fr(sm + laneSum[j]);

        // pass 3 — normalise the in-range columns
        for (let nn = 0; nn < N; nn += BN) {
          for (let tx = 0; tx < WGX; tx++) {
            for (let n = 0; n < TN; n++) {
              const col = nn + tx * TN + n;
              if (col >= N) continue;
              y[base + col] = fr(fr(Math.exp(fr(x[base + col] - mx))) / sm);
            }
          }
        }
      }
    }
  }
  return y;
}

/** Row sums of the output. Every row of a correct softmax sums to 1. */
export function rowSums(y, { M, N }) {
  const out = new Float64Array(M);
  for (let i = 0; i < M; i++) {
    let s = 0;
    for (let j = 0; j < N; j++) s += y[i * N + j];
    out[i] = s;
  }
  return out;
}

/** Distance in ULP, the only scale on which a transcendental's error means anything. */
export function maxUlpDiff(got, want) {
  const gb = new Int32Array(new Float32Array(got).buffer);
  const wb = new Int32Array(new Float32Array(want).buffer);
  let worst = 0, at = -1;
  const ord = (b) => (b < 0 ? 0x80000000 - b : b);   // monotone ordering of f32 bits
  for (let i = 0; i < gb.length; i++) {
    const d = Math.abs(ord(gb[i]) - ord(wb[i]));
    if (d > worst) { worst = d; at = i; }
  }
  return { ulp: worst, index: at };
}

// ---------------------------------------------------------------------------
// Layer normalisation, mirroring the emitted kernel's traversal exactly.
//
//   y[m,n] = (x[m,n] - mean) * rstd,   rstd = 1 / sqrt(E[x^2] - mean^2 + eps)
//
// Same lane geometry as softmax: lane tx owns columns nn + tx*TN + n, accumulates
// within a lane over ascending nn then ascending n, and the WGX lane partials are
// combined in order. Both moments are carried through ONE pass, which is what
// makes this kernel structurally different from softmax and why it was chosen.
//
// The element count is the AXIS EXTENT, not the padded block count — masked lanes
// contribute zero to both moments, so they must not be counted in the mean. That
// is the whole reason `.pad(zero)` is right here and `.pad(negInf)` was right for
// softmax's max, on the same axis with the same raggedness.
//
// Like softmax, this cannot be bit-exact: WGSL's inverseSqrt carries an accuracy
// allowance just as exp does.
// ---------------------------------------------------------------------------
export function layernormF32(x, { M, N }, { eps = 1e-5, ...geom } = {}) {
  const { BM = 64, BN = 64, WGX = 16, WGY = 16, TM = 4, TN = 4 } = geom;
  const fr = Math.fround;
  const y = new Float32Array(M * N);

  for (let blockRow = 0; blockRow * BM < M; blockRow++) {
    for (let ty = 0; ty < WGY; ty++) {
      for (let m = 0; m < TM; m++) {
        const row = blockRow * BM + ty * TM + m;
        if (row >= M) continue;
        const base = row * N;

        const laneS = new Float32Array(WGX);
        const laneQ = new Float32Array(WGX);
        for (let nn = 0; nn < N; nn += BN) {
          for (let tx = 0; tx < WGX; tx++) {
            for (let n = 0; n < TN; n++) {
              const col = nn + tx * TN + n;
              const v = col < N ? x[base + col] : 0;
              laneS[tx] = fr(laneS[tx] + v);
              laneQ[tx] = fr(laneQ[tx] + fr(v * v));
            }
          }
        }
        let s = 0, q = 0;
        for (let j = 0; j < WGX; j++) s = fr(s + laneS[j]);
        for (let j = 0; j < WGX; j++) q = fr(q + laneQ[j]);

        const mu = fr(s / N);
        const inv = fr(1 / Math.sqrt(fr(fr(fr(q / N) - fr(mu * mu)) + eps)));

        for (let nn = 0; nn < N; nn += BN) {
          for (let tx = 0; tx < WGX; tx++) {
            for (let n = 0; n < TN; n++) {
              const col = nn + tx * TN + n;
              if (col >= N) continue;
              y[base + col] = fr(fr(x[base + col] - mu) * inv);
            }
          }
        }
      }
    }
  }
  return y;
}

/** Per-row mean and variance of the output. A correct layernorm gives 0 and 1. */
export function rowMoments(y, { M, N }) {
  const mean = new Float64Array(M), varr = new Float64Array(M);
  for (let i = 0; i < M; i++) {
    let s = 0;
    for (let j = 0; j < N; j++) s += y[i * N + j];
    const mu = s / N;
    let v = 0;
    for (let j = 0; j < N; j++) { const d = y[i * N + j] - mu; v += d * d; }
    mean[i] = mu; varr[i] = v / N;
  }
  return { mean, varr };
}

/**
 * The error scale for a computation whose result can cancel to near zero.
 *
 * ULP on the RESULT is the right measure for a transcendental's output — softmax's
 * exp is inexact and its outputs are all comfortably away from zero. It is the
 * wrong measure for a difference. layernorm computes `(x - mu) * inv`, and where
 * `x` happens to sit near `mu` the result is tiny while its error is inherited
 * from operands of order one. Judging that by ULP on the result reports thousands
 * of them for an absolute error of 1.6e-9.
 *
 * This is the same lesson as the matmul's 912 spurious failures, in a new guise:
 * there the error scaled with the partial sums rather than the sum, here it scales
 * with the operands rather than the difference. Choosing the scale correctly is
 * not the same as loosening the tolerance until it passes.
 *
 * So: absolute error against the magnitude the computation actually worked on.
 */
export function absErrorAgainstScale(got, want) {
  let maxAbs = 0, scale = 0, at = -1;
  for (let i = 0; i < got.length; i++) {
    const d = Math.abs(got[i] - want[i]);
    if (d > maxAbs) { maxAbs = d; at = i; }
    const m = Math.abs(want[i]);
    if (m > scale) scale = m;
  }
  return { maxAbs, scale, ratio: scale > 0 ? maxAbs / scale : maxAbs, at };
}

/**
 * Transpose a row-major [rows, cols] matrix into [cols, rows].
 *
 * Used to feed the same logical B to two kernels that store it differently:
 * `matmul_relu_f32` declares b as [K, N] and `matmul_bt_f32` declares it [N, K]
 * and reads it with `.tileT()`. Given B and Bᵀ, the two must agree bit for bit —
 * which is the whole claim `.tileT()` makes, checked on real data rather than
 * argued from the generated address.
 */
export function transposeF32(src, rows, cols) {
  const out = new Float32Array(src.length);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) out[c * rows + r] = src[r * cols + c];
  }
  return out;
}

/**
 * softmax_n(A · B) in f32, the reference for examples/fused-softmax.kernel.ts.
 *
 * It deliberately does not FUSE: the matmul completes into a full row before the
 * softmax starts, where the kernel folds across lanes as it goes.
 *
 * It does CONTRACT, and the first version of this did not — which reported the
 * kernel as failing at 189 ULP. The arithmetic is the whole explanation. This
 * adapter fuses `acc + a*b` into one rounding, which the plain matmul already
 * establishes (786432/786432 bit-exact against the FMA reference). Two roundings
 * per step instead leaves the scores off by ~1.14e-5, and `exp` turns an absolute
 * score error into a relative one, so 1.14e-5 / 2^-24 ≈ 190 ULP comes out the
 * other side. Predicted 192, measured 189.
 *
 * Same class as the 912 spurious matmul failures in docs/002 §2: a reference that
 * does not match the machine's arithmetic, reported as a kernel bug. `a * b`
 * unrounded inside `fround` is the one-rounding form, the same trick
 * `matmulReluF32({contract: true})` uses.
 *
 * `dims.N` is one block wide by construction, so a row fits one workgroup and the
 * softmax is well defined inside it.
 */
export function fusedSoftmaxF32(a, b, { M, N, K }) {
  const out = new Float32Array(M * N);
  const row = new Float32Array(N);
  // The largest magnitude reaching `exp`. It is returned because it IS the error
  // scale: a GPU computes exp(x) as exp2(x * log2e), and rounding that product to
  // f32 costs |x| * 2^-24 of relative error in the result — one ULP of output per
  // unit of input. Measured against an exact exp: |x| = 20 gives ~1 ULP, 47 gives
  // ~22, 60 gives ~28.
  let maxArg = 0;
  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      let acc = 0;
      // One rounding per step: the multiply stays in f64 and only the sum rounds.
      for (let k = 0; k < K; k++) acc = Math.fround(acc + a[m * K + k] * b[k * N + n]);
      row[n] = acc;
    }
    let mx = -Infinity;
    for (let n = 0; n < N; n++) mx = Math.max(mx, row[n]);
    let sum = 0;
    for (let n = 0; n < N; n++) {
      const arg = Math.fround(row[n] - mx);
      if (-arg > maxArg) maxArg = -arg;
      row[n] = Math.fround(Math.exp(arg));
      sum = Math.fround(sum + row[n]);
    }
    for (let n = 0; n < N; n++) out[m * N + n] = Math.fround(row[n] / sum);
  }
  return { y: out, maxArg };
}
