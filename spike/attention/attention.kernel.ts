// Scaled dot-product attention, one head.
//
//   S[m,n] = sum_d Q[m,d] * K[n,d]
//   P[m,n] = softmax_n(S[m,:])
//   O[m,d] = sum_n P[m,n] * V[n,d]
//
// The FOURTH kernel family, written to test the claim docs/004 ended on: that
// the derivation is general and the vocabulary is finite. Attention is the case
// that stresses something neither of the first three did — it is a COMPOSITION
// of two contractions (over d, then over n) with a row-wise normalisation
// between them, and it needs a register fragment and two row accumulators alive
// in the same body, which `parseBody` currently rejects by name.
//
// Written the way a person would write it, before finding out what is admitted.
// Whatever the compiler refuses here is the work list.

import {
  axis, raggedAxis, tiling, kernel, input, output, f32,
  zeros, rowFill, rowMax, rowSum, subRow, divRow, expTile, mma, negInf, zero,
} from "../src/tessera";

const T = tiling(f32, 64, 64, 16);

const M = axis("m", 1024, T.bm);        // queries
const N = raggedAxis("n", 750, T.bn);   // keys and values; 750 % 64 = 46
const D = axis("d", 64, T.bk);          // head dimension

export const attention = kernel(
  {
    name: "attention_f32",
    grid: [M],
    reduce: [N, D],
    bindings: [
      input("q", [M, D], f32),
      input("k", [N, D], f32),
      input("v", [N, D], f32),
      output("o", [M, D], f32),
    ],
  },
  ({ q, k, v, o, at, reduce }) => {
    // pass 1 — the row maximum of the scores
    let mx = rowFill(T.bm, f32, negInf);
    for (const n of reduce.n) {
      let s = zeros(T.bm, T.bn, f32);
      for (const d of reduce.d) {
        s = mma(q.tile(at.m, d), k.tile(n, d), s);
      }
      mx = rowMax(s, mx);
    }

    // pass 2 — the row sum of the exponentials
    let sm = rowFill(T.bm, f32, zero);
    for (const n of reduce.n) {
      let s = zeros(T.bm, T.bn, f32);
      for (const d of reduce.d) {
        s = mma(q.tile(at.m, d), k.tile(n, d), s);
      }
      sm = rowSum(expTile(subRow(s, mx)), sm);
    }

    // pass 3 — the weighted sum of V
    let acc = zeros(T.bm, T.bn, f32);
    for (const n of reduce.n) {
      let s = zeros(T.bm, T.bn, f32);
      for (const d of reduce.d) {
        s = mma(q.tile(at.m, d), k.tile(n, d), s);
      }
      acc = mma(divRow(expTile(subRow(s, mx)), sm), v.tile(n, at.d), acc);
    }
    o.tile(at.m, at.d).store(acc);
  },
);

// ─────────────────────────────────────────────────────────────────────────────
// WHAT THE ADMISSION GATE SAID
//
// This does not compile, on purpose — it was written the way a person would
// write it, before finding out what is admitted. `tsc` enumerated the gap
// precisely, which is cheaper than reasoning about it. Four distinct things,
// three of them vocabulary and one of them structural:
//
//   G1  transpose.  `mma` wants its second operand as [bk, bn]; attention needs
//       Kᵀ, and `k.tile(n, d)` is [bn, bk]. The type system refuses it, which is
//       the surface working as designed — a transposed axis IS a type error
//       here. So attention needs the transpose to be SAID, not inferred.
//
//   G2  row reductions over a computed fragment. `rowMax(s, mx)` where `s` is a
//       Frag; `rowMax` takes a Tile. The first three families only ever reduced
//       over something loaded from memory.
//
//   G3  contraction and row ops over a computed fragment. `subRow(s, mx)` and
//       `mma(divRow(...), v.tile(...), acc)` both want a Tile where attention
//       has a Frag. Same root as G2: a fragment is not a first-class operand.
//
//   G4  `at.d` does not exist, and this is the structural one. `at` is built
//       from spec.grid, and D is in spec.reduce. But attention needs D
//       CONTRACTED in S = Q·Kᵀ and FREE in O = P·V — the same axis, on both
//       sides of the split, in one kernel. `n` is the mirror image: free in S,
//       contracted in O.
//
// G4 is the one worth having found. `planContraction(accumulate, contract, ...)`
// is already parameterised per contraction — it takes the split as an argument
// and assumes nothing global. What assumes a global split is the SURFACE:
// `spec.grid` and `spec.reduce` partition the axes once, for the whole kernel,
// and `at.*` is built from grid alone.
//
// So the prediction docs/004 R11 made — derivation general, vocabulary finite —
// survives contact here in a specific form: the machinery that derives a
// schedule looks able to compose, and the thing that cannot express attention is
// the surface's one-time partition of the axes.
//
// Not yet verified: that two planContraction calls actually compose in the
// emitter. That needs the surface change first, so it stays a prediction.
