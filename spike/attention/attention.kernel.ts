// Scaled dot-product attention, one head. The whole point of the fourth family.
//
//   S[m,n] = sum_d Q[m,d] * K[n,d]        contract d, accumulate (m, n)
//   P[m,n] = softmax_n(S[m,:])
//   O[m,d] = sum_n P[m,n] * V[n,d]        contract n, accumulate (m, d)
//
// The two contractions accumulate over DIFFERENT axis sets, and that is what is
// left of G4. P comes out of the first laid out with n on the x lane — four of
// the sixty-four n values per invocation — and the second contracts along n, so
// it needs all of them. The fragment has to be redistributed.

import {
  axis, tiling, kernel, input, output, f32,
  zeros, mma, rowFill, rowMax, rowSum, subRow, divRow, expTile, negInf, zero,
} from "../../src/tessera";

// 32x32x16, not this project's usual 64x64x16. Attention has to stage the score
// fragment as well as q, k and v, and a bm x bn fragment at 64x64 is 16384 B on
// its own — the entire guaranteed floor. See docs/004 R12.
const T = tiling(f32, 32, 32, 16);

const M = axis("m", 1024, T.bm);   // queries
const N = axis("n",  768, T.bn);   // keys and values
const D = axis("d",   64, T.bk);   // head dimension

export const attention = kernel(
  {
    name: "attention_f32",
    tile: T,
    axes: [M, N, D],
    bindings: [
      input("q", [M, D], f32),
      input("k", [N, D], f32),      // stored [n, d]; S wants Kᵀ
      input("v", [N, D], f32),
      output("o", [M, D], f32),
    ],
  },
  ({ q, k, v, o, at, reduce }) => {
    let mx = rowFill(T.bm, f32, negInf);
    for (const n of reduce.n) {
      let s = zeros(T.bm, T.bn, f32);
      for (const d of reduce.d) {
        s = mma(q.tile(at.m, d), k.tileT(d, n), s);
      }
      mx = rowMax(s, mx);
    }
    let sm = rowFill(T.bm, f32, zero);
    for (const n of reduce.n) {
      let s = zeros(T.bm, T.bn, f32);
      for (const d of reduce.d) {
        s = mma(q.tile(at.m, d), k.tileT(d, n), s);
      }
      sm = rowSum(expTile(subRow(s, mx)), sm);
    }
    let acc = zeros(T.bm, T.bk, f32);   // the OUTPUT tile: bm x bd, and bd is T.bk
    for (const n of reduce.n) {
      let s = zeros(T.bm, T.bn, f32);
      for (const d of reduce.d) {
        s = mma(q.tile(at.m, d), k.tileT(d, n), s);
      }
      acc = mma(divRow(expTile(subRow(s, mx)), sm), v.tile(n, at.d), acc);
    }
    o.tile(at.m, at.d).store(acc);
  },
);
