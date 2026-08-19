// softmax_n(A · B) with N wider than one block — the minimal NESTED contraction.
//
//   for n:                        outer: reduce over the n blocks
//     s = zeros(bm, bn)
//     for k: s = mma(a,b,s)       inner: contract over k, accumulating (m, n)
//     mx = rowMax(s, mx)          fold this block's scores into the running max
//
// examples/fused-softmax.kernel.ts sidesteps this by making N exactly one block,
// so the whole score fragment exists at once and the reduction needs no outer
// loop. Widen N and the scores no longer fit: they have to be recomputed per
// block, which puts one contraction INSIDE another's loop.
//
// That nesting is the handoff G4 is about. The inner contraction's fragment is
// consumed in the same outer iteration that produced it — a temporary, not
// something carried across — which is the easier half. Attention's harder half is
// that its two contractions accumulate over DIFFERENT axis sets, (m,n) then (m,d),
// so their lane assignments disagree and the fragment must be redistributed.
//
// This probe has one accumulate set throughout, so it isolates the nesting.

import {
  axis, tiling, kernel, input, output, f32,
  zeros, mma, rowFill, rowMax, rowSum, subRow, divRow, expTile, negInf, zero,
} from "../../src/tessera";

const T = tiling(f32, 64, 64, 16);

const M = axis("m", 1024, T.bm);
const N = axis("n",  768, T.bn);   // twelve blocks, not one
const K = axis("k",  512, T.bk);

export const nested = kernel(
  {
    name: "nested_softmax_f32",
    tile: T,
    axes: [M, N, K],
    bindings: [
      input("a", [M, K], f32),
      input("b", [K, N], f32),
      output("y", [M, N], f32),
    ],
  },
  ({ a, b, y, at, reduce }) => {
    let mx = rowFill(T.bm, f32, negInf);
    for (const n of reduce.n) {
      let s = zeros(T.bm, T.bn, f32);
      for (const k of reduce.k) {
        s = mma(a.tile(at.m, k), b.tile(k, n), s);
      }
      mx = rowMax(s, mx);
    }
    let sm = rowFill(T.bm, f32, zero);
    for (const n of reduce.n) {
      let s = zeros(T.bm, T.bn, f32);
      for (const k of reduce.k) {
        s = mma(a.tile(at.m, k), b.tile(k, n), s);
      }
      sm = rowSum(expTile(subRow(s, mx)), sm);
    }
    for (const n of reduce.n) {
      let s = zeros(T.bm, T.bn, f32);
      for (const k of reduce.k) {
        s = mma(a.tile(at.m, k), b.tile(k, n), s);
      }
      y.tile(at.m, n).store(divRow(expTile(subRow(s, mx)), sm));
    }
  },
);
