// y = softmax_n(A · B), one workgroup per row block.
//
// The G2/G3 probe (docs/004 R12): row reductions and row operations over a
// COMPUTED fragment rather than over a tile loaded from memory. The first three
// families only ever reduced over something read from a buffer.
//
// N is deliberately one block wide, so `softmax_n` is well defined inside a
// single workgroup — the fragment covers the whole n range. That keeps this a
// test of the vocabulary (G2, G3) and not of G4, which is contraction
// composition and a separate problem.
//
// What it took: the row ops grew Frag overloads (the type half of G2/G3), the
// parser learned a fragment reduction as a derived value and a fragment-valued
// intermediate, and the emitter grew emitReduceFrag — a cross-lane fold, because
// `accumulate = (m, n)` puts n on the x lane and one row therefore lives in all
// sixteen of them.

import {
  axis, tiling, kernel, input, output, f32,
  zeros, mma, rowFill, rowMax, rowSum, subRow, divRow, expTile, negInf, zero,
} from "../src/tessera";

const T = tiling(f32, 64, 64, 16);

const M = axis("m", 1024, T.bm);
const N = axis("n",   64, T.bn);   // exactly one block: the row fits a workgroup
const K = axis("k",  512, T.bk);

export const fusedSoftmax = kernel(
  {
    name: "fused_softmax_f32",
    tile: T,
    axes: [M, N, K],
    bindings: [
      input("a", [M, K], f32),
      input("b", [K, N], f32),
      output("y", [M, N], f32),
    ],
  },
  ({ a, b, y, at, reduce }) => {
    let s = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      s = mma(a.tile(at.m, k), b.tile(k, at.n), s);
    }
    // Everything below reduces or scales a FRAGMENT, not a loaded tile.
    const mx = rowMax(s, rowFill(T.bm, f32, negInf));
    const e = expTile(subRow(s, mx));
    const sm = rowSum(e, rowFill(T.bm, f32, zero));
    y.tile(at.m, at.n).store(divRow(e, sm));
  },
);
