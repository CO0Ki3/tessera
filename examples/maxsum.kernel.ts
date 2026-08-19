// Two identity elements in one kernel.
//
//   y[m,n] = (p[m,n] - max_n p[m,:]) / sum_n q[m,:]
//
// The case the old error message named — "a fused max-and-sum pass wants both" —
// and which the front end used to refuse, because the IR carried one identity for
// the whole kernel.
//
// The two are not interchangeable and neither is a default. A masked lane of `p`
// must read negative infinity, because it feeds a MAX and a zero there would win
// over a row that is entirely negative. A masked lane of `q` must read zero,
// because it feeds a SUM and negative infinity there would poison the total.
// Swap them and the kernel is wrong only in the ragged tail, and only for some
// data — which is the bug class this surface exists to make unsayable.
//
// n is ragged, so both masks are live.

import {
  axis, raggedAxis, tiling, kernel, input, output, f32,
  rowFill, rowMax, rowSum, subRow, divRow, negInf, zero,
} from "../src/tessera";

const T = tiling(f32, 64, 64, 16);

const M = axis("m", 1024, T.bm);
const N = raggedAxis("n", 750, T.bn);   // 750 % 64 = 46

export const maxsum = kernel(
  {
    name: "maxsum_f32",
    axes: [M, N],
    bindings: [
      input("p", [M, N], f32),
      input("q", [M, N], f32),
      output("y", [M, N], f32),
    ],
  },
  ({ p, q, y, at, reduce }) => {
    let mx = rowFill(T.bm, f32, negInf);
    for (const n of reduce.n) {
      mx = rowMax(p.tile(at.m, n).pad(negInf), mx);   // identity: negInf
    }
    let sm = rowFill(T.bm, f32, zero);
    for (const n of reduce.n) {
      sm = rowSum(q.tile(at.m, n).pad(zero), sm);     // identity: zero
    }
    for (const n of reduce.n) {
      y.tile(at.m, n).store(divRow(subRow(p.tile(at.m, n).pad(negInf), mx), sm));
    }
  },
);
