// Layer normalisation over a ragged axis — the THIRD kernel family.
//
//   y[m,n] = (x[m,n] - mean_n) * rstd_n,   rstd = 1 / sqrt(E[x^2] - mean^2 + eps)
//
// Structurally unlike both earlier schedules in one way: TWO accumulators carried
// through ONE loop. matmul has one, softmax has one and three passes.
//
// Note the identity. A masked lane must contribute nothing to either moment, so
// `zero` is right here — where softmax's max demanded `negInf`. The same axis,
// the same raggedness, a different identity, because the operator differs. That
// is what naming identities buys.

import {
  axis, raggedAxis, kernel, input, output, f32,
  rowFill, rowSum, sqTile, meanRow, rstdRow, subRow, mulRow, zero,
} from "../src/tessera";

const M = axis("m", 1024, 64);
const N = raggedAxis("n", 750, 64);

export const layernorm = kernel(
  {
    name: "layernorm_f32",
    grid: [M],
    reduce: [N],
    bindings: [
      input("x", [M, N], f32),
      output("y", [M, N], f32),
    ],
  },
  ({ x, y, at, reduce }) => {
    let s = rowFill(64, f32, zero);
    let q = rowFill(64, f32, zero);
    for (const n of reduce.n) {
      s = rowSum(x.tile(at.m, n).pad(zero), s);
      q = rowSum(sqTile(x.tile(at.m, n).pad(zero)), q);
    }
    const mu = meanRow(s);
    const inv = rstdRow(q, mu, 1e-5);
    for (const n of reduce.n) {
      y.tile(at.m, n).store(mulRow(subRow(x.tile(at.m, n).pad(zero), mu), inv));
    }
  },
);
