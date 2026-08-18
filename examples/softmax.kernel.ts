// Row-wise softmax with a ragged reduction axis.
//
//   y[m,n] = exp(x[m,n] - max_n x[m,:]) / sum_n exp(x[m,:] - max)
//
// The SECOND kernel family, written to test whether the axis types derive its
// boundary conditions or whether the emitter has to be hand-extended.
// See docs/004-falsification.md.
//
// Note `.pad(negInf)`, not `.pad(zero)`: the identity for a masked max is
// negative infinity, and `zero` here would be right until a tail row happened to
// be entirely negative. rowMax's signature refuses it.

import {
  axis, raggedAxis, tiling, kernel, input, output, f32,
  rowFill, rowMax, rowSum, subRow, divRow, expTile, negInf, zero,
} from "../src/tessera";

const T = tiling(f32, 64, 64, 16);

const M = axis("m", 1024, T.bm);
const N = raggedAxis("n", 750, T.bn);   // 750 % 64 = 46

export const softmax = kernel(
  {
    name: "softmax_f32",
    grid: [M],
    reduce: [N],
    bindings: [
      input("x", [M, N], f32),
      output("y", [M, N], f32),
    ],
  },
  ({ x, y, at, reduce }) => {
    let mx = rowFill(T.bm, f32, negInf);
    for (const n of reduce.n) {
      mx = rowMax(x.tile(at.m, n).pad(negInf), mx);
    }
    let sm = rowFill(T.bm, f32, zero);
    for (const n of reduce.n) {
      sm = rowSum(expTile(subRow(x.tile(at.m, n).pad(negInf), mx)), sm);
    }
    for (const n of reduce.n) {
      y.tile(at.m, n).store(divRow(expTile(subRow(x.tile(at.m, n).pad(negInf), mx)), sm));
    }
  },
);
