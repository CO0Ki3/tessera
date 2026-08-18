// Attempt 1: softmax in the CURRENT surface, unmodified.
//
//   y[m,n] = exp(x[m,n] - max_n x[m,:]) / sum_n exp(x[m,:] - max)
//
// Structurally: one row-block per workgroup, reducing across the whole row.
//   grid   [m]        <- ONE axis, not two
//   reduce [n]
//   y      [m, n]     <- the output spans a grid axis AND the reduce axis
//
// In matmul the output spans exactly the two grid axes. Here it does not. This
// file is expected to fail; the point is which errors come out.

import { axis, raggedAxis, tiling, kernel, input, output, f32 } from "../../src/tessera";

const T = tiling(f32, 64, 64, 16);

const M = axis("m", 1024, T.bm);
const N = raggedAxis("n", 750, T.bn);

export const softmax = kernel(
  {
    name: "softmax_f32",
    tile: T,
    grid: [M],
    reduce: [N],
    bindings: [
      input("x", [M, N], f32),
      output("y", [M, N], f32),
    ],
  },
  ({ x, y, at, reduce }) => {
    // deliberately left empty: the spec above should already fail
  },
);
