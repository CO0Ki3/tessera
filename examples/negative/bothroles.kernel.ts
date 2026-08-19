// An axis reduced and stored along within one contraction.
//
//   c[m,n] = sum_n a[m,n] * b[n,n]
//
// The sum runs over the axis that indexes the output. This is not a feature that
// is missing — it is undefined, and it stays an error however far the emitter
// gets.
//
// It is NOT attention, which an earlier version of this file claimed. Attention
// has the head dimension contracted in `S = Q·Kᵀ` and free in `O = P·V`: the same
// axis in two roles across TWO contractions, which is legal and unimplemented.
// That case cannot even reach this check, because a body reducing over two axes
// is refused earlier. Conflating the two made an ill-defined kernel look like a
// milestone. See spike/attention/ for the real one.
import {
  axis, tiling, kernel, input, output, f32, zeros, mma, relu,
} from "../../src/tessera";
const T = tiling(f32, 64, 64, 16);
const M = axis("m", 1024, T.bm);
const N = axis("n", 768, T.bn);
export const k1 = kernel(
  { name: "both_roles", axes: [M, N],
    bindings: [input("a", [M, N], f32), input("b", [N, N], f32), output("c", [M, N], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const n of reduce.n) {          // n contracted here
      acc = mma(a.tile(at.m, n), b.tile(n, at.n), acc);
    }
    c.tile(at.m, at.n).store(relu(acc)); // ...and free here
  },
);
