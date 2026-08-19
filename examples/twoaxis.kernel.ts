// Two reductions, over two different axes, in one kernel.
//
//   y[m,n] = p[m,n] / (sum_n p[m,:] + sum_k q[m,k])
//
// Normalising `p` by a total that spans a second tensor of a different width —
// the row sum of a concatenation, without materialising the concatenation. The
// two reductions are INDEPENDENT: neither consumes the other's fragment.
//
// That independence is the point. G4 needs two things — more than one
// contraction in a body, and a fragment handed from one to the next through a
// register layout they may not share. This probes the first without the second,
// so whatever breaks here is about holding two plans, not about redistributing
// registers.
//
// What it took: the front end stopped capping the reduction axes at one, and the
// emitter builds one plan PER axis. The lane geometry is shared by construction —
// planContraction derives it from `accumulate` and the contiguous axis alone, and
// neither depends on which axis is contracted — so the plans differ only in what
// they stage and how far they walk. That is checked rather than assumed.
//
// The one thing beyond bookkeeping: a plan is built from the bindings its OWN pass
// reads. With one contraction every read binding took part in it; with two,
// handing a pass an operand whose axes it does not walk asks for the global
// coordinate of an axis that is neither accumulated nor contracted there.

import {
  axis, tiling, kernel, input, output, f32,
  rowFill, rowSum, divRow, zero,
} from "../src/tessera";

const T = tiling(f32, 64, 64, 16);

const M = axis("m", 1024, T.bm);
const N = axis("n",  768, T.bn);
const K = axis("k",  512, T.bn);   // a different extent, reduced separately

export const twoaxis = kernel(
  {
    name: "twoaxis_f32",
    axes: [M, N, K],
    bindings: [
      input("p", [M, N], f32),
      input("q", [M, K], f32),
      output("y", [M, N], f32),
    ],
  },
  ({ p, q, y, at, reduce }) => {
    let sp = rowFill(T.bm, f32, zero);
    for (const n of reduce.n) {
      sp = rowSum(p.tile(at.m, n), sp);
    }
    let sq = rowFill(T.bm, f32, zero);
    for (const k of reduce.k) {
      sq = rowSum(q.tile(at.m, k), sq);
    }
    for (const n of reduce.n) {
      y.tile(at.m, n).store(divRow(p.tile(at.m, n), sp));
    }
  },
);
