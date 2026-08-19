// C = A · Bᵀ, with B stored [N, K].
//
//   c[m,n] = sum_k a[m,k] * b[n,k]
//
// The first piece of the fourth family (docs/004 R12, gap G1). `mma` contracts
// its second operand's FIRST axis, and the common storage order for that operand
// is the other way round — which is also exactly the shape of attention's
// S = Q·Kᵀ.
//
// A transposed axis is a type error here on purpose, so this is not something the
// compiler may quietly accommodate: the kernel has to SAY it, with `.tileT()`,
// and saying it wrong is still an error. Note b's binding is [N, K] — the memory
// order — while `b.tileT(k, at.n)` asks for the [K, N] tile.

import {
  axis, tiling, kernel, input, output, f32, zeros, mma, relu,
} from "../src/tessera";

const T = tiling(f32, 64, 64, 16);

const M = axis("m", 1024, T.bm);
const N = axis("n",  768, T.bn);
const K = axis("k",  512, T.bk);

export const matmulBT = kernel(
  {
    name: "matmul_bt_f32",
    tile: T,
    axes: [M, N, K],
    bindings: [
      input("a", [M, K], f32),
      input("b", [N, K], f32),      // stored transposed
      output("c", [M, N], f32),
    ],
  },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      acc = mma(a.tile(at.m, k), b.tileT(k, at.n), acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  },
);
