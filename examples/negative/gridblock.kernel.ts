// The canonical kernel. tessera reads its SPEC entirely from the types below —
// note that `axis("m", 1024, 32)` passes `T.bm`, not a literal, so nothing
// short of the checker knows the block is 64.

import { axis, tiling, kernel, input, output, zeros, mma, relu, f32 } from "../../src/tessera";

const T = tiling(f32, 64, 64, 16);

const M = axis("m", 1024, 32);
const N = axis("n",  768, T.bn);
const K = axis("k",  512, T.bk);

export const matmulRelu = kernel(
  {
    name: "matmul_relu_f32",
    tile: T,
    axes: [M, N, K],
    bindings: [
      input("a", [M, K], f32),
      input("b", [K, N], f32),
      output("c", [M, N], f32),
    ],
  },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      acc = mma(a.tile(at.m, k), b.tile(k, at.n), acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  },
);
