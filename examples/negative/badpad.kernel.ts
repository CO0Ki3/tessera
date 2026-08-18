// The ragged twin of examples/matmul.kernel.ts.
//
// The ONLY differences are three literals and two `.pad(one)` calls. The loop, the
// accumulator and the store are character-identical to the aligned kernel —
// tessera synthesises every boundary condition from the extents in the types.
//
// In Triton this would mean hand-editing a `mask=` expression on every load and
// on the store, and getting one wrong produces a kernel that is correct at 1024
// and quietly wrong at 1000.

import { raggedAxis, tiling, kernel, input, output, zeros, mma, relu, f32, zero, one } from "../../src/tessera";

const T = tiling(f32, 64, 64, 16);

const M = raggedAxis("m", 1000, T.bm);   // 1000 % 64 = 40
const N = raggedAxis("n",  750, T.bn);   //  750 % 64 = 46
const K = raggedAxis("k",  500, T.bk);   //  500 % 16 =  4

export const matmulReluRagged = kernel(
  {
    name: "matmul_relu_ragged",
    tile: T,
    grid: [M, N],
    reduce: [K],
    bindings: [
      input("a", [M, K], f32),
      input("b", [K, N], f32),
      output("c", [M, N], f32),
    ],
  },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      acc = mma(a.tile(at.m, k).pad(one), b.tile(k, at.n).pad(zero), acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  },
);
