// The identity element for a masked max is -inf, not 0. Padding a max-reduction
// with 0 gives a kernel that is right whenever the row contains a positive value
// and silently wrong when every value is negative — in the ragged tail only.
import { axis, raggedAxis, tiling, kernel, input, output, zeros, mma, relu, f32 } from "../../src/tessera";
const T = tiling(f32, 64, 64, 16);
const M = axis("m", 1024, T.bm);
const N = axis("n", 768, T.bn);
const K = raggedAxis("k", 500, T.bk);
export const k1 = kernel(
  { name: "pad_neginf", tile: T, grid: [M, N], reduce: [K],
    bindings: [input("a",[M,K],f32), input("b",[K,N],f32), output("c",[M,N],f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      acc = mma(a.tile(at.m, k).pad(-Infinity), b.tile(k, at.n).pad(0), acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });
