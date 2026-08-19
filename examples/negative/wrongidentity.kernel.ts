// The identity must match the OPERATOR, not merely be "an identity".
//
// `negInf` annihilates a max. It does not annihilate a sum — a masked lane would
// contribute -3.4e38 to the accumulation instead of nothing. mma sums, so it
// demands the additive identity and this is a type error.
//
// The reverse mistake is the dangerous one and is what motivated named
// identities: padding a masked MAX with `zero` is right whenever the row holds a
// positive value and silently wrong when every value is negative — in the ragged
// tail only. Once a max operator exists it will demand `negInf` the same way.
import { raggedAxis, axis, tiling, kernel, input, output, zeros, mma, relu, f32, negInf } from "../../src/tessera";
const T = tiling(f32, 64, 64, 16);
const M = axis("m", 1024, T.bm);
const N = axis("n", 768, T.bn);
const K = raggedAxis("k", 500, T.bk);
export const k1 = kernel(
  { name: "wrong_identity", tile: T, axes: [M, N, K],
    bindings: [input("a",[M,K],f32), input("b",[K,N],f32), output("c",[M,N],f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      acc = mma(a.tile(at.m, k).pad(negInf), b.tile(k, at.n).pad(negInf), acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });
