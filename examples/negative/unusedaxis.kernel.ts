// An axis declared and never used.
//
// With `spec.grid` and `spec.reduce` this could not happen: every declared axis
// had a role because the declaration WAS the role. Now that roles come from the
// body, a name in `spec.axes` that the body never mentions is a typo or a
// leftover, and it must be an error rather than a silently ignored entry —
// otherwise `axes: [M, N, K]` on a kernel that only reduces over `J` compiles
// and means something the author did not write.
import {
  axis, tiling, kernel, input, output, f32, zeros, mma, relu,
} from "../../src/tessera";
const T = tiling(f32, 64, 64, 16);
const M = axis("m", 1024, T.bm);
const N = axis("n", 768, T.bn);
const K = axis("k", 512, T.bk);
const J = axis("j", 256, T.bk);   // declared below, never mentioned in the body
export const k1 = kernel(
  { name: "unused_axis", tile: T, axes: [M, N, K, J],
    bindings: [input("a", [M, K], f32), input("b", [K, N], f32), output("c", [M, N], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      acc = mma(a.tile(at.m, k), b.tile(k, at.n), acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  },
);
