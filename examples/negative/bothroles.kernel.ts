// An axis that is both reduced and stored along — attention's shape.
//
// `S = Q·Kᵀ` contracts the head dimension while `O = P·V` leaves it free, so the
// same axis sits on both sides of the split. The surface can now SAY this, which
// is the point of deriving roles from the body rather than declaring them. The
// emitter cannot schedule it yet.
//
// So this must fail with a message that names the situation, not with a crash
// and not by silently picking one role. A compiler that quietly reinterprets the
// program is the failure this project's admission rule exists to prevent.
// See spike/attention/.
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
