// THE dangerous direction, finally demonstrable.
//
// Padding a masked MAX with zero gives a kernel that is right whenever the row
// contains a positive value, and silently wrong when every value in a ragged
// tail row is negative — max would return 0 instead of the true maximum, and the
// softmax that follows is wrong for that row only, at that shape only.
//
// This is the failure mode that motivated naming identities instead of writing
// numbers. rowMax takes `"exact" | "negInf"` and nothing else.
import {
  axis, raggedAxis, tiling, kernel, input, output, f32,
  rowFill, rowMax, rowSum, subRow, divRow, expTile, negInf, zero,
} from "../../src/tessera";
const T = tiling(f32, 64, 64, 16);
const M = axis("m", 1024, T.bm);
const N = raggedAxis("n", 750, T.bn);
export const k1 = kernel(
  { name: "max_pad_zero", axes: [M, N],
    bindings: [input("x", [M, N], f32), output("y", [M, N], f32)] },
  ({ x, y, at, reduce }) => {
    let mx = rowFill(T.bm, f32, negInf);
    for (const n of reduce.n) {
      mx = rowMax(x.tile(at.m, n).pad(zero), mx);   // <- wrong identity for a max
    }
    let sm = rowFill(T.bm, f32, zero);
    for (const n of reduce.n) {
      sm = rowSum(expTile(subRow(x.tile(at.m, n).pad(negInf), mx)), sm);
    }
    for (const n of reduce.n) {
      y.tile(at.m, n).store(divRow(expTile(subRow(x.tile(at.m, n).pad(negInf), mx)), sm));
    }
  });
