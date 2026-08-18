import { axis, tiling, kernel, input, output, zeros, mma, relu, f32 } from "../src/tessera";
const T = tiling(f32, 64, 64, 16);
const M = axis("m", 1024, 32);   // blocked at 32, but T.bm is 64
const N = axis("n", 768, T.bn), K = axis("k", 512, T.bk);
export const k7 = kernel(
  { name: "e7", tile: T, grid: [M, N], reduce: [K],
    bindings: [input("a", [M, K], f32), input("b", [K, N], f32), output("c", [M, N], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) { acc = mma(a.tile(at.m, k), b.tile(k, at.n), acc); }
    c.tile(at.m, at.n).store(relu(acc));
  });
