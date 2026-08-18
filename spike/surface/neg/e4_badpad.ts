import { axis, raggedAxis, tiling, kernel, input, output, zeros, mma, relu, f32 } from "../src/tessera";
const T = tiling(f32, 64, 64, 16);
const M = raggedAxis("m", 1000, T.bm), N = axis("n", 768, T.bn), K = axis("k", 512, T.bk);
export const k4 = kernel(
  { name: "e4", tile: T, grid: [M, N], reduce: [K],
    bindings: [input("a", [M, K], f32), input("b", [K, N], f32), output("c", [M, N], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) { acc = mma(a.tile(at.m, k).pad(1), b.tile(k, at.n), acc); }
    c.tile(at.m, at.n).store(relu(acc));
  });
