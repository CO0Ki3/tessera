import { axis, tiling, kernel, input, output, zeros, mma, relu, f32 } from "../src/tessera";
const T = tiling(f32, 64, 64, 16);
const M = axis("m", 1024, T.bm), N = axis("n", 768, T.bn);
const K  = axis("k", 512, T.bk);
const K2 = axis("k", 256, T.bk);          // b's reduction axis is a different length
export const k2 = kernel(
  { name: "e2", tile: T, grid: [M, N], reduce: [K],
    bindings: [input("a", [M, K], f32), input("b", [K2, N], f32), output("c", [M, N], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) { acc = mma(a.tile(at.m, k), b.tile(k, at.n), acc); }
    c.tile(at.m, at.n).store(relu(acc));
  });
