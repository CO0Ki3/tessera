import { axis, tiling, kernel, input, output, zeros, mma, relu, f32 } from "../src/tessera";
const T = tiling(f32, 64, 64, 16);
const M = axis("m", 1024, T.bm), N = axis("n", 768, T.bn), K = axis("k", 512, T.bk);
export const k11 = kernel(
  { name: "e11", tile: T, grid: [M, N], reduce: [K],
    bindings: [input("a", [M, K], f32), input("b", [K, N], f32), output("c", [M, N], f32)] },
  (ctx) => {
    const { a, b, c, at, reduce } = ctx;
    let acc = zeros(T.bm, T.bn, f32);
    let i = 0;
    while (i < 4) { acc = relu(acc); i++; }                 // while: out of subset
    const f = (x: typeof acc) => relu(x);                    // closure: out of subset
    acc = f(acc);
    const anyHandle = a as any;                              // any: out of subset
    acc = anyHandle.whatever(acc);
    for (const k of reduce.k) { acc = mma(a.tile(at.m, k), b.tile(k, at.n), acc); }
    c.tile(at.m, at.n).store(acc);
  });
