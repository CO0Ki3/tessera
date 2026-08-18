import { axis, tiling, kernel, input, output, zeros, f32 } from "../src/tessera";
const T = tiling(f32, 64, 64, 16);
const M = axis("m", 1024, T.bm), N = axis("n", 768, T.bn), K = axis("k", 512, T.bk);
export const k10 = kernel(
  { name: "e10", tile: T, grid: [M, N], reduce: [K],
    bindings: [input("b0", [M, K], f32), input("b1", [M, K], f32), input("b2", [M, K], f32),
               input("b3", [M, K], f32), input("b4", [M, K], f32), input("b5", [M, K], f32),
               input("b6", [M, K], f32), input("b7", [M, K], f32), output("b8", [M, N], f32)] },
  () => {});
