import { axis, tiling, kernel, input, output, zeros, mma, relu, f32, f16,
         requestDevice, randn, emptyHost, upload } from "../src/tessera";
const T = tiling(f32, 64, 64, 16);
const M = axis("m", 1024, T.bm), N = axis("n", 768, T.bn), K = axis("k", 512, T.bk);
const Mbad = axis("m", 999, T.bm);
const kern = kernel(
  { name: "e8", tile: T, grid: [M, N], reduce: [K],
    bindings: [input("a", [M, K], f32), input("b", [K, N], f32), output("c", [M, N], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) { acc = mma(a.tile(at.m, k), b.tile(k, at.n), acc); }
    c.tile(at.m, at.n).store(relu(acc));
  });
const T16 = tiling(f16, 64, 64, 16);
const M16 = axis("m", 1024, T16.bm), N16 = axis("n", 768, T16.bn), K16 = axis("k", 512, T16.bk);
const kern16 = kernel(
  { name: "e8b", tile: T16, grid: [M16, N16], reduce: [K16],
    bindings: [input("a", [M16, K16], f16), input("b", [K16, N16], f16), output("c", [M16, N16], f16)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T16.bm, T16.bn, f16);
    for (const k of reduce.k) { acc = mma(a.tile(at.m, k), b.tile(k, at.n), acc); }
    c.tile(at.m, at.n).store(relu(acc));
  });
export async function go() {
  const dev = await requestDevice();
  const aBad = randn([Mbad, K], f32);
  const b = randn([K, N], f32), c = emptyHost([M, N], f32);
  await kern.run(dev, { a: upload(dev, aBad), b: upload(dev, b), c: upload(dev, c) });  // wrong M
  await kern.run(dev, { a: upload(dev, randn([M, K], f32)), c: upload(dev, c) });       // missing b
  await kern16.run(dev, { a: upload(dev, randn([M16, K16], f16)), b: upload(dev, randn([K16, N16], f16)), c: upload(dev, emptyHost([M16, N16], f16)) });  // needs shader-f16
}
