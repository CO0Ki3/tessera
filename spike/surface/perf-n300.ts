import { axis, tiling, kernel, input, output, zeros, mma, relu, f32 } from "../src/tessera";
const T = tiling(f32, 64, 64, 16);

const M0 = axis("m", 128, T.bm), N0 = axis("n", 192, T.bn), K0 = axis("k", 64, T.bk);
export const kern0 = kernel(
  { name: "k0", tile: T, grid: [M0, N0], reduce: [K0],
    bindings: [input("a", [M0, K0], f32), input("b", [K0, N0], f32), output("c", [M0, N0], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M1 = axis("m", 192, T.bm), N1 = axis("n", 256, T.bn), K1 = axis("k", 80, T.bk);
export const kern1 = kernel(
  { name: "k1", tile: T, grid: [M1, N1], reduce: [K1],
    bindings: [input("a", [M1, K1], f32), input("b", [K1, N1], f32), output("c", [M1, N1], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M2 = axis("m", 256, T.bm), N2 = axis("n", 320, T.bn), K2 = axis("k", 96, T.bk);
export const kern2 = kernel(
  { name: "k2", tile: T, grid: [M2, N2], reduce: [K2],
    bindings: [input("a", [M2, K2], f32), input("b", [K2, N2], f32), output("c", [M2, N2], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M3 = axis("m", 320, T.bm), N3 = axis("n", 384, T.bn), K3 = axis("k", 112, T.bk);
export const kern3 = kernel(
  { name: "k3", tile: T, grid: [M3, N3], reduce: [K3],
    bindings: [input("a", [M3, K3], f32), input("b", [K3, N3], f32), output("c", [M3, N3], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M4 = axis("m", 384, T.bm), N4 = axis("n", 448, T.bn), K4 = axis("k", 128, T.bk);
export const kern4 = kernel(
  { name: "k4", tile: T, grid: [M4, N4], reduce: [K4],
    bindings: [input("a", [M4, K4], f32), input("b", [K4, N4], f32), output("c", [M4, N4], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M5 = axis("m", 448, T.bm), N5 = axis("n", 512, T.bn), K5 = axis("k", 144, T.bk);
export const kern5 = kernel(
  { name: "k5", tile: T, grid: [M5, N5], reduce: [K5],
    bindings: [input("a", [M5, K5], f32), input("b", [K5, N5], f32), output("c", [M5, N5], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M6 = axis("m", 512, T.bm), N6 = axis("n", 576, T.bn), K6 = axis("k", 160, T.bk);
export const kern6 = kernel(
  { name: "k6", tile: T, grid: [M6, N6], reduce: [K6],
    bindings: [input("a", [M6, K6], f32), input("b", [K6, N6], f32), output("c", [M6, N6], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M7 = axis("m", 576, T.bm), N7 = axis("n", 640, T.bn), K7 = axis("k", 176, T.bk);
export const kern7 = kernel(
  { name: "k7", tile: T, grid: [M7, N7], reduce: [K7],
    bindings: [input("a", [M7, K7], f32), input("b", [K7, N7], f32), output("c", [M7, N7], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M8 = axis("m", 640, T.bm), N8 = axis("n", 704, T.bn), K8 = axis("k", 192, T.bk);
export const kern8 = kernel(
  { name: "k8", tile: T, grid: [M8, N8], reduce: [K8],
    bindings: [input("a", [M8, K8], f32), input("b", [K8, N8], f32), output("c", [M8, N8], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M9 = axis("m", 704, T.bm), N9 = axis("n", 768, T.bn), K9 = axis("k", 208, T.bk);
export const kern9 = kernel(
  { name: "k9", tile: T, grid: [M9, N9], reduce: [K9],
    bindings: [input("a", [M9, K9], f32), input("b", [K9, N9], f32), output("c", [M9, N9], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M10 = axis("m", 768, T.bm), N10 = axis("n", 832, T.bn), K10 = axis("k", 224, T.bk);
export const kern10 = kernel(
  { name: "k10", tile: T, grid: [M10, N10], reduce: [K10],
    bindings: [input("a", [M10, K10], f32), input("b", [K10, N10], f32), output("c", [M10, N10], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M11 = axis("m", 832, T.bm), N11 = axis("n", 192, T.bn), K11 = axis("k", 240, T.bk);
export const kern11 = kernel(
  { name: "k11", tile: T, grid: [M11, N11], reduce: [K11],
    bindings: [input("a", [M11, K11], f32), input("b", [K11, N11], f32), output("c", [M11, N11], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M12 = axis("m", 896, T.bm), N12 = axis("n", 256, T.bn), K12 = axis("k", 256, T.bk);
export const kern12 = kernel(
  { name: "k12", tile: T, grid: [M12, N12], reduce: [K12],
    bindings: [input("a", [M12, K12], f32), input("b", [K12, N12], f32), output("c", [M12, N12], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M13 = axis("m", 960, T.bm), N13 = axis("n", 320, T.bn), K13 = axis("k", 272, T.bk);
export const kern13 = kernel(
  { name: "k13", tile: T, grid: [M13, N13], reduce: [K13],
    bindings: [input("a", [M13, K13], f32), input("b", [K13, N13], f32), output("c", [M13, N13], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M14 = axis("m", 128, T.bm), N14 = axis("n", 384, T.bn), K14 = axis("k", 288, T.bk);
export const kern14 = kernel(
  { name: "k14", tile: T, grid: [M14, N14], reduce: [K14],
    bindings: [input("a", [M14, K14], f32), input("b", [K14, N14], f32), output("c", [M14, N14], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M15 = axis("m", 192, T.bm), N15 = axis("n", 448, T.bn), K15 = axis("k", 304, T.bk);
export const kern15 = kernel(
  { name: "k15", tile: T, grid: [M15, N15], reduce: [K15],
    bindings: [input("a", [M15, K15], f32), input("b", [K15, N15], f32), output("c", [M15, N15], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M16 = axis("m", 256, T.bm), N16 = axis("n", 512, T.bn), K16 = axis("k", 320, T.bk);
export const kern16 = kernel(
  { name: "k16", tile: T, grid: [M16, N16], reduce: [K16],
    bindings: [input("a", [M16, K16], f32), input("b", [K16, N16], f32), output("c", [M16, N16], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M17 = axis("m", 320, T.bm), N17 = axis("n", 576, T.bn), K17 = axis("k", 336, T.bk);
export const kern17 = kernel(
  { name: "k17", tile: T, grid: [M17, N17], reduce: [K17],
    bindings: [input("a", [M17, K17], f32), input("b", [K17, N17], f32), output("c", [M17, N17], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M18 = axis("m", 384, T.bm), N18 = axis("n", 640, T.bn), K18 = axis("k", 352, T.bk);
export const kern18 = kernel(
  { name: "k18", tile: T, grid: [M18, N18], reduce: [K18],
    bindings: [input("a", [M18, K18], f32), input("b", [K18, N18], f32), output("c", [M18, N18], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M19 = axis("m", 448, T.bm), N19 = axis("n", 704, T.bn), K19 = axis("k", 368, T.bk);
export const kern19 = kernel(
  { name: "k19", tile: T, grid: [M19, N19], reduce: [K19],
    bindings: [input("a", [M19, K19], f32), input("b", [K19, N19], f32), output("c", [M19, N19], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M20 = axis("m", 512, T.bm), N20 = axis("n", 768, T.bn), K20 = axis("k", 64, T.bk);
export const kern20 = kernel(
  { name: "k20", tile: T, grid: [M20, N20], reduce: [K20],
    bindings: [input("a", [M20, K20], f32), input("b", [K20, N20], f32), output("c", [M20, N20], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M21 = axis("m", 576, T.bm), N21 = axis("n", 832, T.bn), K21 = axis("k", 80, T.bk);
export const kern21 = kernel(
  { name: "k21", tile: T, grid: [M21, N21], reduce: [K21],
    bindings: [input("a", [M21, K21], f32), input("b", [K21, N21], f32), output("c", [M21, N21], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M22 = axis("m", 640, T.bm), N22 = axis("n", 192, T.bn), K22 = axis("k", 96, T.bk);
export const kern22 = kernel(
  { name: "k22", tile: T, grid: [M22, N22], reduce: [K22],
    bindings: [input("a", [M22, K22], f32), input("b", [K22, N22], f32), output("c", [M22, N22], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M23 = axis("m", 704, T.bm), N23 = axis("n", 256, T.bn), K23 = axis("k", 112, T.bk);
export const kern23 = kernel(
  { name: "k23", tile: T, grid: [M23, N23], reduce: [K23],
    bindings: [input("a", [M23, K23], f32), input("b", [K23, N23], f32), output("c", [M23, N23], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M24 = axis("m", 768, T.bm), N24 = axis("n", 320, T.bn), K24 = axis("k", 128, T.bk);
export const kern24 = kernel(
  { name: "k24", tile: T, grid: [M24, N24], reduce: [K24],
    bindings: [input("a", [M24, K24], f32), input("b", [K24, N24], f32), output("c", [M24, N24], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M25 = axis("m", 832, T.bm), N25 = axis("n", 384, T.bn), K25 = axis("k", 144, T.bk);
export const kern25 = kernel(
  { name: "k25", tile: T, grid: [M25, N25], reduce: [K25],
    bindings: [input("a", [M25, K25], f32), input("b", [K25, N25], f32), output("c", [M25, N25], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M26 = axis("m", 896, T.bm), N26 = axis("n", 448, T.bn), K26 = axis("k", 160, T.bk);
export const kern26 = kernel(
  { name: "k26", tile: T, grid: [M26, N26], reduce: [K26],
    bindings: [input("a", [M26, K26], f32), input("b", [K26, N26], f32), output("c", [M26, N26], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M27 = axis("m", 960, T.bm), N27 = axis("n", 512, T.bn), K27 = axis("k", 176, T.bk);
export const kern27 = kernel(
  { name: "k27", tile: T, grid: [M27, N27], reduce: [K27],
    bindings: [input("a", [M27, K27], f32), input("b", [K27, N27], f32), output("c", [M27, N27], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M28 = axis("m", 128, T.bm), N28 = axis("n", 576, T.bn), K28 = axis("k", 192, T.bk);
export const kern28 = kernel(
  { name: "k28", tile: T, grid: [M28, N28], reduce: [K28],
    bindings: [input("a", [M28, K28], f32), input("b", [K28, N28], f32), output("c", [M28, N28], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M29 = axis("m", 192, T.bm), N29 = axis("n", 640, T.bn), K29 = axis("k", 208, T.bk);
export const kern29 = kernel(
  { name: "k29", tile: T, grid: [M29, N29], reduce: [K29],
    bindings: [input("a", [M29, K29], f32), input("b", [K29, N29], f32), output("c", [M29, N29], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M30 = axis("m", 256, T.bm), N30 = axis("n", 704, T.bn), K30 = axis("k", 224, T.bk);
export const kern30 = kernel(
  { name: "k30", tile: T, grid: [M30, N30], reduce: [K30],
    bindings: [input("a", [M30, K30], f32), input("b", [K30, N30], f32), output("c", [M30, N30], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M31 = axis("m", 320, T.bm), N31 = axis("n", 768, T.bn), K31 = axis("k", 240, T.bk);
export const kern31 = kernel(
  { name: "k31", tile: T, grid: [M31, N31], reduce: [K31],
    bindings: [input("a", [M31, K31], f32), input("b", [K31, N31], f32), output("c", [M31, N31], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M32 = axis("m", 384, T.bm), N32 = axis("n", 832, T.bn), K32 = axis("k", 256, T.bk);
export const kern32 = kernel(
  { name: "k32", tile: T, grid: [M32, N32], reduce: [K32],
    bindings: [input("a", [M32, K32], f32), input("b", [K32, N32], f32), output("c", [M32, N32], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M33 = axis("m", 448, T.bm), N33 = axis("n", 192, T.bn), K33 = axis("k", 272, T.bk);
export const kern33 = kernel(
  { name: "k33", tile: T, grid: [M33, N33], reduce: [K33],
    bindings: [input("a", [M33, K33], f32), input("b", [K33, N33], f32), output("c", [M33, N33], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M34 = axis("m", 512, T.bm), N34 = axis("n", 256, T.bn), K34 = axis("k", 288, T.bk);
export const kern34 = kernel(
  { name: "k34", tile: T, grid: [M34, N34], reduce: [K34],
    bindings: [input("a", [M34, K34], f32), input("b", [K34, N34], f32), output("c", [M34, N34], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M35 = axis("m", 576, T.bm), N35 = axis("n", 320, T.bn), K35 = axis("k", 304, T.bk);
export const kern35 = kernel(
  { name: "k35", tile: T, grid: [M35, N35], reduce: [K35],
    bindings: [input("a", [M35, K35], f32), input("b", [K35, N35], f32), output("c", [M35, N35], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M36 = axis("m", 640, T.bm), N36 = axis("n", 384, T.bn), K36 = axis("k", 320, T.bk);
export const kern36 = kernel(
  { name: "k36", tile: T, grid: [M36, N36], reduce: [K36],
    bindings: [input("a", [M36, K36], f32), input("b", [K36, N36], f32), output("c", [M36, N36], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M37 = axis("m", 704, T.bm), N37 = axis("n", 448, T.bn), K37 = axis("k", 336, T.bk);
export const kern37 = kernel(
  { name: "k37", tile: T, grid: [M37, N37], reduce: [K37],
    bindings: [input("a", [M37, K37], f32), input("b", [K37, N37], f32), output("c", [M37, N37], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M38 = axis("m", 768, T.bm), N38 = axis("n", 512, T.bn), K38 = axis("k", 352, T.bk);
export const kern38 = kernel(
  { name: "k38", tile: T, grid: [M38, N38], reduce: [K38],
    bindings: [input("a", [M38, K38], f32), input("b", [K38, N38], f32), output("c", [M38, N38], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M39 = axis("m", 832, T.bm), N39 = axis("n", 576, T.bn), K39 = axis("k", 368, T.bk);
export const kern39 = kernel(
  { name: "k39", tile: T, grid: [M39, N39], reduce: [K39],
    bindings: [input("a", [M39, K39], f32), input("b", [K39, N39], f32), output("c", [M39, N39], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M40 = axis("m", 896, T.bm), N40 = axis("n", 640, T.bn), K40 = axis("k", 64, T.bk);
export const kern40 = kernel(
  { name: "k40", tile: T, grid: [M40, N40], reduce: [K40],
    bindings: [input("a", [M40, K40], f32), input("b", [K40, N40], f32), output("c", [M40, N40], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M41 = axis("m", 960, T.bm), N41 = axis("n", 704, T.bn), K41 = axis("k", 80, T.bk);
export const kern41 = kernel(
  { name: "k41", tile: T, grid: [M41, N41], reduce: [K41],
    bindings: [input("a", [M41, K41], f32), input("b", [K41, N41], f32), output("c", [M41, N41], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M42 = axis("m", 128, T.bm), N42 = axis("n", 768, T.bn), K42 = axis("k", 96, T.bk);
export const kern42 = kernel(
  { name: "k42", tile: T, grid: [M42, N42], reduce: [K42],
    bindings: [input("a", [M42, K42], f32), input("b", [K42, N42], f32), output("c", [M42, N42], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M43 = axis("m", 192, T.bm), N43 = axis("n", 832, T.bn), K43 = axis("k", 112, T.bk);
export const kern43 = kernel(
  { name: "k43", tile: T, grid: [M43, N43], reduce: [K43],
    bindings: [input("a", [M43, K43], f32), input("b", [K43, N43], f32), output("c", [M43, N43], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M44 = axis("m", 256, T.bm), N44 = axis("n", 192, T.bn), K44 = axis("k", 128, T.bk);
export const kern44 = kernel(
  { name: "k44", tile: T, grid: [M44, N44], reduce: [K44],
    bindings: [input("a", [M44, K44], f32), input("b", [K44, N44], f32), output("c", [M44, N44], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M45 = axis("m", 320, T.bm), N45 = axis("n", 256, T.bn), K45 = axis("k", 144, T.bk);
export const kern45 = kernel(
  { name: "k45", tile: T, grid: [M45, N45], reduce: [K45],
    bindings: [input("a", [M45, K45], f32), input("b", [K45, N45], f32), output("c", [M45, N45], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M46 = axis("m", 384, T.bm), N46 = axis("n", 320, T.bn), K46 = axis("k", 160, T.bk);
export const kern46 = kernel(
  { name: "k46", tile: T, grid: [M46, N46], reduce: [K46],
    bindings: [input("a", [M46, K46], f32), input("b", [K46, N46], f32), output("c", [M46, N46], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M47 = axis("m", 448, T.bm), N47 = axis("n", 384, T.bn), K47 = axis("k", 176, T.bk);
export const kern47 = kernel(
  { name: "k47", tile: T, grid: [M47, N47], reduce: [K47],
    bindings: [input("a", [M47, K47], f32), input("b", [K47, N47], f32), output("c", [M47, N47], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M48 = axis("m", 512, T.bm), N48 = axis("n", 448, T.bn), K48 = axis("k", 192, T.bk);
export const kern48 = kernel(
  { name: "k48", tile: T, grid: [M48, N48], reduce: [K48],
    bindings: [input("a", [M48, K48], f32), input("b", [K48, N48], f32), output("c", [M48, N48], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M49 = axis("m", 576, T.bm), N49 = axis("n", 512, T.bn), K49 = axis("k", 208, T.bk);
export const kern49 = kernel(
  { name: "k49", tile: T, grid: [M49, N49], reduce: [K49],
    bindings: [input("a", [M49, K49], f32), input("b", [K49, N49], f32), output("c", [M49, N49], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M50 = axis("m", 640, T.bm), N50 = axis("n", 576, T.bn), K50 = axis("k", 224, T.bk);
export const kern50 = kernel(
  { name: "k50", tile: T, grid: [M50, N50], reduce: [K50],
    bindings: [input("a", [M50, K50], f32), input("b", [K50, N50], f32), output("c", [M50, N50], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M51 = axis("m", 704, T.bm), N51 = axis("n", 640, T.bn), K51 = axis("k", 240, T.bk);
export const kern51 = kernel(
  { name: "k51", tile: T, grid: [M51, N51], reduce: [K51],
    bindings: [input("a", [M51, K51], f32), input("b", [K51, N51], f32), output("c", [M51, N51], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M52 = axis("m", 768, T.bm), N52 = axis("n", 704, T.bn), K52 = axis("k", 256, T.bk);
export const kern52 = kernel(
  { name: "k52", tile: T, grid: [M52, N52], reduce: [K52],
    bindings: [input("a", [M52, K52], f32), input("b", [K52, N52], f32), output("c", [M52, N52], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M53 = axis("m", 832, T.bm), N53 = axis("n", 768, T.bn), K53 = axis("k", 272, T.bk);
export const kern53 = kernel(
  { name: "k53", tile: T, grid: [M53, N53], reduce: [K53],
    bindings: [input("a", [M53, K53], f32), input("b", [K53, N53], f32), output("c", [M53, N53], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M54 = axis("m", 896, T.bm), N54 = axis("n", 832, T.bn), K54 = axis("k", 288, T.bk);
export const kern54 = kernel(
  { name: "k54", tile: T, grid: [M54, N54], reduce: [K54],
    bindings: [input("a", [M54, K54], f32), input("b", [K54, N54], f32), output("c", [M54, N54], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M55 = axis("m", 960, T.bm), N55 = axis("n", 192, T.bn), K55 = axis("k", 304, T.bk);
export const kern55 = kernel(
  { name: "k55", tile: T, grid: [M55, N55], reduce: [K55],
    bindings: [input("a", [M55, K55], f32), input("b", [K55, N55], f32), output("c", [M55, N55], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M56 = axis("m", 128, T.bm), N56 = axis("n", 256, T.bn), K56 = axis("k", 320, T.bk);
export const kern56 = kernel(
  { name: "k56", tile: T, grid: [M56, N56], reduce: [K56],
    bindings: [input("a", [M56, K56], f32), input("b", [K56, N56], f32), output("c", [M56, N56], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M57 = axis("m", 192, T.bm), N57 = axis("n", 320, T.bn), K57 = axis("k", 336, T.bk);
export const kern57 = kernel(
  { name: "k57", tile: T, grid: [M57, N57], reduce: [K57],
    bindings: [input("a", [M57, K57], f32), input("b", [K57, N57], f32), output("c", [M57, N57], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M58 = axis("m", 256, T.bm), N58 = axis("n", 384, T.bn), K58 = axis("k", 352, T.bk);
export const kern58 = kernel(
  { name: "k58", tile: T, grid: [M58, N58], reduce: [K58],
    bindings: [input("a", [M58, K58], f32), input("b", [K58, N58], f32), output("c", [M58, N58], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M59 = axis("m", 320, T.bm), N59 = axis("n", 448, T.bn), K59 = axis("k", 368, T.bk);
export const kern59 = kernel(
  { name: "k59", tile: T, grid: [M59, N59], reduce: [K59],
    bindings: [input("a", [M59, K59], f32), input("b", [K59, N59], f32), output("c", [M59, N59], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M60 = axis("m", 384, T.bm), N60 = axis("n", 512, T.bn), K60 = axis("k", 64, T.bk);
export const kern60 = kernel(
  { name: "k60", tile: T, grid: [M60, N60], reduce: [K60],
    bindings: [input("a", [M60, K60], f32), input("b", [K60, N60], f32), output("c", [M60, N60], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M61 = axis("m", 448, T.bm), N61 = axis("n", 576, T.bn), K61 = axis("k", 80, T.bk);
export const kern61 = kernel(
  { name: "k61", tile: T, grid: [M61, N61], reduce: [K61],
    bindings: [input("a", [M61, K61], f32), input("b", [K61, N61], f32), output("c", [M61, N61], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M62 = axis("m", 512, T.bm), N62 = axis("n", 640, T.bn), K62 = axis("k", 96, T.bk);
export const kern62 = kernel(
  { name: "k62", tile: T, grid: [M62, N62], reduce: [K62],
    bindings: [input("a", [M62, K62], f32), input("b", [K62, N62], f32), output("c", [M62, N62], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M63 = axis("m", 576, T.bm), N63 = axis("n", 704, T.bn), K63 = axis("k", 112, T.bk);
export const kern63 = kernel(
  { name: "k63", tile: T, grid: [M63, N63], reduce: [K63],
    bindings: [input("a", [M63, K63], f32), input("b", [K63, N63], f32), output("c", [M63, N63], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M64 = axis("m", 640, T.bm), N64 = axis("n", 768, T.bn), K64 = axis("k", 128, T.bk);
export const kern64 = kernel(
  { name: "k64", tile: T, grid: [M64, N64], reduce: [K64],
    bindings: [input("a", [M64, K64], f32), input("b", [K64, N64], f32), output("c", [M64, N64], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M65 = axis("m", 704, T.bm), N65 = axis("n", 832, T.bn), K65 = axis("k", 144, T.bk);
export const kern65 = kernel(
  { name: "k65", tile: T, grid: [M65, N65], reduce: [K65],
    bindings: [input("a", [M65, K65], f32), input("b", [K65, N65], f32), output("c", [M65, N65], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M66 = axis("m", 768, T.bm), N66 = axis("n", 192, T.bn), K66 = axis("k", 160, T.bk);
export const kern66 = kernel(
  { name: "k66", tile: T, grid: [M66, N66], reduce: [K66],
    bindings: [input("a", [M66, K66], f32), input("b", [K66, N66], f32), output("c", [M66, N66], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M67 = axis("m", 832, T.bm), N67 = axis("n", 256, T.bn), K67 = axis("k", 176, T.bk);
export const kern67 = kernel(
  { name: "k67", tile: T, grid: [M67, N67], reduce: [K67],
    bindings: [input("a", [M67, K67], f32), input("b", [K67, N67], f32), output("c", [M67, N67], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M68 = axis("m", 896, T.bm), N68 = axis("n", 320, T.bn), K68 = axis("k", 192, T.bk);
export const kern68 = kernel(
  { name: "k68", tile: T, grid: [M68, N68], reduce: [K68],
    bindings: [input("a", [M68, K68], f32), input("b", [K68, N68], f32), output("c", [M68, N68], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M69 = axis("m", 960, T.bm), N69 = axis("n", 384, T.bn), K69 = axis("k", 208, T.bk);
export const kern69 = kernel(
  { name: "k69", tile: T, grid: [M69, N69], reduce: [K69],
    bindings: [input("a", [M69, K69], f32), input("b", [K69, N69], f32), output("c", [M69, N69], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M70 = axis("m", 128, T.bm), N70 = axis("n", 448, T.bn), K70 = axis("k", 224, T.bk);
export const kern70 = kernel(
  { name: "k70", tile: T, grid: [M70, N70], reduce: [K70],
    bindings: [input("a", [M70, K70], f32), input("b", [K70, N70], f32), output("c", [M70, N70], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M71 = axis("m", 192, T.bm), N71 = axis("n", 512, T.bn), K71 = axis("k", 240, T.bk);
export const kern71 = kernel(
  { name: "k71", tile: T, grid: [M71, N71], reduce: [K71],
    bindings: [input("a", [M71, K71], f32), input("b", [K71, N71], f32), output("c", [M71, N71], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M72 = axis("m", 256, T.bm), N72 = axis("n", 576, T.bn), K72 = axis("k", 256, T.bk);
export const kern72 = kernel(
  { name: "k72", tile: T, grid: [M72, N72], reduce: [K72],
    bindings: [input("a", [M72, K72], f32), input("b", [K72, N72], f32), output("c", [M72, N72], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M73 = axis("m", 320, T.bm), N73 = axis("n", 640, T.bn), K73 = axis("k", 272, T.bk);
export const kern73 = kernel(
  { name: "k73", tile: T, grid: [M73, N73], reduce: [K73],
    bindings: [input("a", [M73, K73], f32), input("b", [K73, N73], f32), output("c", [M73, N73], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M74 = axis("m", 384, T.bm), N74 = axis("n", 704, T.bn), K74 = axis("k", 288, T.bk);
export const kern74 = kernel(
  { name: "k74", tile: T, grid: [M74, N74], reduce: [K74],
    bindings: [input("a", [M74, K74], f32), input("b", [K74, N74], f32), output("c", [M74, N74], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M75 = axis("m", 448, T.bm), N75 = axis("n", 768, T.bn), K75 = axis("k", 304, T.bk);
export const kern75 = kernel(
  { name: "k75", tile: T, grid: [M75, N75], reduce: [K75],
    bindings: [input("a", [M75, K75], f32), input("b", [K75, N75], f32), output("c", [M75, N75], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M76 = axis("m", 512, T.bm), N76 = axis("n", 832, T.bn), K76 = axis("k", 320, T.bk);
export const kern76 = kernel(
  { name: "k76", tile: T, grid: [M76, N76], reduce: [K76],
    bindings: [input("a", [M76, K76], f32), input("b", [K76, N76], f32), output("c", [M76, N76], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M77 = axis("m", 576, T.bm), N77 = axis("n", 192, T.bn), K77 = axis("k", 336, T.bk);
export const kern77 = kernel(
  { name: "k77", tile: T, grid: [M77, N77], reduce: [K77],
    bindings: [input("a", [M77, K77], f32), input("b", [K77, N77], f32), output("c", [M77, N77], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M78 = axis("m", 640, T.bm), N78 = axis("n", 256, T.bn), K78 = axis("k", 352, T.bk);
export const kern78 = kernel(
  { name: "k78", tile: T, grid: [M78, N78], reduce: [K78],
    bindings: [input("a", [M78, K78], f32), input("b", [K78, N78], f32), output("c", [M78, N78], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M79 = axis("m", 704, T.bm), N79 = axis("n", 320, T.bn), K79 = axis("k", 368, T.bk);
export const kern79 = kernel(
  { name: "k79", tile: T, grid: [M79, N79], reduce: [K79],
    bindings: [input("a", [M79, K79], f32), input("b", [K79, N79], f32), output("c", [M79, N79], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M80 = axis("m", 768, T.bm), N80 = axis("n", 384, T.bn), K80 = axis("k", 64, T.bk);
export const kern80 = kernel(
  { name: "k80", tile: T, grid: [M80, N80], reduce: [K80],
    bindings: [input("a", [M80, K80], f32), input("b", [K80, N80], f32), output("c", [M80, N80], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M81 = axis("m", 832, T.bm), N81 = axis("n", 448, T.bn), K81 = axis("k", 80, T.bk);
export const kern81 = kernel(
  { name: "k81", tile: T, grid: [M81, N81], reduce: [K81],
    bindings: [input("a", [M81, K81], f32), input("b", [K81, N81], f32), output("c", [M81, N81], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M82 = axis("m", 896, T.bm), N82 = axis("n", 512, T.bn), K82 = axis("k", 96, T.bk);
export const kern82 = kernel(
  { name: "k82", tile: T, grid: [M82, N82], reduce: [K82],
    bindings: [input("a", [M82, K82], f32), input("b", [K82, N82], f32), output("c", [M82, N82], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M83 = axis("m", 960, T.bm), N83 = axis("n", 576, T.bn), K83 = axis("k", 112, T.bk);
export const kern83 = kernel(
  { name: "k83", tile: T, grid: [M83, N83], reduce: [K83],
    bindings: [input("a", [M83, K83], f32), input("b", [K83, N83], f32), output("c", [M83, N83], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M84 = axis("m", 128, T.bm), N84 = axis("n", 640, T.bn), K84 = axis("k", 128, T.bk);
export const kern84 = kernel(
  { name: "k84", tile: T, grid: [M84, N84], reduce: [K84],
    bindings: [input("a", [M84, K84], f32), input("b", [K84, N84], f32), output("c", [M84, N84], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M85 = axis("m", 192, T.bm), N85 = axis("n", 704, T.bn), K85 = axis("k", 144, T.bk);
export const kern85 = kernel(
  { name: "k85", tile: T, grid: [M85, N85], reduce: [K85],
    bindings: [input("a", [M85, K85], f32), input("b", [K85, N85], f32), output("c", [M85, N85], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M86 = axis("m", 256, T.bm), N86 = axis("n", 768, T.bn), K86 = axis("k", 160, T.bk);
export const kern86 = kernel(
  { name: "k86", tile: T, grid: [M86, N86], reduce: [K86],
    bindings: [input("a", [M86, K86], f32), input("b", [K86, N86], f32), output("c", [M86, N86], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M87 = axis("m", 320, T.bm), N87 = axis("n", 832, T.bn), K87 = axis("k", 176, T.bk);
export const kern87 = kernel(
  { name: "k87", tile: T, grid: [M87, N87], reduce: [K87],
    bindings: [input("a", [M87, K87], f32), input("b", [K87, N87], f32), output("c", [M87, N87], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M88 = axis("m", 384, T.bm), N88 = axis("n", 192, T.bn), K88 = axis("k", 192, T.bk);
export const kern88 = kernel(
  { name: "k88", tile: T, grid: [M88, N88], reduce: [K88],
    bindings: [input("a", [M88, K88], f32), input("b", [K88, N88], f32), output("c", [M88, N88], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M89 = axis("m", 448, T.bm), N89 = axis("n", 256, T.bn), K89 = axis("k", 208, T.bk);
export const kern89 = kernel(
  { name: "k89", tile: T, grid: [M89, N89], reduce: [K89],
    bindings: [input("a", [M89, K89], f32), input("b", [K89, N89], f32), output("c", [M89, N89], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M90 = axis("m", 512, T.bm), N90 = axis("n", 320, T.bn), K90 = axis("k", 224, T.bk);
export const kern90 = kernel(
  { name: "k90", tile: T, grid: [M90, N90], reduce: [K90],
    bindings: [input("a", [M90, K90], f32), input("b", [K90, N90], f32), output("c", [M90, N90], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M91 = axis("m", 576, T.bm), N91 = axis("n", 384, T.bn), K91 = axis("k", 240, T.bk);
export const kern91 = kernel(
  { name: "k91", tile: T, grid: [M91, N91], reduce: [K91],
    bindings: [input("a", [M91, K91], f32), input("b", [K91, N91], f32), output("c", [M91, N91], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M92 = axis("m", 640, T.bm), N92 = axis("n", 448, T.bn), K92 = axis("k", 256, T.bk);
export const kern92 = kernel(
  { name: "k92", tile: T, grid: [M92, N92], reduce: [K92],
    bindings: [input("a", [M92, K92], f32), input("b", [K92, N92], f32), output("c", [M92, N92], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M93 = axis("m", 704, T.bm), N93 = axis("n", 512, T.bn), K93 = axis("k", 272, T.bk);
export const kern93 = kernel(
  { name: "k93", tile: T, grid: [M93, N93], reduce: [K93],
    bindings: [input("a", [M93, K93], f32), input("b", [K93, N93], f32), output("c", [M93, N93], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M94 = axis("m", 768, T.bm), N94 = axis("n", 576, T.bn), K94 = axis("k", 288, T.bk);
export const kern94 = kernel(
  { name: "k94", tile: T, grid: [M94, N94], reduce: [K94],
    bindings: [input("a", [M94, K94], f32), input("b", [K94, N94], f32), output("c", [M94, N94], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M95 = axis("m", 832, T.bm), N95 = axis("n", 640, T.bn), K95 = axis("k", 304, T.bk);
export const kern95 = kernel(
  { name: "k95", tile: T, grid: [M95, N95], reduce: [K95],
    bindings: [input("a", [M95, K95], f32), input("b", [K95, N95], f32), output("c", [M95, N95], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M96 = axis("m", 896, T.bm), N96 = axis("n", 704, T.bn), K96 = axis("k", 320, T.bk);
export const kern96 = kernel(
  { name: "k96", tile: T, grid: [M96, N96], reduce: [K96],
    bindings: [input("a", [M96, K96], f32), input("b", [K96, N96], f32), output("c", [M96, N96], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M97 = axis("m", 960, T.bm), N97 = axis("n", 768, T.bn), K97 = axis("k", 336, T.bk);
export const kern97 = kernel(
  { name: "k97", tile: T, grid: [M97, N97], reduce: [K97],
    bindings: [input("a", [M97, K97], f32), input("b", [K97, N97], f32), output("c", [M97, N97], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M98 = axis("m", 128, T.bm), N98 = axis("n", 832, T.bn), K98 = axis("k", 352, T.bk);
export const kern98 = kernel(
  { name: "k98", tile: T, grid: [M98, N98], reduce: [K98],
    bindings: [input("a", [M98, K98], f32), input("b", [K98, N98], f32), output("c", [M98, N98], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M99 = axis("m", 192, T.bm), N99 = axis("n", 192, T.bn), K99 = axis("k", 368, T.bk);
export const kern99 = kernel(
  { name: "k99", tile: T, grid: [M99, N99], reduce: [K99],
    bindings: [input("a", [M99, K99], f32), input("b", [K99, N99], f32), output("c", [M99, N99], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M100 = axis("m", 256, T.bm), N100 = axis("n", 256, T.bn), K100 = axis("k", 64, T.bk);
export const kern100 = kernel(
  { name: "k100", tile: T, grid: [M100, N100], reduce: [K100],
    bindings: [input("a", [M100, K100], f32), input("b", [K100, N100], f32), output("c", [M100, N100], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M101 = axis("m", 320, T.bm), N101 = axis("n", 320, T.bn), K101 = axis("k", 80, T.bk);
export const kern101 = kernel(
  { name: "k101", tile: T, grid: [M101, N101], reduce: [K101],
    bindings: [input("a", [M101, K101], f32), input("b", [K101, N101], f32), output("c", [M101, N101], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M102 = axis("m", 384, T.bm), N102 = axis("n", 384, T.bn), K102 = axis("k", 96, T.bk);
export const kern102 = kernel(
  { name: "k102", tile: T, grid: [M102, N102], reduce: [K102],
    bindings: [input("a", [M102, K102], f32), input("b", [K102, N102], f32), output("c", [M102, N102], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M103 = axis("m", 448, T.bm), N103 = axis("n", 448, T.bn), K103 = axis("k", 112, T.bk);
export const kern103 = kernel(
  { name: "k103", tile: T, grid: [M103, N103], reduce: [K103],
    bindings: [input("a", [M103, K103], f32), input("b", [K103, N103], f32), output("c", [M103, N103], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M104 = axis("m", 512, T.bm), N104 = axis("n", 512, T.bn), K104 = axis("k", 128, T.bk);
export const kern104 = kernel(
  { name: "k104", tile: T, grid: [M104, N104], reduce: [K104],
    bindings: [input("a", [M104, K104], f32), input("b", [K104, N104], f32), output("c", [M104, N104], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M105 = axis("m", 576, T.bm), N105 = axis("n", 576, T.bn), K105 = axis("k", 144, T.bk);
export const kern105 = kernel(
  { name: "k105", tile: T, grid: [M105, N105], reduce: [K105],
    bindings: [input("a", [M105, K105], f32), input("b", [K105, N105], f32), output("c", [M105, N105], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M106 = axis("m", 640, T.bm), N106 = axis("n", 640, T.bn), K106 = axis("k", 160, T.bk);
export const kern106 = kernel(
  { name: "k106", tile: T, grid: [M106, N106], reduce: [K106],
    bindings: [input("a", [M106, K106], f32), input("b", [K106, N106], f32), output("c", [M106, N106], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M107 = axis("m", 704, T.bm), N107 = axis("n", 704, T.bn), K107 = axis("k", 176, T.bk);
export const kern107 = kernel(
  { name: "k107", tile: T, grid: [M107, N107], reduce: [K107],
    bindings: [input("a", [M107, K107], f32), input("b", [K107, N107], f32), output("c", [M107, N107], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M108 = axis("m", 768, T.bm), N108 = axis("n", 768, T.bn), K108 = axis("k", 192, T.bk);
export const kern108 = kernel(
  { name: "k108", tile: T, grid: [M108, N108], reduce: [K108],
    bindings: [input("a", [M108, K108], f32), input("b", [K108, N108], f32), output("c", [M108, N108], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M109 = axis("m", 832, T.bm), N109 = axis("n", 832, T.bn), K109 = axis("k", 208, T.bk);
export const kern109 = kernel(
  { name: "k109", tile: T, grid: [M109, N109], reduce: [K109],
    bindings: [input("a", [M109, K109], f32), input("b", [K109, N109], f32), output("c", [M109, N109], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M110 = axis("m", 896, T.bm), N110 = axis("n", 192, T.bn), K110 = axis("k", 224, T.bk);
export const kern110 = kernel(
  { name: "k110", tile: T, grid: [M110, N110], reduce: [K110],
    bindings: [input("a", [M110, K110], f32), input("b", [K110, N110], f32), output("c", [M110, N110], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M111 = axis("m", 960, T.bm), N111 = axis("n", 256, T.bn), K111 = axis("k", 240, T.bk);
export const kern111 = kernel(
  { name: "k111", tile: T, grid: [M111, N111], reduce: [K111],
    bindings: [input("a", [M111, K111], f32), input("b", [K111, N111], f32), output("c", [M111, N111], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M112 = axis("m", 128, T.bm), N112 = axis("n", 320, T.bn), K112 = axis("k", 256, T.bk);
export const kern112 = kernel(
  { name: "k112", tile: T, grid: [M112, N112], reduce: [K112],
    bindings: [input("a", [M112, K112], f32), input("b", [K112, N112], f32), output("c", [M112, N112], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M113 = axis("m", 192, T.bm), N113 = axis("n", 384, T.bn), K113 = axis("k", 272, T.bk);
export const kern113 = kernel(
  { name: "k113", tile: T, grid: [M113, N113], reduce: [K113],
    bindings: [input("a", [M113, K113], f32), input("b", [K113, N113], f32), output("c", [M113, N113], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M114 = axis("m", 256, T.bm), N114 = axis("n", 448, T.bn), K114 = axis("k", 288, T.bk);
export const kern114 = kernel(
  { name: "k114", tile: T, grid: [M114, N114], reduce: [K114],
    bindings: [input("a", [M114, K114], f32), input("b", [K114, N114], f32), output("c", [M114, N114], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M115 = axis("m", 320, T.bm), N115 = axis("n", 512, T.bn), K115 = axis("k", 304, T.bk);
export const kern115 = kernel(
  { name: "k115", tile: T, grid: [M115, N115], reduce: [K115],
    bindings: [input("a", [M115, K115], f32), input("b", [K115, N115], f32), output("c", [M115, N115], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M116 = axis("m", 384, T.bm), N116 = axis("n", 576, T.bn), K116 = axis("k", 320, T.bk);
export const kern116 = kernel(
  { name: "k116", tile: T, grid: [M116, N116], reduce: [K116],
    bindings: [input("a", [M116, K116], f32), input("b", [K116, N116], f32), output("c", [M116, N116], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M117 = axis("m", 448, T.bm), N117 = axis("n", 640, T.bn), K117 = axis("k", 336, T.bk);
export const kern117 = kernel(
  { name: "k117", tile: T, grid: [M117, N117], reduce: [K117],
    bindings: [input("a", [M117, K117], f32), input("b", [K117, N117], f32), output("c", [M117, N117], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M118 = axis("m", 512, T.bm), N118 = axis("n", 704, T.bn), K118 = axis("k", 352, T.bk);
export const kern118 = kernel(
  { name: "k118", tile: T, grid: [M118, N118], reduce: [K118],
    bindings: [input("a", [M118, K118], f32), input("b", [K118, N118], f32), output("c", [M118, N118], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M119 = axis("m", 576, T.bm), N119 = axis("n", 768, T.bn), K119 = axis("k", 368, T.bk);
export const kern119 = kernel(
  { name: "k119", tile: T, grid: [M119, N119], reduce: [K119],
    bindings: [input("a", [M119, K119], f32), input("b", [K119, N119], f32), output("c", [M119, N119], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M120 = axis("m", 640, T.bm), N120 = axis("n", 832, T.bn), K120 = axis("k", 64, T.bk);
export const kern120 = kernel(
  { name: "k120", tile: T, grid: [M120, N120], reduce: [K120],
    bindings: [input("a", [M120, K120], f32), input("b", [K120, N120], f32), output("c", [M120, N120], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M121 = axis("m", 704, T.bm), N121 = axis("n", 192, T.bn), K121 = axis("k", 80, T.bk);
export const kern121 = kernel(
  { name: "k121", tile: T, grid: [M121, N121], reduce: [K121],
    bindings: [input("a", [M121, K121], f32), input("b", [K121, N121], f32), output("c", [M121, N121], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M122 = axis("m", 768, T.bm), N122 = axis("n", 256, T.bn), K122 = axis("k", 96, T.bk);
export const kern122 = kernel(
  { name: "k122", tile: T, grid: [M122, N122], reduce: [K122],
    bindings: [input("a", [M122, K122], f32), input("b", [K122, N122], f32), output("c", [M122, N122], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M123 = axis("m", 832, T.bm), N123 = axis("n", 320, T.bn), K123 = axis("k", 112, T.bk);
export const kern123 = kernel(
  { name: "k123", tile: T, grid: [M123, N123], reduce: [K123],
    bindings: [input("a", [M123, K123], f32), input("b", [K123, N123], f32), output("c", [M123, N123], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M124 = axis("m", 896, T.bm), N124 = axis("n", 384, T.bn), K124 = axis("k", 128, T.bk);
export const kern124 = kernel(
  { name: "k124", tile: T, grid: [M124, N124], reduce: [K124],
    bindings: [input("a", [M124, K124], f32), input("b", [K124, N124], f32), output("c", [M124, N124], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M125 = axis("m", 960, T.bm), N125 = axis("n", 448, T.bn), K125 = axis("k", 144, T.bk);
export const kern125 = kernel(
  { name: "k125", tile: T, grid: [M125, N125], reduce: [K125],
    bindings: [input("a", [M125, K125], f32), input("b", [K125, N125], f32), output("c", [M125, N125], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M126 = axis("m", 128, T.bm), N126 = axis("n", 512, T.bn), K126 = axis("k", 160, T.bk);
export const kern126 = kernel(
  { name: "k126", tile: T, grid: [M126, N126], reduce: [K126],
    bindings: [input("a", [M126, K126], f32), input("b", [K126, N126], f32), output("c", [M126, N126], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M127 = axis("m", 192, T.bm), N127 = axis("n", 576, T.bn), K127 = axis("k", 176, T.bk);
export const kern127 = kernel(
  { name: "k127", tile: T, grid: [M127, N127], reduce: [K127],
    bindings: [input("a", [M127, K127], f32), input("b", [K127, N127], f32), output("c", [M127, N127], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M128 = axis("m", 256, T.bm), N128 = axis("n", 640, T.bn), K128 = axis("k", 192, T.bk);
export const kern128 = kernel(
  { name: "k128", tile: T, grid: [M128, N128], reduce: [K128],
    bindings: [input("a", [M128, K128], f32), input("b", [K128, N128], f32), output("c", [M128, N128], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M129 = axis("m", 320, T.bm), N129 = axis("n", 704, T.bn), K129 = axis("k", 208, T.bk);
export const kern129 = kernel(
  { name: "k129", tile: T, grid: [M129, N129], reduce: [K129],
    bindings: [input("a", [M129, K129], f32), input("b", [K129, N129], f32), output("c", [M129, N129], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M130 = axis("m", 384, T.bm), N130 = axis("n", 768, T.bn), K130 = axis("k", 224, T.bk);
export const kern130 = kernel(
  { name: "k130", tile: T, grid: [M130, N130], reduce: [K130],
    bindings: [input("a", [M130, K130], f32), input("b", [K130, N130], f32), output("c", [M130, N130], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M131 = axis("m", 448, T.bm), N131 = axis("n", 832, T.bn), K131 = axis("k", 240, T.bk);
export const kern131 = kernel(
  { name: "k131", tile: T, grid: [M131, N131], reduce: [K131],
    bindings: [input("a", [M131, K131], f32), input("b", [K131, N131], f32), output("c", [M131, N131], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M132 = axis("m", 512, T.bm), N132 = axis("n", 192, T.bn), K132 = axis("k", 256, T.bk);
export const kern132 = kernel(
  { name: "k132", tile: T, grid: [M132, N132], reduce: [K132],
    bindings: [input("a", [M132, K132], f32), input("b", [K132, N132], f32), output("c", [M132, N132], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M133 = axis("m", 576, T.bm), N133 = axis("n", 256, T.bn), K133 = axis("k", 272, T.bk);
export const kern133 = kernel(
  { name: "k133", tile: T, grid: [M133, N133], reduce: [K133],
    bindings: [input("a", [M133, K133], f32), input("b", [K133, N133], f32), output("c", [M133, N133], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M134 = axis("m", 640, T.bm), N134 = axis("n", 320, T.bn), K134 = axis("k", 288, T.bk);
export const kern134 = kernel(
  { name: "k134", tile: T, grid: [M134, N134], reduce: [K134],
    bindings: [input("a", [M134, K134], f32), input("b", [K134, N134], f32), output("c", [M134, N134], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M135 = axis("m", 704, T.bm), N135 = axis("n", 384, T.bn), K135 = axis("k", 304, T.bk);
export const kern135 = kernel(
  { name: "k135", tile: T, grid: [M135, N135], reduce: [K135],
    bindings: [input("a", [M135, K135], f32), input("b", [K135, N135], f32), output("c", [M135, N135], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M136 = axis("m", 768, T.bm), N136 = axis("n", 448, T.bn), K136 = axis("k", 320, T.bk);
export const kern136 = kernel(
  { name: "k136", tile: T, grid: [M136, N136], reduce: [K136],
    bindings: [input("a", [M136, K136], f32), input("b", [K136, N136], f32), output("c", [M136, N136], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M137 = axis("m", 832, T.bm), N137 = axis("n", 512, T.bn), K137 = axis("k", 336, T.bk);
export const kern137 = kernel(
  { name: "k137", tile: T, grid: [M137, N137], reduce: [K137],
    bindings: [input("a", [M137, K137], f32), input("b", [K137, N137], f32), output("c", [M137, N137], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M138 = axis("m", 896, T.bm), N138 = axis("n", 576, T.bn), K138 = axis("k", 352, T.bk);
export const kern138 = kernel(
  { name: "k138", tile: T, grid: [M138, N138], reduce: [K138],
    bindings: [input("a", [M138, K138], f32), input("b", [K138, N138], f32), output("c", [M138, N138], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M139 = axis("m", 960, T.bm), N139 = axis("n", 640, T.bn), K139 = axis("k", 368, T.bk);
export const kern139 = kernel(
  { name: "k139", tile: T, grid: [M139, N139], reduce: [K139],
    bindings: [input("a", [M139, K139], f32), input("b", [K139, N139], f32), output("c", [M139, N139], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M140 = axis("m", 128, T.bm), N140 = axis("n", 704, T.bn), K140 = axis("k", 64, T.bk);
export const kern140 = kernel(
  { name: "k140", tile: T, grid: [M140, N140], reduce: [K140],
    bindings: [input("a", [M140, K140], f32), input("b", [K140, N140], f32), output("c", [M140, N140], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M141 = axis("m", 192, T.bm), N141 = axis("n", 768, T.bn), K141 = axis("k", 80, T.bk);
export const kern141 = kernel(
  { name: "k141", tile: T, grid: [M141, N141], reduce: [K141],
    bindings: [input("a", [M141, K141], f32), input("b", [K141, N141], f32), output("c", [M141, N141], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M142 = axis("m", 256, T.bm), N142 = axis("n", 832, T.bn), K142 = axis("k", 96, T.bk);
export const kern142 = kernel(
  { name: "k142", tile: T, grid: [M142, N142], reduce: [K142],
    bindings: [input("a", [M142, K142], f32), input("b", [K142, N142], f32), output("c", [M142, N142], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M143 = axis("m", 320, T.bm), N143 = axis("n", 192, T.bn), K143 = axis("k", 112, T.bk);
export const kern143 = kernel(
  { name: "k143", tile: T, grid: [M143, N143], reduce: [K143],
    bindings: [input("a", [M143, K143], f32), input("b", [K143, N143], f32), output("c", [M143, N143], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M144 = axis("m", 384, T.bm), N144 = axis("n", 256, T.bn), K144 = axis("k", 128, T.bk);
export const kern144 = kernel(
  { name: "k144", tile: T, grid: [M144, N144], reduce: [K144],
    bindings: [input("a", [M144, K144], f32), input("b", [K144, N144], f32), output("c", [M144, N144], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M145 = axis("m", 448, T.bm), N145 = axis("n", 320, T.bn), K145 = axis("k", 144, T.bk);
export const kern145 = kernel(
  { name: "k145", tile: T, grid: [M145, N145], reduce: [K145],
    bindings: [input("a", [M145, K145], f32), input("b", [K145, N145], f32), output("c", [M145, N145], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M146 = axis("m", 512, T.bm), N146 = axis("n", 384, T.bn), K146 = axis("k", 160, T.bk);
export const kern146 = kernel(
  { name: "k146", tile: T, grid: [M146, N146], reduce: [K146],
    bindings: [input("a", [M146, K146], f32), input("b", [K146, N146], f32), output("c", [M146, N146], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M147 = axis("m", 576, T.bm), N147 = axis("n", 448, T.bn), K147 = axis("k", 176, T.bk);
export const kern147 = kernel(
  { name: "k147", tile: T, grid: [M147, N147], reduce: [K147],
    bindings: [input("a", [M147, K147], f32), input("b", [K147, N147], f32), output("c", [M147, N147], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M148 = axis("m", 640, T.bm), N148 = axis("n", 512, T.bn), K148 = axis("k", 192, T.bk);
export const kern148 = kernel(
  { name: "k148", tile: T, grid: [M148, N148], reduce: [K148],
    bindings: [input("a", [M148, K148], f32), input("b", [K148, N148], f32), output("c", [M148, N148], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M149 = axis("m", 704, T.bm), N149 = axis("n", 576, T.bn), K149 = axis("k", 208, T.bk);
export const kern149 = kernel(
  { name: "k149", tile: T, grid: [M149, N149], reduce: [K149],
    bindings: [input("a", [M149, K149], f32), input("b", [K149, N149], f32), output("c", [M149, N149], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M150 = axis("m", 768, T.bm), N150 = axis("n", 640, T.bn), K150 = axis("k", 224, T.bk);
export const kern150 = kernel(
  { name: "k150", tile: T, grid: [M150, N150], reduce: [K150],
    bindings: [input("a", [M150, K150], f32), input("b", [K150, N150], f32), output("c", [M150, N150], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M151 = axis("m", 832, T.bm), N151 = axis("n", 704, T.bn), K151 = axis("k", 240, T.bk);
export const kern151 = kernel(
  { name: "k151", tile: T, grid: [M151, N151], reduce: [K151],
    bindings: [input("a", [M151, K151], f32), input("b", [K151, N151], f32), output("c", [M151, N151], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M152 = axis("m", 896, T.bm), N152 = axis("n", 768, T.bn), K152 = axis("k", 256, T.bk);
export const kern152 = kernel(
  { name: "k152", tile: T, grid: [M152, N152], reduce: [K152],
    bindings: [input("a", [M152, K152], f32), input("b", [K152, N152], f32), output("c", [M152, N152], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M153 = axis("m", 960, T.bm), N153 = axis("n", 832, T.bn), K153 = axis("k", 272, T.bk);
export const kern153 = kernel(
  { name: "k153", tile: T, grid: [M153, N153], reduce: [K153],
    bindings: [input("a", [M153, K153], f32), input("b", [K153, N153], f32), output("c", [M153, N153], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M154 = axis("m", 128, T.bm), N154 = axis("n", 192, T.bn), K154 = axis("k", 288, T.bk);
export const kern154 = kernel(
  { name: "k154", tile: T, grid: [M154, N154], reduce: [K154],
    bindings: [input("a", [M154, K154], f32), input("b", [K154, N154], f32), output("c", [M154, N154], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M155 = axis("m", 192, T.bm), N155 = axis("n", 256, T.bn), K155 = axis("k", 304, T.bk);
export const kern155 = kernel(
  { name: "k155", tile: T, grid: [M155, N155], reduce: [K155],
    bindings: [input("a", [M155, K155], f32), input("b", [K155, N155], f32), output("c", [M155, N155], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M156 = axis("m", 256, T.bm), N156 = axis("n", 320, T.bn), K156 = axis("k", 320, T.bk);
export const kern156 = kernel(
  { name: "k156", tile: T, grid: [M156, N156], reduce: [K156],
    bindings: [input("a", [M156, K156], f32), input("b", [K156, N156], f32), output("c", [M156, N156], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M157 = axis("m", 320, T.bm), N157 = axis("n", 384, T.bn), K157 = axis("k", 336, T.bk);
export const kern157 = kernel(
  { name: "k157", tile: T, grid: [M157, N157], reduce: [K157],
    bindings: [input("a", [M157, K157], f32), input("b", [K157, N157], f32), output("c", [M157, N157], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M158 = axis("m", 384, T.bm), N158 = axis("n", 448, T.bn), K158 = axis("k", 352, T.bk);
export const kern158 = kernel(
  { name: "k158", tile: T, grid: [M158, N158], reduce: [K158],
    bindings: [input("a", [M158, K158], f32), input("b", [K158, N158], f32), output("c", [M158, N158], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M159 = axis("m", 448, T.bm), N159 = axis("n", 512, T.bn), K159 = axis("k", 368, T.bk);
export const kern159 = kernel(
  { name: "k159", tile: T, grid: [M159, N159], reduce: [K159],
    bindings: [input("a", [M159, K159], f32), input("b", [K159, N159], f32), output("c", [M159, N159], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M160 = axis("m", 512, T.bm), N160 = axis("n", 576, T.bn), K160 = axis("k", 64, T.bk);
export const kern160 = kernel(
  { name: "k160", tile: T, grid: [M160, N160], reduce: [K160],
    bindings: [input("a", [M160, K160], f32), input("b", [K160, N160], f32), output("c", [M160, N160], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M161 = axis("m", 576, T.bm), N161 = axis("n", 640, T.bn), K161 = axis("k", 80, T.bk);
export const kern161 = kernel(
  { name: "k161", tile: T, grid: [M161, N161], reduce: [K161],
    bindings: [input("a", [M161, K161], f32), input("b", [K161, N161], f32), output("c", [M161, N161], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M162 = axis("m", 640, T.bm), N162 = axis("n", 704, T.bn), K162 = axis("k", 96, T.bk);
export const kern162 = kernel(
  { name: "k162", tile: T, grid: [M162, N162], reduce: [K162],
    bindings: [input("a", [M162, K162], f32), input("b", [K162, N162], f32), output("c", [M162, N162], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M163 = axis("m", 704, T.bm), N163 = axis("n", 768, T.bn), K163 = axis("k", 112, T.bk);
export const kern163 = kernel(
  { name: "k163", tile: T, grid: [M163, N163], reduce: [K163],
    bindings: [input("a", [M163, K163], f32), input("b", [K163, N163], f32), output("c", [M163, N163], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M164 = axis("m", 768, T.bm), N164 = axis("n", 832, T.bn), K164 = axis("k", 128, T.bk);
export const kern164 = kernel(
  { name: "k164", tile: T, grid: [M164, N164], reduce: [K164],
    bindings: [input("a", [M164, K164], f32), input("b", [K164, N164], f32), output("c", [M164, N164], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M165 = axis("m", 832, T.bm), N165 = axis("n", 192, T.bn), K165 = axis("k", 144, T.bk);
export const kern165 = kernel(
  { name: "k165", tile: T, grid: [M165, N165], reduce: [K165],
    bindings: [input("a", [M165, K165], f32), input("b", [K165, N165], f32), output("c", [M165, N165], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M166 = axis("m", 896, T.bm), N166 = axis("n", 256, T.bn), K166 = axis("k", 160, T.bk);
export const kern166 = kernel(
  { name: "k166", tile: T, grid: [M166, N166], reduce: [K166],
    bindings: [input("a", [M166, K166], f32), input("b", [K166, N166], f32), output("c", [M166, N166], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M167 = axis("m", 960, T.bm), N167 = axis("n", 320, T.bn), K167 = axis("k", 176, T.bk);
export const kern167 = kernel(
  { name: "k167", tile: T, grid: [M167, N167], reduce: [K167],
    bindings: [input("a", [M167, K167], f32), input("b", [K167, N167], f32), output("c", [M167, N167], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M168 = axis("m", 128, T.bm), N168 = axis("n", 384, T.bn), K168 = axis("k", 192, T.bk);
export const kern168 = kernel(
  { name: "k168", tile: T, grid: [M168, N168], reduce: [K168],
    bindings: [input("a", [M168, K168], f32), input("b", [K168, N168], f32), output("c", [M168, N168], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M169 = axis("m", 192, T.bm), N169 = axis("n", 448, T.bn), K169 = axis("k", 208, T.bk);
export const kern169 = kernel(
  { name: "k169", tile: T, grid: [M169, N169], reduce: [K169],
    bindings: [input("a", [M169, K169], f32), input("b", [K169, N169], f32), output("c", [M169, N169], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M170 = axis("m", 256, T.bm), N170 = axis("n", 512, T.bn), K170 = axis("k", 224, T.bk);
export const kern170 = kernel(
  { name: "k170", tile: T, grid: [M170, N170], reduce: [K170],
    bindings: [input("a", [M170, K170], f32), input("b", [K170, N170], f32), output("c", [M170, N170], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M171 = axis("m", 320, T.bm), N171 = axis("n", 576, T.bn), K171 = axis("k", 240, T.bk);
export const kern171 = kernel(
  { name: "k171", tile: T, grid: [M171, N171], reduce: [K171],
    bindings: [input("a", [M171, K171], f32), input("b", [K171, N171], f32), output("c", [M171, N171], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M172 = axis("m", 384, T.bm), N172 = axis("n", 640, T.bn), K172 = axis("k", 256, T.bk);
export const kern172 = kernel(
  { name: "k172", tile: T, grid: [M172, N172], reduce: [K172],
    bindings: [input("a", [M172, K172], f32), input("b", [K172, N172], f32), output("c", [M172, N172], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M173 = axis("m", 448, T.bm), N173 = axis("n", 704, T.bn), K173 = axis("k", 272, T.bk);
export const kern173 = kernel(
  { name: "k173", tile: T, grid: [M173, N173], reduce: [K173],
    bindings: [input("a", [M173, K173], f32), input("b", [K173, N173], f32), output("c", [M173, N173], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M174 = axis("m", 512, T.bm), N174 = axis("n", 768, T.bn), K174 = axis("k", 288, T.bk);
export const kern174 = kernel(
  { name: "k174", tile: T, grid: [M174, N174], reduce: [K174],
    bindings: [input("a", [M174, K174], f32), input("b", [K174, N174], f32), output("c", [M174, N174], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M175 = axis("m", 576, T.bm), N175 = axis("n", 832, T.bn), K175 = axis("k", 304, T.bk);
export const kern175 = kernel(
  { name: "k175", tile: T, grid: [M175, N175], reduce: [K175],
    bindings: [input("a", [M175, K175], f32), input("b", [K175, N175], f32), output("c", [M175, N175], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M176 = axis("m", 640, T.bm), N176 = axis("n", 192, T.bn), K176 = axis("k", 320, T.bk);
export const kern176 = kernel(
  { name: "k176", tile: T, grid: [M176, N176], reduce: [K176],
    bindings: [input("a", [M176, K176], f32), input("b", [K176, N176], f32), output("c", [M176, N176], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M177 = axis("m", 704, T.bm), N177 = axis("n", 256, T.bn), K177 = axis("k", 336, T.bk);
export const kern177 = kernel(
  { name: "k177", tile: T, grid: [M177, N177], reduce: [K177],
    bindings: [input("a", [M177, K177], f32), input("b", [K177, N177], f32), output("c", [M177, N177], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M178 = axis("m", 768, T.bm), N178 = axis("n", 320, T.bn), K178 = axis("k", 352, T.bk);
export const kern178 = kernel(
  { name: "k178", tile: T, grid: [M178, N178], reduce: [K178],
    bindings: [input("a", [M178, K178], f32), input("b", [K178, N178], f32), output("c", [M178, N178], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M179 = axis("m", 832, T.bm), N179 = axis("n", 384, T.bn), K179 = axis("k", 368, T.bk);
export const kern179 = kernel(
  { name: "k179", tile: T, grid: [M179, N179], reduce: [K179],
    bindings: [input("a", [M179, K179], f32), input("b", [K179, N179], f32), output("c", [M179, N179], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M180 = axis("m", 896, T.bm), N180 = axis("n", 448, T.bn), K180 = axis("k", 64, T.bk);
export const kern180 = kernel(
  { name: "k180", tile: T, grid: [M180, N180], reduce: [K180],
    bindings: [input("a", [M180, K180], f32), input("b", [K180, N180], f32), output("c", [M180, N180], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M181 = axis("m", 960, T.bm), N181 = axis("n", 512, T.bn), K181 = axis("k", 80, T.bk);
export const kern181 = kernel(
  { name: "k181", tile: T, grid: [M181, N181], reduce: [K181],
    bindings: [input("a", [M181, K181], f32), input("b", [K181, N181], f32), output("c", [M181, N181], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M182 = axis("m", 128, T.bm), N182 = axis("n", 576, T.bn), K182 = axis("k", 96, T.bk);
export const kern182 = kernel(
  { name: "k182", tile: T, grid: [M182, N182], reduce: [K182],
    bindings: [input("a", [M182, K182], f32), input("b", [K182, N182], f32), output("c", [M182, N182], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M183 = axis("m", 192, T.bm), N183 = axis("n", 640, T.bn), K183 = axis("k", 112, T.bk);
export const kern183 = kernel(
  { name: "k183", tile: T, grid: [M183, N183], reduce: [K183],
    bindings: [input("a", [M183, K183], f32), input("b", [K183, N183], f32), output("c", [M183, N183], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M184 = axis("m", 256, T.bm), N184 = axis("n", 704, T.bn), K184 = axis("k", 128, T.bk);
export const kern184 = kernel(
  { name: "k184", tile: T, grid: [M184, N184], reduce: [K184],
    bindings: [input("a", [M184, K184], f32), input("b", [K184, N184], f32), output("c", [M184, N184], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M185 = axis("m", 320, T.bm), N185 = axis("n", 768, T.bn), K185 = axis("k", 144, T.bk);
export const kern185 = kernel(
  { name: "k185", tile: T, grid: [M185, N185], reduce: [K185],
    bindings: [input("a", [M185, K185], f32), input("b", [K185, N185], f32), output("c", [M185, N185], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M186 = axis("m", 384, T.bm), N186 = axis("n", 832, T.bn), K186 = axis("k", 160, T.bk);
export const kern186 = kernel(
  { name: "k186", tile: T, grid: [M186, N186], reduce: [K186],
    bindings: [input("a", [M186, K186], f32), input("b", [K186, N186], f32), output("c", [M186, N186], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M187 = axis("m", 448, T.bm), N187 = axis("n", 192, T.bn), K187 = axis("k", 176, T.bk);
export const kern187 = kernel(
  { name: "k187", tile: T, grid: [M187, N187], reduce: [K187],
    bindings: [input("a", [M187, K187], f32), input("b", [K187, N187], f32), output("c", [M187, N187], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M188 = axis("m", 512, T.bm), N188 = axis("n", 256, T.bn), K188 = axis("k", 192, T.bk);
export const kern188 = kernel(
  { name: "k188", tile: T, grid: [M188, N188], reduce: [K188],
    bindings: [input("a", [M188, K188], f32), input("b", [K188, N188], f32), output("c", [M188, N188], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M189 = axis("m", 576, T.bm), N189 = axis("n", 320, T.bn), K189 = axis("k", 208, T.bk);
export const kern189 = kernel(
  { name: "k189", tile: T, grid: [M189, N189], reduce: [K189],
    bindings: [input("a", [M189, K189], f32), input("b", [K189, N189], f32), output("c", [M189, N189], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M190 = axis("m", 640, T.bm), N190 = axis("n", 384, T.bn), K190 = axis("k", 224, T.bk);
export const kern190 = kernel(
  { name: "k190", tile: T, grid: [M190, N190], reduce: [K190],
    bindings: [input("a", [M190, K190], f32), input("b", [K190, N190], f32), output("c", [M190, N190], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M191 = axis("m", 704, T.bm), N191 = axis("n", 448, T.bn), K191 = axis("k", 240, T.bk);
export const kern191 = kernel(
  { name: "k191", tile: T, grid: [M191, N191], reduce: [K191],
    bindings: [input("a", [M191, K191], f32), input("b", [K191, N191], f32), output("c", [M191, N191], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M192 = axis("m", 768, T.bm), N192 = axis("n", 512, T.bn), K192 = axis("k", 256, T.bk);
export const kern192 = kernel(
  { name: "k192", tile: T, grid: [M192, N192], reduce: [K192],
    bindings: [input("a", [M192, K192], f32), input("b", [K192, N192], f32), output("c", [M192, N192], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M193 = axis("m", 832, T.bm), N193 = axis("n", 576, T.bn), K193 = axis("k", 272, T.bk);
export const kern193 = kernel(
  { name: "k193", tile: T, grid: [M193, N193], reduce: [K193],
    bindings: [input("a", [M193, K193], f32), input("b", [K193, N193], f32), output("c", [M193, N193], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M194 = axis("m", 896, T.bm), N194 = axis("n", 640, T.bn), K194 = axis("k", 288, T.bk);
export const kern194 = kernel(
  { name: "k194", tile: T, grid: [M194, N194], reduce: [K194],
    bindings: [input("a", [M194, K194], f32), input("b", [K194, N194], f32), output("c", [M194, N194], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M195 = axis("m", 960, T.bm), N195 = axis("n", 704, T.bn), K195 = axis("k", 304, T.bk);
export const kern195 = kernel(
  { name: "k195", tile: T, grid: [M195, N195], reduce: [K195],
    bindings: [input("a", [M195, K195], f32), input("b", [K195, N195], f32), output("c", [M195, N195], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M196 = axis("m", 128, T.bm), N196 = axis("n", 768, T.bn), K196 = axis("k", 320, T.bk);
export const kern196 = kernel(
  { name: "k196", tile: T, grid: [M196, N196], reduce: [K196],
    bindings: [input("a", [M196, K196], f32), input("b", [K196, N196], f32), output("c", [M196, N196], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M197 = axis("m", 192, T.bm), N197 = axis("n", 832, T.bn), K197 = axis("k", 336, T.bk);
export const kern197 = kernel(
  { name: "k197", tile: T, grid: [M197, N197], reduce: [K197],
    bindings: [input("a", [M197, K197], f32), input("b", [K197, N197], f32), output("c", [M197, N197], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M198 = axis("m", 256, T.bm), N198 = axis("n", 192, T.bn), K198 = axis("k", 352, T.bk);
export const kern198 = kernel(
  { name: "k198", tile: T, grid: [M198, N198], reduce: [K198],
    bindings: [input("a", [M198, K198], f32), input("b", [K198, N198], f32), output("c", [M198, N198], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M199 = axis("m", 320, T.bm), N199 = axis("n", 256, T.bn), K199 = axis("k", 368, T.bk);
export const kern199 = kernel(
  { name: "k199", tile: T, grid: [M199, N199], reduce: [K199],
    bindings: [input("a", [M199, K199], f32), input("b", [K199, N199], f32), output("c", [M199, N199], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M200 = axis("m", 384, T.bm), N200 = axis("n", 320, T.bn), K200 = axis("k", 64, T.bk);
export const kern200 = kernel(
  { name: "k200", tile: T, grid: [M200, N200], reduce: [K200],
    bindings: [input("a", [M200, K200], f32), input("b", [K200, N200], f32), output("c", [M200, N200], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M201 = axis("m", 448, T.bm), N201 = axis("n", 384, T.bn), K201 = axis("k", 80, T.bk);
export const kern201 = kernel(
  { name: "k201", tile: T, grid: [M201, N201], reduce: [K201],
    bindings: [input("a", [M201, K201], f32), input("b", [K201, N201], f32), output("c", [M201, N201], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M202 = axis("m", 512, T.bm), N202 = axis("n", 448, T.bn), K202 = axis("k", 96, T.bk);
export const kern202 = kernel(
  { name: "k202", tile: T, grid: [M202, N202], reduce: [K202],
    bindings: [input("a", [M202, K202], f32), input("b", [K202, N202], f32), output("c", [M202, N202], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M203 = axis("m", 576, T.bm), N203 = axis("n", 512, T.bn), K203 = axis("k", 112, T.bk);
export const kern203 = kernel(
  { name: "k203", tile: T, grid: [M203, N203], reduce: [K203],
    bindings: [input("a", [M203, K203], f32), input("b", [K203, N203], f32), output("c", [M203, N203], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M204 = axis("m", 640, T.bm), N204 = axis("n", 576, T.bn), K204 = axis("k", 128, T.bk);
export const kern204 = kernel(
  { name: "k204", tile: T, grid: [M204, N204], reduce: [K204],
    bindings: [input("a", [M204, K204], f32), input("b", [K204, N204], f32), output("c", [M204, N204], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M205 = axis("m", 704, T.bm), N205 = axis("n", 640, T.bn), K205 = axis("k", 144, T.bk);
export const kern205 = kernel(
  { name: "k205", tile: T, grid: [M205, N205], reduce: [K205],
    bindings: [input("a", [M205, K205], f32), input("b", [K205, N205], f32), output("c", [M205, N205], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M206 = axis("m", 768, T.bm), N206 = axis("n", 704, T.bn), K206 = axis("k", 160, T.bk);
export const kern206 = kernel(
  { name: "k206", tile: T, grid: [M206, N206], reduce: [K206],
    bindings: [input("a", [M206, K206], f32), input("b", [K206, N206], f32), output("c", [M206, N206], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M207 = axis("m", 832, T.bm), N207 = axis("n", 768, T.bn), K207 = axis("k", 176, T.bk);
export const kern207 = kernel(
  { name: "k207", tile: T, grid: [M207, N207], reduce: [K207],
    bindings: [input("a", [M207, K207], f32), input("b", [K207, N207], f32), output("c", [M207, N207], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M208 = axis("m", 896, T.bm), N208 = axis("n", 832, T.bn), K208 = axis("k", 192, T.bk);
export const kern208 = kernel(
  { name: "k208", tile: T, grid: [M208, N208], reduce: [K208],
    bindings: [input("a", [M208, K208], f32), input("b", [K208, N208], f32), output("c", [M208, N208], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M209 = axis("m", 960, T.bm), N209 = axis("n", 192, T.bn), K209 = axis("k", 208, T.bk);
export const kern209 = kernel(
  { name: "k209", tile: T, grid: [M209, N209], reduce: [K209],
    bindings: [input("a", [M209, K209], f32), input("b", [K209, N209], f32), output("c", [M209, N209], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M210 = axis("m", 128, T.bm), N210 = axis("n", 256, T.bn), K210 = axis("k", 224, T.bk);
export const kern210 = kernel(
  { name: "k210", tile: T, grid: [M210, N210], reduce: [K210],
    bindings: [input("a", [M210, K210], f32), input("b", [K210, N210], f32), output("c", [M210, N210], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M211 = axis("m", 192, T.bm), N211 = axis("n", 320, T.bn), K211 = axis("k", 240, T.bk);
export const kern211 = kernel(
  { name: "k211", tile: T, grid: [M211, N211], reduce: [K211],
    bindings: [input("a", [M211, K211], f32), input("b", [K211, N211], f32), output("c", [M211, N211], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M212 = axis("m", 256, T.bm), N212 = axis("n", 384, T.bn), K212 = axis("k", 256, T.bk);
export const kern212 = kernel(
  { name: "k212", tile: T, grid: [M212, N212], reduce: [K212],
    bindings: [input("a", [M212, K212], f32), input("b", [K212, N212], f32), output("c", [M212, N212], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M213 = axis("m", 320, T.bm), N213 = axis("n", 448, T.bn), K213 = axis("k", 272, T.bk);
export const kern213 = kernel(
  { name: "k213", tile: T, grid: [M213, N213], reduce: [K213],
    bindings: [input("a", [M213, K213], f32), input("b", [K213, N213], f32), output("c", [M213, N213], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M214 = axis("m", 384, T.bm), N214 = axis("n", 512, T.bn), K214 = axis("k", 288, T.bk);
export const kern214 = kernel(
  { name: "k214", tile: T, grid: [M214, N214], reduce: [K214],
    bindings: [input("a", [M214, K214], f32), input("b", [K214, N214], f32), output("c", [M214, N214], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M215 = axis("m", 448, T.bm), N215 = axis("n", 576, T.bn), K215 = axis("k", 304, T.bk);
export const kern215 = kernel(
  { name: "k215", tile: T, grid: [M215, N215], reduce: [K215],
    bindings: [input("a", [M215, K215], f32), input("b", [K215, N215], f32), output("c", [M215, N215], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M216 = axis("m", 512, T.bm), N216 = axis("n", 640, T.bn), K216 = axis("k", 320, T.bk);
export const kern216 = kernel(
  { name: "k216", tile: T, grid: [M216, N216], reduce: [K216],
    bindings: [input("a", [M216, K216], f32), input("b", [K216, N216], f32), output("c", [M216, N216], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M217 = axis("m", 576, T.bm), N217 = axis("n", 704, T.bn), K217 = axis("k", 336, T.bk);
export const kern217 = kernel(
  { name: "k217", tile: T, grid: [M217, N217], reduce: [K217],
    bindings: [input("a", [M217, K217], f32), input("b", [K217, N217], f32), output("c", [M217, N217], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M218 = axis("m", 640, T.bm), N218 = axis("n", 768, T.bn), K218 = axis("k", 352, T.bk);
export const kern218 = kernel(
  { name: "k218", tile: T, grid: [M218, N218], reduce: [K218],
    bindings: [input("a", [M218, K218], f32), input("b", [K218, N218], f32), output("c", [M218, N218], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M219 = axis("m", 704, T.bm), N219 = axis("n", 832, T.bn), K219 = axis("k", 368, T.bk);
export const kern219 = kernel(
  { name: "k219", tile: T, grid: [M219, N219], reduce: [K219],
    bindings: [input("a", [M219, K219], f32), input("b", [K219, N219], f32), output("c", [M219, N219], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M220 = axis("m", 768, T.bm), N220 = axis("n", 192, T.bn), K220 = axis("k", 64, T.bk);
export const kern220 = kernel(
  { name: "k220", tile: T, grid: [M220, N220], reduce: [K220],
    bindings: [input("a", [M220, K220], f32), input("b", [K220, N220], f32), output("c", [M220, N220], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M221 = axis("m", 832, T.bm), N221 = axis("n", 256, T.bn), K221 = axis("k", 80, T.bk);
export const kern221 = kernel(
  { name: "k221", tile: T, grid: [M221, N221], reduce: [K221],
    bindings: [input("a", [M221, K221], f32), input("b", [K221, N221], f32), output("c", [M221, N221], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M222 = axis("m", 896, T.bm), N222 = axis("n", 320, T.bn), K222 = axis("k", 96, T.bk);
export const kern222 = kernel(
  { name: "k222", tile: T, grid: [M222, N222], reduce: [K222],
    bindings: [input("a", [M222, K222], f32), input("b", [K222, N222], f32), output("c", [M222, N222], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M223 = axis("m", 960, T.bm), N223 = axis("n", 384, T.bn), K223 = axis("k", 112, T.bk);
export const kern223 = kernel(
  { name: "k223", tile: T, grid: [M223, N223], reduce: [K223],
    bindings: [input("a", [M223, K223], f32), input("b", [K223, N223], f32), output("c", [M223, N223], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M224 = axis("m", 128, T.bm), N224 = axis("n", 448, T.bn), K224 = axis("k", 128, T.bk);
export const kern224 = kernel(
  { name: "k224", tile: T, grid: [M224, N224], reduce: [K224],
    bindings: [input("a", [M224, K224], f32), input("b", [K224, N224], f32), output("c", [M224, N224], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M225 = axis("m", 192, T.bm), N225 = axis("n", 512, T.bn), K225 = axis("k", 144, T.bk);
export const kern225 = kernel(
  { name: "k225", tile: T, grid: [M225, N225], reduce: [K225],
    bindings: [input("a", [M225, K225], f32), input("b", [K225, N225], f32), output("c", [M225, N225], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M226 = axis("m", 256, T.bm), N226 = axis("n", 576, T.bn), K226 = axis("k", 160, T.bk);
export const kern226 = kernel(
  { name: "k226", tile: T, grid: [M226, N226], reduce: [K226],
    bindings: [input("a", [M226, K226], f32), input("b", [K226, N226], f32), output("c", [M226, N226], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M227 = axis("m", 320, T.bm), N227 = axis("n", 640, T.bn), K227 = axis("k", 176, T.bk);
export const kern227 = kernel(
  { name: "k227", tile: T, grid: [M227, N227], reduce: [K227],
    bindings: [input("a", [M227, K227], f32), input("b", [K227, N227], f32), output("c", [M227, N227], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M228 = axis("m", 384, T.bm), N228 = axis("n", 704, T.bn), K228 = axis("k", 192, T.bk);
export const kern228 = kernel(
  { name: "k228", tile: T, grid: [M228, N228], reduce: [K228],
    bindings: [input("a", [M228, K228], f32), input("b", [K228, N228], f32), output("c", [M228, N228], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M229 = axis("m", 448, T.bm), N229 = axis("n", 768, T.bn), K229 = axis("k", 208, T.bk);
export const kern229 = kernel(
  { name: "k229", tile: T, grid: [M229, N229], reduce: [K229],
    bindings: [input("a", [M229, K229], f32), input("b", [K229, N229], f32), output("c", [M229, N229], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M230 = axis("m", 512, T.bm), N230 = axis("n", 832, T.bn), K230 = axis("k", 224, T.bk);
export const kern230 = kernel(
  { name: "k230", tile: T, grid: [M230, N230], reduce: [K230],
    bindings: [input("a", [M230, K230], f32), input("b", [K230, N230], f32), output("c", [M230, N230], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M231 = axis("m", 576, T.bm), N231 = axis("n", 192, T.bn), K231 = axis("k", 240, T.bk);
export const kern231 = kernel(
  { name: "k231", tile: T, grid: [M231, N231], reduce: [K231],
    bindings: [input("a", [M231, K231], f32), input("b", [K231, N231], f32), output("c", [M231, N231], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M232 = axis("m", 640, T.bm), N232 = axis("n", 256, T.bn), K232 = axis("k", 256, T.bk);
export const kern232 = kernel(
  { name: "k232", tile: T, grid: [M232, N232], reduce: [K232],
    bindings: [input("a", [M232, K232], f32), input("b", [K232, N232], f32), output("c", [M232, N232], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M233 = axis("m", 704, T.bm), N233 = axis("n", 320, T.bn), K233 = axis("k", 272, T.bk);
export const kern233 = kernel(
  { name: "k233", tile: T, grid: [M233, N233], reduce: [K233],
    bindings: [input("a", [M233, K233], f32), input("b", [K233, N233], f32), output("c", [M233, N233], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M234 = axis("m", 768, T.bm), N234 = axis("n", 384, T.bn), K234 = axis("k", 288, T.bk);
export const kern234 = kernel(
  { name: "k234", tile: T, grid: [M234, N234], reduce: [K234],
    bindings: [input("a", [M234, K234], f32), input("b", [K234, N234], f32), output("c", [M234, N234], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M235 = axis("m", 832, T.bm), N235 = axis("n", 448, T.bn), K235 = axis("k", 304, T.bk);
export const kern235 = kernel(
  { name: "k235", tile: T, grid: [M235, N235], reduce: [K235],
    bindings: [input("a", [M235, K235], f32), input("b", [K235, N235], f32), output("c", [M235, N235], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M236 = axis("m", 896, T.bm), N236 = axis("n", 512, T.bn), K236 = axis("k", 320, T.bk);
export const kern236 = kernel(
  { name: "k236", tile: T, grid: [M236, N236], reduce: [K236],
    bindings: [input("a", [M236, K236], f32), input("b", [K236, N236], f32), output("c", [M236, N236], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M237 = axis("m", 960, T.bm), N237 = axis("n", 576, T.bn), K237 = axis("k", 336, T.bk);
export const kern237 = kernel(
  { name: "k237", tile: T, grid: [M237, N237], reduce: [K237],
    bindings: [input("a", [M237, K237], f32), input("b", [K237, N237], f32), output("c", [M237, N237], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M238 = axis("m", 128, T.bm), N238 = axis("n", 640, T.bn), K238 = axis("k", 352, T.bk);
export const kern238 = kernel(
  { name: "k238", tile: T, grid: [M238, N238], reduce: [K238],
    bindings: [input("a", [M238, K238], f32), input("b", [K238, N238], f32), output("c", [M238, N238], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M239 = axis("m", 192, T.bm), N239 = axis("n", 704, T.bn), K239 = axis("k", 368, T.bk);
export const kern239 = kernel(
  { name: "k239", tile: T, grid: [M239, N239], reduce: [K239],
    bindings: [input("a", [M239, K239], f32), input("b", [K239, N239], f32), output("c", [M239, N239], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M240 = axis("m", 256, T.bm), N240 = axis("n", 768, T.bn), K240 = axis("k", 64, T.bk);
export const kern240 = kernel(
  { name: "k240", tile: T, grid: [M240, N240], reduce: [K240],
    bindings: [input("a", [M240, K240], f32), input("b", [K240, N240], f32), output("c", [M240, N240], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M241 = axis("m", 320, T.bm), N241 = axis("n", 832, T.bn), K241 = axis("k", 80, T.bk);
export const kern241 = kernel(
  { name: "k241", tile: T, grid: [M241, N241], reduce: [K241],
    bindings: [input("a", [M241, K241], f32), input("b", [K241, N241], f32), output("c", [M241, N241], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M242 = axis("m", 384, T.bm), N242 = axis("n", 192, T.bn), K242 = axis("k", 96, T.bk);
export const kern242 = kernel(
  { name: "k242", tile: T, grid: [M242, N242], reduce: [K242],
    bindings: [input("a", [M242, K242], f32), input("b", [K242, N242], f32), output("c", [M242, N242], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M243 = axis("m", 448, T.bm), N243 = axis("n", 256, T.bn), K243 = axis("k", 112, T.bk);
export const kern243 = kernel(
  { name: "k243", tile: T, grid: [M243, N243], reduce: [K243],
    bindings: [input("a", [M243, K243], f32), input("b", [K243, N243], f32), output("c", [M243, N243], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M244 = axis("m", 512, T.bm), N244 = axis("n", 320, T.bn), K244 = axis("k", 128, T.bk);
export const kern244 = kernel(
  { name: "k244", tile: T, grid: [M244, N244], reduce: [K244],
    bindings: [input("a", [M244, K244], f32), input("b", [K244, N244], f32), output("c", [M244, N244], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M245 = axis("m", 576, T.bm), N245 = axis("n", 384, T.bn), K245 = axis("k", 144, T.bk);
export const kern245 = kernel(
  { name: "k245", tile: T, grid: [M245, N245], reduce: [K245],
    bindings: [input("a", [M245, K245], f32), input("b", [K245, N245], f32), output("c", [M245, N245], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M246 = axis("m", 640, T.bm), N246 = axis("n", 448, T.bn), K246 = axis("k", 160, T.bk);
export const kern246 = kernel(
  { name: "k246", tile: T, grid: [M246, N246], reduce: [K246],
    bindings: [input("a", [M246, K246], f32), input("b", [K246, N246], f32), output("c", [M246, N246], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M247 = axis("m", 704, T.bm), N247 = axis("n", 512, T.bn), K247 = axis("k", 176, T.bk);
export const kern247 = kernel(
  { name: "k247", tile: T, grid: [M247, N247], reduce: [K247],
    bindings: [input("a", [M247, K247], f32), input("b", [K247, N247], f32), output("c", [M247, N247], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M248 = axis("m", 768, T.bm), N248 = axis("n", 576, T.bn), K248 = axis("k", 192, T.bk);
export const kern248 = kernel(
  { name: "k248", tile: T, grid: [M248, N248], reduce: [K248],
    bindings: [input("a", [M248, K248], f32), input("b", [K248, N248], f32), output("c", [M248, N248], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M249 = axis("m", 832, T.bm), N249 = axis("n", 640, T.bn), K249 = axis("k", 208, T.bk);
export const kern249 = kernel(
  { name: "k249", tile: T, grid: [M249, N249], reduce: [K249],
    bindings: [input("a", [M249, K249], f32), input("b", [K249, N249], f32), output("c", [M249, N249], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M250 = axis("m", 896, T.bm), N250 = axis("n", 704, T.bn), K250 = axis("k", 224, T.bk);
export const kern250 = kernel(
  { name: "k250", tile: T, grid: [M250, N250], reduce: [K250],
    bindings: [input("a", [M250, K250], f32), input("b", [K250, N250], f32), output("c", [M250, N250], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M251 = axis("m", 960, T.bm), N251 = axis("n", 768, T.bn), K251 = axis("k", 240, T.bk);
export const kern251 = kernel(
  { name: "k251", tile: T, grid: [M251, N251], reduce: [K251],
    bindings: [input("a", [M251, K251], f32), input("b", [K251, N251], f32), output("c", [M251, N251], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M252 = axis("m", 128, T.bm), N252 = axis("n", 832, T.bn), K252 = axis("k", 256, T.bk);
export const kern252 = kernel(
  { name: "k252", tile: T, grid: [M252, N252], reduce: [K252],
    bindings: [input("a", [M252, K252], f32), input("b", [K252, N252], f32), output("c", [M252, N252], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M253 = axis("m", 192, T.bm), N253 = axis("n", 192, T.bn), K253 = axis("k", 272, T.bk);
export const kern253 = kernel(
  { name: "k253", tile: T, grid: [M253, N253], reduce: [K253],
    bindings: [input("a", [M253, K253], f32), input("b", [K253, N253], f32), output("c", [M253, N253], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M254 = axis("m", 256, T.bm), N254 = axis("n", 256, T.bn), K254 = axis("k", 288, T.bk);
export const kern254 = kernel(
  { name: "k254", tile: T, grid: [M254, N254], reduce: [K254],
    bindings: [input("a", [M254, K254], f32), input("b", [K254, N254], f32), output("c", [M254, N254], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M255 = axis("m", 320, T.bm), N255 = axis("n", 320, T.bn), K255 = axis("k", 304, T.bk);
export const kern255 = kernel(
  { name: "k255", tile: T, grid: [M255, N255], reduce: [K255],
    bindings: [input("a", [M255, K255], f32), input("b", [K255, N255], f32), output("c", [M255, N255], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M256 = axis("m", 384, T.bm), N256 = axis("n", 384, T.bn), K256 = axis("k", 320, T.bk);
export const kern256 = kernel(
  { name: "k256", tile: T, grid: [M256, N256], reduce: [K256],
    bindings: [input("a", [M256, K256], f32), input("b", [K256, N256], f32), output("c", [M256, N256], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M257 = axis("m", 448, T.bm), N257 = axis("n", 448, T.bn), K257 = axis("k", 336, T.bk);
export const kern257 = kernel(
  { name: "k257", tile: T, grid: [M257, N257], reduce: [K257],
    bindings: [input("a", [M257, K257], f32), input("b", [K257, N257], f32), output("c", [M257, N257], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M258 = axis("m", 512, T.bm), N258 = axis("n", 512, T.bn), K258 = axis("k", 352, T.bk);
export const kern258 = kernel(
  { name: "k258", tile: T, grid: [M258, N258], reduce: [K258],
    bindings: [input("a", [M258, K258], f32), input("b", [K258, N258], f32), output("c", [M258, N258], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M259 = axis("m", 576, T.bm), N259 = axis("n", 576, T.bn), K259 = axis("k", 368, T.bk);
export const kern259 = kernel(
  { name: "k259", tile: T, grid: [M259, N259], reduce: [K259],
    bindings: [input("a", [M259, K259], f32), input("b", [K259, N259], f32), output("c", [M259, N259], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M260 = axis("m", 640, T.bm), N260 = axis("n", 640, T.bn), K260 = axis("k", 64, T.bk);
export const kern260 = kernel(
  { name: "k260", tile: T, grid: [M260, N260], reduce: [K260],
    bindings: [input("a", [M260, K260], f32), input("b", [K260, N260], f32), output("c", [M260, N260], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M261 = axis("m", 704, T.bm), N261 = axis("n", 704, T.bn), K261 = axis("k", 80, T.bk);
export const kern261 = kernel(
  { name: "k261", tile: T, grid: [M261, N261], reduce: [K261],
    bindings: [input("a", [M261, K261], f32), input("b", [K261, N261], f32), output("c", [M261, N261], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M262 = axis("m", 768, T.bm), N262 = axis("n", 768, T.bn), K262 = axis("k", 96, T.bk);
export const kern262 = kernel(
  { name: "k262", tile: T, grid: [M262, N262], reduce: [K262],
    bindings: [input("a", [M262, K262], f32), input("b", [K262, N262], f32), output("c", [M262, N262], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M263 = axis("m", 832, T.bm), N263 = axis("n", 832, T.bn), K263 = axis("k", 112, T.bk);
export const kern263 = kernel(
  { name: "k263", tile: T, grid: [M263, N263], reduce: [K263],
    bindings: [input("a", [M263, K263], f32), input("b", [K263, N263], f32), output("c", [M263, N263], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M264 = axis("m", 896, T.bm), N264 = axis("n", 192, T.bn), K264 = axis("k", 128, T.bk);
export const kern264 = kernel(
  { name: "k264", tile: T, grid: [M264, N264], reduce: [K264],
    bindings: [input("a", [M264, K264], f32), input("b", [K264, N264], f32), output("c", [M264, N264], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M265 = axis("m", 960, T.bm), N265 = axis("n", 256, T.bn), K265 = axis("k", 144, T.bk);
export const kern265 = kernel(
  { name: "k265", tile: T, grid: [M265, N265], reduce: [K265],
    bindings: [input("a", [M265, K265], f32), input("b", [K265, N265], f32), output("c", [M265, N265], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M266 = axis("m", 128, T.bm), N266 = axis("n", 320, T.bn), K266 = axis("k", 160, T.bk);
export const kern266 = kernel(
  { name: "k266", tile: T, grid: [M266, N266], reduce: [K266],
    bindings: [input("a", [M266, K266], f32), input("b", [K266, N266], f32), output("c", [M266, N266], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M267 = axis("m", 192, T.bm), N267 = axis("n", 384, T.bn), K267 = axis("k", 176, T.bk);
export const kern267 = kernel(
  { name: "k267", tile: T, grid: [M267, N267], reduce: [K267],
    bindings: [input("a", [M267, K267], f32), input("b", [K267, N267], f32), output("c", [M267, N267], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M268 = axis("m", 256, T.bm), N268 = axis("n", 448, T.bn), K268 = axis("k", 192, T.bk);
export const kern268 = kernel(
  { name: "k268", tile: T, grid: [M268, N268], reduce: [K268],
    bindings: [input("a", [M268, K268], f32), input("b", [K268, N268], f32), output("c", [M268, N268], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M269 = axis("m", 320, T.bm), N269 = axis("n", 512, T.bn), K269 = axis("k", 208, T.bk);
export const kern269 = kernel(
  { name: "k269", tile: T, grid: [M269, N269], reduce: [K269],
    bindings: [input("a", [M269, K269], f32), input("b", [K269, N269], f32), output("c", [M269, N269], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M270 = axis("m", 384, T.bm), N270 = axis("n", 576, T.bn), K270 = axis("k", 224, T.bk);
export const kern270 = kernel(
  { name: "k270", tile: T, grid: [M270, N270], reduce: [K270],
    bindings: [input("a", [M270, K270], f32), input("b", [K270, N270], f32), output("c", [M270, N270], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M271 = axis("m", 448, T.bm), N271 = axis("n", 640, T.bn), K271 = axis("k", 240, T.bk);
export const kern271 = kernel(
  { name: "k271", tile: T, grid: [M271, N271], reduce: [K271],
    bindings: [input("a", [M271, K271], f32), input("b", [K271, N271], f32), output("c", [M271, N271], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M272 = axis("m", 512, T.bm), N272 = axis("n", 704, T.bn), K272 = axis("k", 256, T.bk);
export const kern272 = kernel(
  { name: "k272", tile: T, grid: [M272, N272], reduce: [K272],
    bindings: [input("a", [M272, K272], f32), input("b", [K272, N272], f32), output("c", [M272, N272], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M273 = axis("m", 576, T.bm), N273 = axis("n", 768, T.bn), K273 = axis("k", 272, T.bk);
export const kern273 = kernel(
  { name: "k273", tile: T, grid: [M273, N273], reduce: [K273],
    bindings: [input("a", [M273, K273], f32), input("b", [K273, N273], f32), output("c", [M273, N273], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M274 = axis("m", 640, T.bm), N274 = axis("n", 832, T.bn), K274 = axis("k", 288, T.bk);
export const kern274 = kernel(
  { name: "k274", tile: T, grid: [M274, N274], reduce: [K274],
    bindings: [input("a", [M274, K274], f32), input("b", [K274, N274], f32), output("c", [M274, N274], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M275 = axis("m", 704, T.bm), N275 = axis("n", 192, T.bn), K275 = axis("k", 304, T.bk);
export const kern275 = kernel(
  { name: "k275", tile: T, grid: [M275, N275], reduce: [K275],
    bindings: [input("a", [M275, K275], f32), input("b", [K275, N275], f32), output("c", [M275, N275], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M276 = axis("m", 768, T.bm), N276 = axis("n", 256, T.bn), K276 = axis("k", 320, T.bk);
export const kern276 = kernel(
  { name: "k276", tile: T, grid: [M276, N276], reduce: [K276],
    bindings: [input("a", [M276, K276], f32), input("b", [K276, N276], f32), output("c", [M276, N276], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M277 = axis("m", 832, T.bm), N277 = axis("n", 320, T.bn), K277 = axis("k", 336, T.bk);
export const kern277 = kernel(
  { name: "k277", tile: T, grid: [M277, N277], reduce: [K277],
    bindings: [input("a", [M277, K277], f32), input("b", [K277, N277], f32), output("c", [M277, N277], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M278 = axis("m", 896, T.bm), N278 = axis("n", 384, T.bn), K278 = axis("k", 352, T.bk);
export const kern278 = kernel(
  { name: "k278", tile: T, grid: [M278, N278], reduce: [K278],
    bindings: [input("a", [M278, K278], f32), input("b", [K278, N278], f32), output("c", [M278, N278], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M279 = axis("m", 960, T.bm), N279 = axis("n", 448, T.bn), K279 = axis("k", 368, T.bk);
export const kern279 = kernel(
  { name: "k279", tile: T, grid: [M279, N279], reduce: [K279],
    bindings: [input("a", [M279, K279], f32), input("b", [K279, N279], f32), output("c", [M279, N279], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M280 = axis("m", 128, T.bm), N280 = axis("n", 512, T.bn), K280 = axis("k", 64, T.bk);
export const kern280 = kernel(
  { name: "k280", tile: T, grid: [M280, N280], reduce: [K280],
    bindings: [input("a", [M280, K280], f32), input("b", [K280, N280], f32), output("c", [M280, N280], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M281 = axis("m", 192, T.bm), N281 = axis("n", 576, T.bn), K281 = axis("k", 80, T.bk);
export const kern281 = kernel(
  { name: "k281", tile: T, grid: [M281, N281], reduce: [K281],
    bindings: [input("a", [M281, K281], f32), input("b", [K281, N281], f32), output("c", [M281, N281], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M282 = axis("m", 256, T.bm), N282 = axis("n", 640, T.bn), K282 = axis("k", 96, T.bk);
export const kern282 = kernel(
  { name: "k282", tile: T, grid: [M282, N282], reduce: [K282],
    bindings: [input("a", [M282, K282], f32), input("b", [K282, N282], f32), output("c", [M282, N282], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M283 = axis("m", 320, T.bm), N283 = axis("n", 704, T.bn), K283 = axis("k", 112, T.bk);
export const kern283 = kernel(
  { name: "k283", tile: T, grid: [M283, N283], reduce: [K283],
    bindings: [input("a", [M283, K283], f32), input("b", [K283, N283], f32), output("c", [M283, N283], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M284 = axis("m", 384, T.bm), N284 = axis("n", 768, T.bn), K284 = axis("k", 128, T.bk);
export const kern284 = kernel(
  { name: "k284", tile: T, grid: [M284, N284], reduce: [K284],
    bindings: [input("a", [M284, K284], f32), input("b", [K284, N284], f32), output("c", [M284, N284], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M285 = axis("m", 448, T.bm), N285 = axis("n", 832, T.bn), K285 = axis("k", 144, T.bk);
export const kern285 = kernel(
  { name: "k285", tile: T, grid: [M285, N285], reduce: [K285],
    bindings: [input("a", [M285, K285], f32), input("b", [K285, N285], f32), output("c", [M285, N285], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M286 = axis("m", 512, T.bm), N286 = axis("n", 192, T.bn), K286 = axis("k", 160, T.bk);
export const kern286 = kernel(
  { name: "k286", tile: T, grid: [M286, N286], reduce: [K286],
    bindings: [input("a", [M286, K286], f32), input("b", [K286, N286], f32), output("c", [M286, N286], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M287 = axis("m", 576, T.bm), N287 = axis("n", 256, T.bn), K287 = axis("k", 176, T.bk);
export const kern287 = kernel(
  { name: "k287", tile: T, grid: [M287, N287], reduce: [K287],
    bindings: [input("a", [M287, K287], f32), input("b", [K287, N287], f32), output("c", [M287, N287], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M288 = axis("m", 640, T.bm), N288 = axis("n", 320, T.bn), K288 = axis("k", 192, T.bk);
export const kern288 = kernel(
  { name: "k288", tile: T, grid: [M288, N288], reduce: [K288],
    bindings: [input("a", [M288, K288], f32), input("b", [K288, N288], f32), output("c", [M288, N288], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M289 = axis("m", 704, T.bm), N289 = axis("n", 384, T.bn), K289 = axis("k", 208, T.bk);
export const kern289 = kernel(
  { name: "k289", tile: T, grid: [M289, N289], reduce: [K289],
    bindings: [input("a", [M289, K289], f32), input("b", [K289, N289], f32), output("c", [M289, N289], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M290 = axis("m", 768, T.bm), N290 = axis("n", 448, T.bn), K290 = axis("k", 224, T.bk);
export const kern290 = kernel(
  { name: "k290", tile: T, grid: [M290, N290], reduce: [K290],
    bindings: [input("a", [M290, K290], f32), input("b", [K290, N290], f32), output("c", [M290, N290], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M291 = axis("m", 832, T.bm), N291 = axis("n", 512, T.bn), K291 = axis("k", 240, T.bk);
export const kern291 = kernel(
  { name: "k291", tile: T, grid: [M291, N291], reduce: [K291],
    bindings: [input("a", [M291, K291], f32), input("b", [K291, N291], f32), output("c", [M291, N291], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M292 = axis("m", 896, T.bm), N292 = axis("n", 576, T.bn), K292 = axis("k", 256, T.bk);
export const kern292 = kernel(
  { name: "k292", tile: T, grid: [M292, N292], reduce: [K292],
    bindings: [input("a", [M292, K292], f32), input("b", [K292, N292], f32), output("c", [M292, N292], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M293 = axis("m", 960, T.bm), N293 = axis("n", 640, T.bn), K293 = axis("k", 272, T.bk);
export const kern293 = kernel(
  { name: "k293", tile: T, grid: [M293, N293], reduce: [K293],
    bindings: [input("a", [M293, K293], f32), input("b", [K293, N293], f32), output("c", [M293, N293], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M294 = axis("m", 128, T.bm), N294 = axis("n", 704, T.bn), K294 = axis("k", 288, T.bk);
export const kern294 = kernel(
  { name: "k294", tile: T, grid: [M294, N294], reduce: [K294],
    bindings: [input("a", [M294, K294], f32), input("b", [K294, N294], f32), output("c", [M294, N294], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M295 = axis("m", 192, T.bm), N295 = axis("n", 768, T.bn), K295 = axis("k", 304, T.bk);
export const kern295 = kernel(
  { name: "k295", tile: T, grid: [M295, N295], reduce: [K295],
    bindings: [input("a", [M295, K295], f32), input("b", [K295, N295], f32), output("c", [M295, N295], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M296 = axis("m", 256, T.bm), N296 = axis("n", 832, T.bn), K296 = axis("k", 320, T.bk);
export const kern296 = kernel(
  { name: "k296", tile: T, grid: [M296, N296], reduce: [K296],
    bindings: [input("a", [M296, K296], f32), input("b", [K296, N296], f32), output("c", [M296, N296], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M297 = axis("m", 320, T.bm), N297 = axis("n", 192, T.bn), K297 = axis("k", 336, T.bk);
export const kern297 = kernel(
  { name: "k297", tile: T, grid: [M297, N297], reduce: [K297],
    bindings: [input("a", [M297, K297], f32), input("b", [K297, N297], f32), output("c", [M297, N297], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M298 = axis("m", 384, T.bm), N298 = axis("n", 256, T.bn), K298 = axis("k", 352, T.bk);
export const kern298 = kernel(
  { name: "k298", tile: T, grid: [M298, N298], reduce: [K298],
    bindings: [input("a", [M298, K298], f32), input("b", [K298, N298], f32), output("c", [M298, N298], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });

const M299 = axis("m", 448, T.bm), N299 = axis("n", 320, T.bn), K299 = axis("k", 368, T.bk);
export const kern299 = kernel(
  { name: "k299", tile: T, grid: [M299, N299], reduce: [K299],
    bindings: [input("a", [M299, K299], f32), input("b", [K299, N299], f32), output("c", [M299, N299], f32)] },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aT = a.tile(at.m, k);
      const bT = b.tile(k, at.n);
      acc = mma(aT, bT, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  });
