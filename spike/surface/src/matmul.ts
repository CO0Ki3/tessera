// matmul.kernel.ts -- a tiled matmul + fused ReLU, written in tessera.

import {
  axis, raggedAxis, tiling, kernel, input, output,
  zeros, mma, relu, f32,
  requestDevice, randn, emptyHost, upload, download, allClose,
} from "./tessera";

// ---- The tile. The only performance knob, and the only one that is required.
// `tiling` admits only triples that fit the workgroup-storage budget for this
// dtype: (64*16 + 16*64) * 4 B = 8192 B of the 12288 B tessera makes available
// out of WebGPU's guaranteed 16384 B.
const T = tiling(f32, 64, 64, 16);

// ---- The problem, as named axes. The name is part of the type, so a
// transposed coordinate pair reads `Type '"n"' is not assignable to type '"k"'`.
const M = axis("m", 1024, T.bm); // Axis<"m", 1024, 64, "exact">
const N = axis("n", 768, T.bn);  // Axis<"n",  768, 64, "exact">
const K = axis("k", 512, T.bk);  // Axis<"k",  512, 16, "exact">

export const matmulRelu = kernel(
  {
    name: "matmul_relu_f32",
    tile: T,
    grid: [M, N],
    reduce: [K],
    bindings: [
      input("a", [M, K], f32),
      input("b", [K, N], f32),
      output("c", [M, N], f32),
    ],
    // No `workgroup`, no `order`. tessera picks [16,16,1] -- 256 invocations,
    // a 4x4 f32 fragment each -- and row-major dispatch. Both are overridable;
    // deleting them changes nothing about what the program means.
  },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);      // Frag<[64, 64], f32>

    for (const k of reduce.k) {            // Idx<"k", 512, 16>, 32 steps
      const aTile = a.tile(at.m, k);       // Tile<[64, 16], f32, 0>
      const bTile = b.tile(k, at.n);       // Tile<[16, 64], f32, 0>
      acc = mma(aTile, bTile, acc);        // Frag<[64, 64], f32>
    }

    c.tile(at.m, at.n).store(relu(acc));   // fused epilogue, edge-clipped store
  },
);

// ---------------------------------------------------------------------------
// The ragged variant. 1000 is not a multiple of 64. `axis("m", 1000, T.bm)`
// would be TSA0301, naming the blocks that do divide 1000; `raggedAxis` opts
// in, and every load through that axis becomes a RaggedTile, so the file stops
// compiling until you name the identity element. tessera synthesizes the mask
// from the literal extents -- the loop, the accumulator and the store are
// character-identical to the aligned kernel above.
// ---------------------------------------------------------------------------
const Mr = raggedAxis("m", 1000, T.bm);
const Nr = axis("n", 768, T.bn);
const Kr = axis("k", 512, T.bk);

export const matmulReluRagged = kernel(
  {
    name: "matmul_relu_f32_ragged",
    tile: T,
    grid: [Mr, Nr],
    reduce: [Kr],
    bindings: [
      input("a", [Mr, Kr], f32),
      input("b", [Kr, Nr], f32),
      output("c", [Mr, Nr], f32),
    ],
  },
  ({ a, b, c, at, reduce }) => {
    let acc = zeros(T.bm, T.bn, f32);
    for (const k of reduce.k) {
      const aTile = a.tile(at.m, k).pad(0); // required here and nowhere else
      const bTile = b.tile(k, at.n);
      acc = mma(aTile, bTile, acc);
    }
    c.tile(at.m, at.n).store(relu(acc));
  },
);

// ---------------------------------------------------------------------------
// Host side + the v0 differential test. `reference` runs THE SAME kernel body
// on tessera's CPU backend under plain node, so there is no second
// implementation that can drift out of sync.
// ---------------------------------------------------------------------------
export async function main(): Promise<void> {
  const device = await requestDevice();

  const a = randn([M, K], f32, 1);
  const b = randn([K, N], f32, 2);
  const cGpuHost = emptyHost([M, N], f32);
  const cCpu = emptyHost([M, N], f32);

  await matmulRelu.run(device, {
    a: upload(device, a),
    b: upload(device, b),
    c: upload(device, cGpuHost),
  });
  const cGpu = await download(device, upload(device, cGpuHost));

  matmulRelu.reference({ a, b, c: cCpu });

  const cmp = allClose(cGpu, cCpu, { rtol: 1e-5, atol: 1e-6 });
  if (!cmp.ok) throw new Error(`GPU/CPU divergence: maxRelDiff=${cmp.maxRelDiff}`);

  console.log(matmulRelu.report);
  // { workgroup: [16,16,1], dispatch: [16,12,1], workgroupBytes: 8192,
  //   fragment: [4,4], maskedLoads: [], storageBuffers: 3 }
}
