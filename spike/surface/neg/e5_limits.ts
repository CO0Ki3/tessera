import { axis, tiling, f32, f16 } from "../src/tessera";
export const t1 = tiling(f32, 64, 64, 32);    // 16384 B > 12288 B budget
export const t2 = tiling(f32, 128, 128, 16);  // way over
export const t3 = tiling(f16, 64, 64, 32);    // half precision: LEGAL, no error
export const a1 = axis("m", 1024, 48);        // 48 is not a legal block
declare function dynamicSize(): number;
export const a2 = axis("h", dynamicSize(), 64);  // extent lost its literalness
