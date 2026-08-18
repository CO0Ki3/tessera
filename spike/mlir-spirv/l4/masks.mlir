module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @masked(%src: memref<64xf32, #spirv.storage_class<StorageBuffer>>,
                     %dst: memref<64xf32, #spirv.storage_class<StorageBuffer>>) kernel
        attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      %c0 = arith.constant 0 : index
      %c40 = arith.constant 40 : index
      %c63 = arith.constant 63 : index
      %pad = arith.constant 0.0 : f32
      %tid = gpu.thread_id x

      // masked LOAD: clamp the address, then select. Branchless.
      %inb = arith.cmpi ult, %tid, %c40 : index
      %safe = arith.minui %tid, %c63 : index
      %raw = memref.load %src[%safe] : memref<64xf32, #spirv.storage_class<StorageBuffer>>
      %val = arith.select %inb, %raw, %pad : f32

      // masked STORE: must be a real conditional; clamping would corrupt data.
      scf.if %inb {
        memref.store %val, %dst[%tid] : memref<64xf32, #spirv.storage_class<StorageBuffer>>
      }
      gpu.return
    }
  }
}
