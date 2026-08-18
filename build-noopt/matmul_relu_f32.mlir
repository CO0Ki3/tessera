module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @matmul_relu_f32(%a: memref<524288xf32, #spirv.storage_class<StorageBuffer>>,
                     %b: memref<393216xf32, #spirv.storage_class<StorageBuffer>>,
                     %c: memref<786432xf32, #spirv.storage_class<StorageBuffer>>) kernel
        attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 16, 1]>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c512 = arith.constant 512 : index
      %c768 = arith.constant 768 : index
      %c1024 = arith.constant 1024 : index
      %f0 = arith.constant 0.0 : f32

      %As = memref.alloc() : memref<1024xf32, #spirv.storage_class<Workgroup>>
      %Bs = memref.alloc() : memref<1024xf32, #spirv.storage_class<Workgroup>>

      %bx = gpu.block_id x
      %by = gpu.block_id y
      %tx = gpu.thread_id x
      %ty = gpu.thread_id y
      %tid0 = arith.muli %ty, %c16 : index
      %tid = arith.addi %tid0, %tx : index
      %ty4 = arith.muli %ty, %c4 : index
      %tx4 = arith.muli %tx, %c4 : index
      %rowBase = arith.muli %by, %c64 : index
      %colBase = arith.muli %bx, %c64 : index

      %out:16 = scf.for %kk = %c0 to %c512 step %c16 iter_args(%acc0 = %f0, %acc1 = %f0, %acc2 = %f0, %acc3 = %f0, %acc4 = %f0, %acc5 = %f0, %acc6 = %f0, %acc7 = %f0, %acc8 = %f0, %acc9 = %f0, %acc10 = %f0, %acc11 = %f0, %acc12 = %f0, %acc13 = %f0, %acc14 = %f0, %acc15 = %f0) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
        scf.for %i = %tid to %c1024 step %c256 {
          %r = arith.divui %i, %c16 : index
          %cc = arith.remui %i, %c16 : index
          %gr = arith.addi %rowBase, %r : index
          %gr2 = arith.muli %gr, %c512 : index
          %gc = arith.addi %kk, %cc : index
          %off = arith.addi %gr2, %gc : index
          %val = memref.load %a[%off] : memref<524288xf32, #spirv.storage_class<StorageBuffer>>
          memref.store %val, %As[%i] : memref<1024xf32, #spirv.storage_class<Workgroup>>
        }
        scf.for %i = %tid to %c1024 step %c256 {
          %r = arith.divui %i, %c64 : index
          %cc = arith.remui %i, %c64 : index
          %gr = arith.addi %kk, %r : index
          %gr2 = arith.muli %gr, %c768 : index
          %gc = arith.addi %colBase, %cc : index
          %off = arith.addi %gr2, %gc : index
          %val = memref.load %b[%off] : memref<393216xf32, #spirv.storage_class<StorageBuffer>>
          memref.store %val, %Bs[%i] : memref<1024xf32, #spirv.storage_class<Workgroup>>
        }
        gpu.barrier
        %acc:16 = scf.for %k = %c0 to %c16 step %c1 iter_args(%i0 = %acc0, %i1 = %acc1, %i2 = %acc2, %i3 = %acc3, %i4 = %acc4, %i5 = %acc5, %i6 = %acc6, %i7 = %acc7, %i8 = %acc8, %i9 = %acc9, %i10 = %acc10, %i11 = %acc11, %i12 = %acc12, %i13 = %acc13, %i14 = %acc14, %i15 = %acc15) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
          %am0a = arith.addi %ty4, %c0 : index
          %am0b = arith.muli %am0a, %c16 : index
          %am0c = arith.addi %am0b, %k : index
          %af0 = memref.load %As[%am0c] : memref<1024xf32, #spirv.storage_class<Workgroup>>
          %am1a = arith.addi %ty4, %c1 : index
          %am1b = arith.muli %am1a, %c16 : index
          %am1c = arith.addi %am1b, %k : index
          %af1 = memref.load %As[%am1c] : memref<1024xf32, #spirv.storage_class<Workgroup>>
          %am2a = arith.addi %ty4, %c2 : index
          %am2b = arith.muli %am2a, %c16 : index
          %am2c = arith.addi %am2b, %k : index
          %af2 = memref.load %As[%am2c] : memref<1024xf32, #spirv.storage_class<Workgroup>>
          %am3a = arith.addi %ty4, %c3 : index
          %am3b = arith.muli %am3a, %c16 : index
          %am3c = arith.addi %am3b, %k : index
          %af3 = memref.load %As[%am3c] : memref<1024xf32, #spirv.storage_class<Workgroup>>
          %bn0a = arith.muli %k, %c64 : index
          %bn0b = arith.addi %tx4, %c0 : index
          %bn0c = arith.addi %bn0a, %bn0b : index
          %bf0 = memref.load %Bs[%bn0c] : memref<1024xf32, #spirv.storage_class<Workgroup>>
          %bn1a = arith.muli %k, %c64 : index
          %bn1b = arith.addi %tx4, %c1 : index
          %bn1c = arith.addi %bn1a, %bn1b : index
          %bf1 = memref.load %Bs[%bn1c] : memref<1024xf32, #spirv.storage_class<Workgroup>>
          %bn2a = arith.muli %k, %c64 : index
          %bn2b = arith.addi %tx4, %c2 : index
          %bn2c = arith.addi %bn2a, %bn2b : index
          %bf2 = memref.load %Bs[%bn2c] : memref<1024xf32, #spirv.storage_class<Workgroup>>
          %bn3a = arith.muli %k, %c64 : index
          %bn3b = arith.addi %tx4, %c3 : index
          %bn3c = arith.addi %bn3a, %bn3b : index
          %bf3 = memref.load %Bs[%bn3c] : memref<1024xf32, #spirv.storage_class<Workgroup>>
          %p0 = arith.mulf %af0, %bf0 : f32
          %s0 = arith.addf %i0, %p0 : f32
          %p1 = arith.mulf %af0, %bf1 : f32
          %s1 = arith.addf %i1, %p1 : f32
          %p2 = arith.mulf %af0, %bf2 : f32
          %s2 = arith.addf %i2, %p2 : f32
          %p3 = arith.mulf %af0, %bf3 : f32
          %s3 = arith.addf %i3, %p3 : f32
          %p4 = arith.mulf %af1, %bf0 : f32
          %s4 = arith.addf %i4, %p4 : f32
          %p5 = arith.mulf %af1, %bf1 : f32
          %s5 = arith.addf %i5, %p5 : f32
          %p6 = arith.mulf %af1, %bf2 : f32
          %s6 = arith.addf %i6, %p6 : f32
          %p7 = arith.mulf %af1, %bf3 : f32
          %s7 = arith.addf %i7, %p7 : f32
          %p8 = arith.mulf %af2, %bf0 : f32
          %s8 = arith.addf %i8, %p8 : f32
          %p9 = arith.mulf %af2, %bf1 : f32
          %s9 = arith.addf %i9, %p9 : f32
          %p10 = arith.mulf %af2, %bf2 : f32
          %s10 = arith.addf %i10, %p10 : f32
          %p11 = arith.mulf %af2, %bf3 : f32
          %s11 = arith.addf %i11, %p11 : f32
          %p12 = arith.mulf %af3, %bf0 : f32
          %s12 = arith.addf %i12, %p12 : f32
          %p13 = arith.mulf %af3, %bf1 : f32
          %s13 = arith.addf %i13, %p13 : f32
          %p14 = arith.mulf %af3, %bf2 : f32
          %s14 = arith.addf %i14, %p14 : f32
          %p15 = arith.mulf %af3, %bf3 : f32
          %s15 = arith.addf %i15, %p15 : f32
          scf.yield %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8, %s9, %s10, %s11, %s12, %s13, %s14, %s15 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
        }
        gpu.barrier
        scf.yield %acc#0, %acc#1, %acc#2, %acc#3, %acc#4, %acc#5, %acc#6, %acc#7, %acc#8, %acc#9, %acc#10, %acc#11, %acc#12, %acc#13, %acc#14, %acc#15 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
      }

      %sr0a = arith.addi %ty4, %c0 : index
      %sr0 = arith.addi %rowBase, %sr0a : index
      %sr0n = arith.muli %sr0, %c768 : index
      %sc0a = arith.addi %tx4, %c0 : index
      %sc0 = arith.addi %colBase, %sc0a : index
      %so0 = arith.addi %sr0n, %sc0 : index
      %rc0 = arith.cmpf ogt, %out#0, %f0 : f32
      %rl0 = arith.select %rc0, %out#0, %f0 : f32
      memref.store %rl0, %c[%so0] : memref<786432xf32, #spirv.storage_class<StorageBuffer>>
      %sc1a = arith.addi %tx4, %c1 : index
      %sc1 = arith.addi %colBase, %sc1a : index
      %so1 = arith.addi %sr0n, %sc1 : index
      %rc1 = arith.cmpf ogt, %out#1, %f0 : f32
      %rl1 = arith.select %rc1, %out#1, %f0 : f32
      memref.store %rl1, %c[%so1] : memref<786432xf32, #spirv.storage_class<StorageBuffer>>
      %sc2a = arith.addi %tx4, %c2 : index
      %sc2 = arith.addi %colBase, %sc2a : index
      %so2 = arith.addi %sr0n, %sc2 : index
      %rc2 = arith.cmpf ogt, %out#2, %f0 : f32
      %rl2 = arith.select %rc2, %out#2, %f0 : f32
      memref.store %rl2, %c[%so2] : memref<786432xf32, #spirv.storage_class<StorageBuffer>>
      %sc3a = arith.addi %tx4, %c3 : index
      %sc3 = arith.addi %colBase, %sc3a : index
      %so3 = arith.addi %sr0n, %sc3 : index
      %rc3 = arith.cmpf ogt, %out#3, %f0 : f32
      %rl3 = arith.select %rc3, %out#3, %f0 : f32
      memref.store %rl3, %c[%so3] : memref<786432xf32, #spirv.storage_class<StorageBuffer>>
      %sr1a = arith.addi %ty4, %c1 : index
      %sr1 = arith.addi %rowBase, %sr1a : index
      %sr1n = arith.muli %sr1, %c768 : index
      %sc4a = arith.addi %tx4, %c0 : index
      %sc4 = arith.addi %colBase, %sc4a : index
      %so4 = arith.addi %sr1n, %sc4 : index
      %rc4 = arith.cmpf ogt, %out#4, %f0 : f32
      %rl4 = arith.select %rc4, %out#4, %f0 : f32
      memref.store %rl4, %c[%so4] : memref<786432xf32, #spirv.storage_class<StorageBuffer>>
      %sc5a = arith.addi %tx4, %c1 : index
      %sc5 = arith.addi %colBase, %sc5a : index
      %so5 = arith.addi %sr1n, %sc5 : index
      %rc5 = arith.cmpf ogt, %out#5, %f0 : f32
      %rl5 = arith.select %rc5, %out#5, %f0 : f32
      memref.store %rl5, %c[%so5] : memref<786432xf32, #spirv.storage_class<StorageBuffer>>
      %sc6a = arith.addi %tx4, %c2 : index
      %sc6 = arith.addi %colBase, %sc6a : index
      %so6 = arith.addi %sr1n, %sc6 : index
      %rc6 = arith.cmpf ogt, %out#6, %f0 : f32
      %rl6 = arith.select %rc6, %out#6, %f0 : f32
      memref.store %rl6, %c[%so6] : memref<786432xf32, #spirv.storage_class<StorageBuffer>>
      %sc7a = arith.addi %tx4, %c3 : index
      %sc7 = arith.addi %colBase, %sc7a : index
      %so7 = arith.addi %sr1n, %sc7 : index
      %rc7 = arith.cmpf ogt, %out#7, %f0 : f32
      %rl7 = arith.select %rc7, %out#7, %f0 : f32
      memref.store %rl7, %c[%so7] : memref<786432xf32, #spirv.storage_class<StorageBuffer>>
      %sr2a = arith.addi %ty4, %c2 : index
      %sr2 = arith.addi %rowBase, %sr2a : index
      %sr2n = arith.muli %sr2, %c768 : index
      %sc8a = arith.addi %tx4, %c0 : index
      %sc8 = arith.addi %colBase, %sc8a : index
      %so8 = arith.addi %sr2n, %sc8 : index
      %rc8 = arith.cmpf ogt, %out#8, %f0 : f32
      %rl8 = arith.select %rc8, %out#8, %f0 : f32
      memref.store %rl8, %c[%so8] : memref<786432xf32, #spirv.storage_class<StorageBuffer>>
      %sc9a = arith.addi %tx4, %c1 : index
      %sc9 = arith.addi %colBase, %sc9a : index
      %so9 = arith.addi %sr2n, %sc9 : index
      %rc9 = arith.cmpf ogt, %out#9, %f0 : f32
      %rl9 = arith.select %rc9, %out#9, %f0 : f32
      memref.store %rl9, %c[%so9] : memref<786432xf32, #spirv.storage_class<StorageBuffer>>
      %sc10a = arith.addi %tx4, %c2 : index
      %sc10 = arith.addi %colBase, %sc10a : index
      %so10 = arith.addi %sr2n, %sc10 : index
      %rc10 = arith.cmpf ogt, %out#10, %f0 : f32
      %rl10 = arith.select %rc10, %out#10, %f0 : f32
      memref.store %rl10, %c[%so10] : memref<786432xf32, #spirv.storage_class<StorageBuffer>>
      %sc11a = arith.addi %tx4, %c3 : index
      %sc11 = arith.addi %colBase, %sc11a : index
      %so11 = arith.addi %sr2n, %sc11 : index
      %rc11 = arith.cmpf ogt, %out#11, %f0 : f32
      %rl11 = arith.select %rc11, %out#11, %f0 : f32
      memref.store %rl11, %c[%so11] : memref<786432xf32, #spirv.storage_class<StorageBuffer>>
      %sr3a = arith.addi %ty4, %c3 : index
      %sr3 = arith.addi %rowBase, %sr3a : index
      %sr3n = arith.muli %sr3, %c768 : index
      %sc12a = arith.addi %tx4, %c0 : index
      %sc12 = arith.addi %colBase, %sc12a : index
      %so12 = arith.addi %sr3n, %sc12 : index
      %rc12 = arith.cmpf ogt, %out#12, %f0 : f32
      %rl12 = arith.select %rc12, %out#12, %f0 : f32
      memref.store %rl12, %c[%so12] : memref<786432xf32, #spirv.storage_class<StorageBuffer>>
      %sc13a = arith.addi %tx4, %c1 : index
      %sc13 = arith.addi %colBase, %sc13a : index
      %so13 = arith.addi %sr3n, %sc13 : index
      %rc13 = arith.cmpf ogt, %out#13, %f0 : f32
      %rl13 = arith.select %rc13, %out#13, %f0 : f32
      memref.store %rl13, %c[%so13] : memref<786432xf32, #spirv.storage_class<StorageBuffer>>
      %sc14a = arith.addi %tx4, %c2 : index
      %sc14 = arith.addi %colBase, %sc14a : index
      %so14 = arith.addi %sr3n, %sc14 : index
      %rc14 = arith.cmpf ogt, %out#14, %f0 : f32
      %rl14 = arith.select %rc14, %out#14, %f0 : f32
      memref.store %rl14, %c[%so14] : memref<786432xf32, #spirv.storage_class<StorageBuffer>>
      %sc15a = arith.addi %tx4, %c3 : index
      %sc15 = arith.addi %colBase, %sc15a : index
      %so15 = arith.addi %sr3n, %sc15 : index
      %rc15 = arith.cmpf ogt, %out#15, %f0 : f32
      %rl15 = arith.select %rc15, %out#15, %f0 : f32
      memref.store %rl15, %c[%so15] : memref<786432xf32, #spirv.storage_class<StorageBuffer>>

      gpu.return
    }
  }
}
