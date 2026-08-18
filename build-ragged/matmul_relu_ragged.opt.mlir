module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @matmul_relu_ragged(%arg0: memref<500000xf32, #spirv.storage_class<StorageBuffer>>, %arg1: memref<375000xf32, #spirv.storage_class<StorageBuffer>>, %arg2: memref<750000xf32, #spirv.storage_class<StorageBuffer>>) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 16, 1]>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c500 = arith.constant 500 : index
      %c750 = arith.constant 750 : index
      %c1000 = arith.constant 1000 : index
      %c1024 = arith.constant 1024 : index
      %c374999 = arith.constant 374999 : index
      %c499999 = arith.constant 499999 : index
      %cst = arith.constant 0.000000e+00 : f32
      %alloc = memref.alloc() : memref<1024xf32, #spirv.storage_class<Workgroup>>
      %alloc_0 = memref.alloc() : memref<1024xf32, #spirv.storage_class<Workgroup>>
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %0 = arith.muli %thread_id_y, %c16 : index
      %1 = arith.addi %0, %thread_id_x : index
      %2 = arith.muli %thread_id_y, %c4 : index
      %3 = arith.muli %thread_id_x, %c4 : index
      %4 = arith.muli %block_id_y, %c64 : index
      %5 = arith.muli %block_id_x, %c64 : index
      %6 = arith.muli %thread_id_y, %c64 : index
      %7 = arith.addi %2, %c1 : index
      %8 = arith.muli %7, %c16 : index
      %9 = arith.addi %2, %c2 : index
      %10 = arith.muli %9, %c16 : index
      %11 = arith.addi %2, %c3 : index
      %12 = arith.muli %11, %c16 : index
      %13 = arith.addi %3, %c1 : index
      %14 = arith.addi %3, %c2 : index
      %15 = arith.addi %3, %c3 : index
      %16:16 = scf.for %arg3 = %c0 to %c500 step %c16 iter_args(%arg4 = %cst, %arg5 = %cst, %arg6 = %cst, %arg7 = %cst, %arg8 = %cst, %arg9 = %cst, %arg10 = %cst, %arg11 = %cst, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst, %arg19 = %cst) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
        scf.for %arg20 = %1 to %c1024 step %c256 {
          %102 = arith.divui %arg20, %c16 : index
          %103 = arith.remui %arg20, %c16 : index
          %104 = arith.addi %4, %102 : index
          %105 = arith.muli %104, %c500 : index
          %106 = arith.addi %arg3, %103 : index
          %107 = arith.addi %105, %106 : index
          %108 = arith.cmpi ult, %104, %c1000 : index
          %109 = arith.cmpi ult, %106, %c500 : index
          %110 = arith.andi %108, %109 : i1
          %111 = arith.minui %107, %c499999 : index
          %112 = memref.load %arg0[%111] : memref<500000xf32, #spirv.storage_class<StorageBuffer>>
          %113 = arith.select %110, %112, %cst : f32
          memref.store %113, %alloc[%arg20] : memref<1024xf32, #spirv.storage_class<Workgroup>>
        }
        scf.for %arg20 = %1 to %c1024 step %c256 {
          %102 = arith.divui %arg20, %c64 : index
          %103 = arith.remui %arg20, %c64 : index
          %104 = arith.addi %arg3, %102 : index
          %105 = arith.muli %104, %c750 : index
          %106 = arith.addi %5, %103 : index
          %107 = arith.addi %105, %106 : index
          %108 = arith.cmpi ult, %104, %c500 : index
          %109 = arith.cmpi ult, %106, %c750 : index
          %110 = arith.andi %108, %109 : i1
          %111 = arith.minui %107, %c374999 : index
          %112 = memref.load %arg1[%111] : memref<375000xf32, #spirv.storage_class<StorageBuffer>>
          %113 = arith.select %110, %112, %cst : f32
          memref.store %113, %alloc_0[%arg20] : memref<1024xf32, #spirv.storage_class<Workgroup>>
        }
        gpu.barrier
        %101:16 = scf.for %arg20 = %c0 to %c16 step %c1 iter_args(%arg21 = %arg4, %arg22 = %arg5, %arg23 = %arg6, %arg24 = %arg7, %arg25 = %arg8, %arg26 = %arg9, %arg27 = %arg10, %arg28 = %arg11, %arg29 = %arg12, %arg30 = %arg13, %arg31 = %arg14, %arg32 = %arg15, %arg33 = %arg16, %arg34 = %arg17, %arg35 = %arg18, %arg36 = %arg19) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
          %102 = arith.addi %6, %arg20 : index
          %103 = memref.load %alloc[%102] : memref<1024xf32, #spirv.storage_class<Workgroup>>
          %104 = arith.addi %8, %arg20 : index
          %105 = memref.load %alloc[%104] : memref<1024xf32, #spirv.storage_class<Workgroup>>
          %106 = arith.addi %10, %arg20 : index
          %107 = memref.load %alloc[%106] : memref<1024xf32, #spirv.storage_class<Workgroup>>
          %108 = arith.addi %12, %arg20 : index
          %109 = memref.load %alloc[%108] : memref<1024xf32, #spirv.storage_class<Workgroup>>
          %110 = arith.muli %arg20, %c64 : index
          %111 = arith.addi %110, %3 : index
          %112 = memref.load %alloc_0[%111] : memref<1024xf32, #spirv.storage_class<Workgroup>>
          %113 = arith.addi %110, %13 : index
          %114 = memref.load %alloc_0[%113] : memref<1024xf32, #spirv.storage_class<Workgroup>>
          %115 = arith.addi %110, %14 : index
          %116 = memref.load %alloc_0[%115] : memref<1024xf32, #spirv.storage_class<Workgroup>>
          %117 = arith.addi %110, %15 : index
          %118 = memref.load %alloc_0[%117] : memref<1024xf32, #spirv.storage_class<Workgroup>>
          %119 = arith.mulf %103, %112 : f32
          %120 = arith.addf %arg21, %119 : f32
          %121 = arith.mulf %103, %114 : f32
          %122 = arith.addf %arg22, %121 : f32
          %123 = arith.mulf %103, %116 : f32
          %124 = arith.addf %arg23, %123 : f32
          %125 = arith.mulf %103, %118 : f32
          %126 = arith.addf %arg24, %125 : f32
          %127 = arith.mulf %105, %112 : f32
          %128 = arith.addf %arg25, %127 : f32
          %129 = arith.mulf %105, %114 : f32
          %130 = arith.addf %arg26, %129 : f32
          %131 = arith.mulf %105, %116 : f32
          %132 = arith.addf %arg27, %131 : f32
          %133 = arith.mulf %105, %118 : f32
          %134 = arith.addf %arg28, %133 : f32
          %135 = arith.mulf %107, %112 : f32
          %136 = arith.addf %arg29, %135 : f32
          %137 = arith.mulf %107, %114 : f32
          %138 = arith.addf %arg30, %137 : f32
          %139 = arith.mulf %107, %116 : f32
          %140 = arith.addf %arg31, %139 : f32
          %141 = arith.mulf %107, %118 : f32
          %142 = arith.addf %arg32, %141 : f32
          %143 = arith.mulf %109, %112 : f32
          %144 = arith.addf %arg33, %143 : f32
          %145 = arith.mulf %109, %114 : f32
          %146 = arith.addf %arg34, %145 : f32
          %147 = arith.mulf %109, %116 : f32
          %148 = arith.addf %arg35, %147 : f32
          %149 = arith.mulf %109, %118 : f32
          %150 = arith.addf %arg36, %149 : f32
          scf.yield %120, %122, %124, %126, %128, %130, %132, %134, %136, %138, %140, %142, %144, %146, %148, %150 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
        }
        gpu.barrier
        scf.yield %101#0, %101#1, %101#2, %101#3, %101#4, %101#5, %101#6, %101#7, %101#8, %101#9, %101#10, %101#11, %101#12, %101#13, %101#14, %101#15 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
      }
      %17 = arith.addi %4, %2 : index
      %18 = arith.muli %17, %c750 : index
      %19 = arith.addi %5, %3 : index
      %20 = arith.addi %18, %19 : index
      %21 = arith.cmpf ogt, %16#0, %cst : f32
      %22 = arith.select %21, %16#0, %cst : f32
      %23 = arith.cmpi ult, %17, %c1000 : index
      %24 = arith.cmpi ult, %19, %c750 : index
      %25 = arith.andi %23, %24 : i1
      scf.if %25 {
        memref.store %22, %arg2[%20] : memref<750000xf32, #spirv.storage_class<StorageBuffer>>
      }
      %26 = arith.addi %5, %13 : index
      %27 = arith.addi %18, %26 : index
      %28 = arith.cmpf ogt, %16#1, %cst : f32
      %29 = arith.select %28, %16#1, %cst : f32
      %30 = arith.cmpi ult, %26, %c750 : index
      %31 = arith.andi %23, %30 : i1
      scf.if %31 {
        memref.store %29, %arg2[%27] : memref<750000xf32, #spirv.storage_class<StorageBuffer>>
      }
      %32 = arith.addi %5, %14 : index
      %33 = arith.addi %18, %32 : index
      %34 = arith.cmpf ogt, %16#2, %cst : f32
      %35 = arith.select %34, %16#2, %cst : f32
      %36 = arith.cmpi ult, %32, %c750 : index
      %37 = arith.andi %23, %36 : i1
      scf.if %37 {
        memref.store %35, %arg2[%33] : memref<750000xf32, #spirv.storage_class<StorageBuffer>>
      }
      %38 = arith.addi %5, %15 : index
      %39 = arith.addi %18, %38 : index
      %40 = arith.cmpf ogt, %16#3, %cst : f32
      %41 = arith.select %40, %16#3, %cst : f32
      %42 = arith.cmpi ult, %38, %c750 : index
      %43 = arith.andi %23, %42 : i1
      scf.if %43 {
        memref.store %41, %arg2[%39] : memref<750000xf32, #spirv.storage_class<StorageBuffer>>
      }
      %44 = arith.addi %4, %7 : index
      %45 = arith.muli %44, %c750 : index
      %46 = arith.addi %45, %19 : index
      %47 = arith.cmpf ogt, %16#4, %cst : f32
      %48 = arith.select %47, %16#4, %cst : f32
      %49 = arith.cmpi ult, %44, %c1000 : index
      %50 = arith.andi %49, %24 : i1
      scf.if %50 {
        memref.store %48, %arg2[%46] : memref<750000xf32, #spirv.storage_class<StorageBuffer>>
      }
      %51 = arith.addi %45, %26 : index
      %52 = arith.cmpf ogt, %16#5, %cst : f32
      %53 = arith.select %52, %16#5, %cst : f32
      %54 = arith.andi %49, %30 : i1
      scf.if %54 {
        memref.store %53, %arg2[%51] : memref<750000xf32, #spirv.storage_class<StorageBuffer>>
      }
      %55 = arith.addi %45, %32 : index
      %56 = arith.cmpf ogt, %16#6, %cst : f32
      %57 = arith.select %56, %16#6, %cst : f32
      %58 = arith.andi %49, %36 : i1
      scf.if %58 {
        memref.store %57, %arg2[%55] : memref<750000xf32, #spirv.storage_class<StorageBuffer>>
      }
      %59 = arith.addi %45, %38 : index
      %60 = arith.cmpf ogt, %16#7, %cst : f32
      %61 = arith.select %60, %16#7, %cst : f32
      %62 = arith.andi %49, %42 : i1
      scf.if %62 {
        memref.store %61, %arg2[%59] : memref<750000xf32, #spirv.storage_class<StorageBuffer>>
      }
      %63 = arith.addi %4, %9 : index
      %64 = arith.muli %63, %c750 : index
      %65 = arith.addi %64, %19 : index
      %66 = arith.cmpf ogt, %16#8, %cst : f32
      %67 = arith.select %66, %16#8, %cst : f32
      %68 = arith.cmpi ult, %63, %c1000 : index
      %69 = arith.andi %68, %24 : i1
      scf.if %69 {
        memref.store %67, %arg2[%65] : memref<750000xf32, #spirv.storage_class<StorageBuffer>>
      }
      %70 = arith.addi %64, %26 : index
      %71 = arith.cmpf ogt, %16#9, %cst : f32
      %72 = arith.select %71, %16#9, %cst : f32
      %73 = arith.andi %68, %30 : i1
      scf.if %73 {
        memref.store %72, %arg2[%70] : memref<750000xf32, #spirv.storage_class<StorageBuffer>>
      }
      %74 = arith.addi %64, %32 : index
      %75 = arith.cmpf ogt, %16#10, %cst : f32
      %76 = arith.select %75, %16#10, %cst : f32
      %77 = arith.andi %68, %36 : i1
      scf.if %77 {
        memref.store %76, %arg2[%74] : memref<750000xf32, #spirv.storage_class<StorageBuffer>>
      }
      %78 = arith.addi %64, %38 : index
      %79 = arith.cmpf ogt, %16#11, %cst : f32
      %80 = arith.select %79, %16#11, %cst : f32
      %81 = arith.andi %68, %42 : i1
      scf.if %81 {
        memref.store %80, %arg2[%78] : memref<750000xf32, #spirv.storage_class<StorageBuffer>>
      }
      %82 = arith.addi %4, %11 : index
      %83 = arith.muli %82, %c750 : index
      %84 = arith.addi %83, %19 : index
      %85 = arith.cmpf ogt, %16#12, %cst : f32
      %86 = arith.select %85, %16#12, %cst : f32
      %87 = arith.cmpi ult, %82, %c1000 : index
      %88 = arith.andi %87, %24 : i1
      scf.if %88 {
        memref.store %86, %arg2[%84] : memref<750000xf32, #spirv.storage_class<StorageBuffer>>
      }
      %89 = arith.addi %83, %26 : index
      %90 = arith.cmpf ogt, %16#13, %cst : f32
      %91 = arith.select %90, %16#13, %cst : f32
      %92 = arith.andi %87, %30 : i1
      scf.if %92 {
        memref.store %91, %arg2[%89] : memref<750000xf32, #spirv.storage_class<StorageBuffer>>
      }
      %93 = arith.addi %83, %32 : index
      %94 = arith.cmpf ogt, %16#14, %cst : f32
      %95 = arith.select %94, %16#14, %cst : f32
      %96 = arith.andi %87, %36 : i1
      scf.if %96 {
        memref.store %95, %arg2[%93] : memref<750000xf32, #spirv.storage_class<StorageBuffer>>
      }
      %97 = arith.addi %83, %38 : index
      %98 = arith.cmpf ogt, %16#15, %cst : f32
      %99 = arith.select %98, %16#15, %cst : f32
      %100 = arith.andi %87, %42 : i1
      scf.if %100 {
        memref.store %99, %arg2[%97] : memref<750000xf32, #spirv.storage_class<StorageBuffer>>
      }
      gpu.return
    }
  }
}

