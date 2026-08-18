  spirv.module @__spv__kernels Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Matrix], [SPV_KHR_storage_buffer_storage_class]> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin__LocalInvocationId__ built_in("LocalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
    spirv.GlobalVariable @masked_arg_0 bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.array<64 x f32, stride=4> [0])>, StorageBuffer>
    spirv.GlobalVariable @masked_arg_1 bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.array<64 x f32, stride=4> [0])>, StorageBuffer>
    spirv.func @masked() "None" attributes {workgroup_attributions = 0 : i64} {
      %masked_arg_0_addr = spirv.mlir.addressof @masked_arg_0 : !spirv.ptr<!spirv.struct<(!spirv.array<64 x f32, stride=4> [0])>, StorageBuffer>
      %masked_arg_1_addr = spirv.mlir.addressof @masked_arg_1 : !spirv.ptr<!spirv.struct<(!spirv.array<64 x f32, stride=4> [0])>, StorageBuffer>
      %cst0_i32 = spirv.Constant 0 : i32
      %cst40_i32 = spirv.Constant 40 : i32
      %cst63_i32 = spirv.Constant 63 : i32
      %cst_f32 = spirv.Constant 0.000000e+00 : f32
      %__builtin__LocalInvocationId___addr = spirv.mlir.addressof @__builtin__LocalInvocationId__ : !spirv.ptr<vector<3xi32>, Input>
      %0 = spirv.Load "Input" %__builtin__LocalInvocationId___addr : vector<3xi32>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi32>
      %2 = spirv.ULessThan %1, %cst40_i32 : i32
      %3 = spirv.GL.UMin %1, %cst63_i32 : i32
      %cst0_i32_0 = spirv.Constant 0 : i32
      %cst0_i32_1 = spirv.Constant 0 : i32
      %cst1_i32 = spirv.Constant 1 : i32
      %4 = spirv.AccessChain %masked_arg_0_addr[%cst0_i32_0, %3] : !spirv.ptr<!spirv.struct<(!spirv.array<64 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      %5 = spirv.Load "StorageBuffer" %4 : f32
      %6 = spirv.Select %2, %5, %cst_f32 : i1, f32
      spirv.mlir.selection {
        spirv.BranchConditional %2, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        %cst0_i32_2 = spirv.Constant 0 : i32
        %cst0_i32_3 = spirv.Constant 0 : i32
        %cst1_i32_4 = spirv.Constant 1 : i32
        %7 = spirv.AccessChain %masked_arg_1_addr[%cst0_i32_2, %1] : !spirv.ptr<!spirv.struct<(!spirv.array<64 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
        spirv.Store "StorageBuffer" %7, %6 : f32
        spirv.Branch ^bb2
      ^bb2:  // 2 preds: ^bb0, ^bb1
        spirv.mlir.merge
      }
      spirv.Return
    }
    spirv.EntryPoint "GLCompute" @masked, @__builtin__LocalInvocationId__
    spirv.ExecutionMode @masked "LocalSize", 16, 1, 1
  }
