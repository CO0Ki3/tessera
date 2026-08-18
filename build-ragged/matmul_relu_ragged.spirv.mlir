  spirv.module @__spv__kernels Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Matrix], [SPV_KHR_storage_buffer_storage_class]> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin__LocalInvocationId__ built_in("LocalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
    spirv.GlobalVariable @__builtin__WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi32>, Input>
    spirv.GlobalVariable @__workgroup_mem__1 : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>
    spirv.GlobalVariable @__workgroup_mem__0 : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>
    spirv.GlobalVariable @matmul_relu_ragged_arg_0 bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.array<500000 x f32, stride=4> [0])>, StorageBuffer>
    spirv.GlobalVariable @matmul_relu_ragged_arg_1 bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.array<375000 x f32, stride=4> [0])>, StorageBuffer>
    spirv.GlobalVariable @matmul_relu_ragged_arg_2 bind(0, 2) : !spirv.ptr<!spirv.struct<(!spirv.array<750000 x f32, stride=4> [0])>, StorageBuffer>
    spirv.func @matmul_relu_ragged() "None" attributes {workgroup_attributions = 0 : i64} {
      %matmul_relu_ragged_arg_0_addr = spirv.mlir.addressof @matmul_relu_ragged_arg_0 : !spirv.ptr<!spirv.struct<(!spirv.array<500000 x f32, stride=4> [0])>, StorageBuffer>
      %matmul_relu_ragged_arg_1_addr = spirv.mlir.addressof @matmul_relu_ragged_arg_1 : !spirv.ptr<!spirv.struct<(!spirv.array<375000 x f32, stride=4> [0])>, StorageBuffer>
      %matmul_relu_ragged_arg_2_addr = spirv.mlir.addressof @matmul_relu_ragged_arg_2 : !spirv.ptr<!spirv.struct<(!spirv.array<750000 x f32, stride=4> [0])>, StorageBuffer>
      %cst0_i32 = spirv.Constant 0 : i32
      %cst1_i32 = spirv.Constant 1 : i32
      %cst2_i32 = spirv.Constant 2 : i32
      %cst3_i32 = spirv.Constant 3 : i32
      %cst4_i32 = spirv.Constant 4 : i32
      %cst16_i32 = spirv.Constant 16 : i32
      %cst64_i32 = spirv.Constant 64 : i32
      %cst256_i32 = spirv.Constant 256 : i32
      %cst500_i32 = spirv.Constant 500 : i32
      %cst750_i32 = spirv.Constant 750 : i32
      %cst1000_i32 = spirv.Constant 1000 : i32
      %cst1024_i32 = spirv.Constant 1024 : i32
      %cst374999_i32 = spirv.Constant 374999 : i32
      %cst499999_i32 = spirv.Constant 499999 : i32
      %cst_f32 = spirv.Constant 0.000000e+00 : f32
      %__workgroup_mem__0_addr = spirv.mlir.addressof @__workgroup_mem__0 : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>
      %__workgroup_mem__1_addr = spirv.mlir.addressof @__workgroup_mem__1 : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>
      %__builtin__WorkgroupId___addr = spirv.mlir.addressof @__builtin__WorkgroupId__ : !spirv.ptr<vector<3xi32>, Input>
      %0 = spirv.Load "Input" %__builtin__WorkgroupId___addr : vector<3xi32>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi32>
      %__builtin__WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin__WorkgroupId__ : !spirv.ptr<vector<3xi32>, Input>
      %2 = spirv.Load "Input" %__builtin__WorkgroupId___addr_0 : vector<3xi32>
      %3 = spirv.CompositeExtract %2[1 : i32] : vector<3xi32>
      %__builtin__LocalInvocationId___addr = spirv.mlir.addressof @__builtin__LocalInvocationId__ : !spirv.ptr<vector<3xi32>, Input>
      %4 = spirv.Load "Input" %__builtin__LocalInvocationId___addr : vector<3xi32>
      %5 = spirv.CompositeExtract %4[0 : i32] : vector<3xi32>
      %__builtin__LocalInvocationId___addr_1 = spirv.mlir.addressof @__builtin__LocalInvocationId__ : !spirv.ptr<vector<3xi32>, Input>
      %6 = spirv.Load "Input" %__builtin__LocalInvocationId___addr_1 : vector<3xi32>
      %7 = spirv.CompositeExtract %6[1 : i32] : vector<3xi32>
      %8 = spirv.IMul %7, %cst16_i32 : i32
      %9 = spirv.IAdd %8, %5 : i32
      %10 = spirv.IMul %7, %cst4_i32 : i32
      %11 = spirv.IMul %5, %cst4_i32 : i32
      %12 = spirv.IMul %3, %cst64_i32 : i32
      %13 = spirv.IMul %1, %cst64_i32 : i32
      %14 = spirv.IMul %7, %cst64_i32 : i32
      %15 = spirv.IAdd %10, %cst1_i32 : i32
      %16 = spirv.IMul %15, %cst16_i32 : i32
      %17 = spirv.IAdd %10, %cst2_i32 : i32
      %18 = spirv.IMul %17, %cst16_i32 : i32
      %19 = spirv.IAdd %10, %cst3_i32 : i32
      %20 = spirv.IMul %19, %cst16_i32 : i32
      %21 = spirv.IAdd %11, %cst1_i32 : i32
      %22 = spirv.IAdd %11, %cst2_i32 : i32
      %23 = spirv.IAdd %11, %cst3_i32 : i32
      %24 = spirv.Variable : !spirv.ptr<f32, Function>
      %25 = spirv.Variable : !spirv.ptr<f32, Function>
      %26 = spirv.Variable : !spirv.ptr<f32, Function>
      %27 = spirv.Variable : !spirv.ptr<f32, Function>
      %28 = spirv.Variable : !spirv.ptr<f32, Function>
      %29 = spirv.Variable : !spirv.ptr<f32, Function>
      %30 = spirv.Variable : !spirv.ptr<f32, Function>
      %31 = spirv.Variable : !spirv.ptr<f32, Function>
      %32 = spirv.Variable : !spirv.ptr<f32, Function>
      %33 = spirv.Variable : !spirv.ptr<f32, Function>
      %34 = spirv.Variable : !spirv.ptr<f32, Function>
      %35 = spirv.Variable : !spirv.ptr<f32, Function>
      %36 = spirv.Variable : !spirv.ptr<f32, Function>
      %37 = spirv.Variable : !spirv.ptr<f32, Function>
      %38 = spirv.Variable : !spirv.ptr<f32, Function>
      %39 = spirv.Variable : !spirv.ptr<f32, Function>
      spirv.mlir.loop {
        spirv.Branch ^bb1(%cst0_i32, %cst_f32, %cst_f32, %cst_f32, %cst_f32, %cst_f32, %cst_f32, %cst_f32, %cst_f32, %cst_f32, %cst_f32, %cst_f32, %cst_f32, %cst_f32, %cst_f32, %cst_f32, %cst_f32 : i32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)
      ^bb1(%140: i32, %141: f32, %142: f32, %143: f32, %144: f32, %145: f32, %146: f32, %147: f32, %148: f32, %149: f32, %150: f32, %151: f32, %152: f32, %153: f32, %154: f32, %155: f32, %156: f32):  // 2 preds: ^bb0, ^bb2
        %157 = spirv.SLessThan %140, %cst500_i32 : i32
        spirv.BranchConditional %157, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        spirv.mlir.loop {
          spirv.Branch ^bb1(%9 : i32)
        ^bb1(%191: i32):  // 2 preds: ^bb0, ^bb2
          %192 = spirv.SLessThan %191, %cst1024_i32 : i32
          spirv.BranchConditional %192, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          %193 = spirv.UDiv %191, %cst16_i32 : i32
          %194 = spirv.UMod %191, %cst16_i32 : i32
          %195 = spirv.IAdd %12, %193 : i32
          %196 = spirv.IMul %195, %cst500_i32 : i32
          %197 = spirv.IAdd %140, %194 : i32
          %198 = spirv.IAdd %196, %197 : i32
          %199 = spirv.ULessThan %195, %cst1000_i32 : i32
          %200 = spirv.ULessThan %197, %cst500_i32 : i32
          %201 = spirv.LogicalAnd %199, %200 : i1
          %202 = spirv.GL.UMin %198, %cst499999_i32 : i32
          %cst0_i32_2 = spirv.Constant 0 : i32
          %cst0_i32_3 = spirv.Constant 0 : i32
          %cst1_i32_4 = spirv.Constant 1 : i32
          %203 = spirv.AccessChain %matmul_relu_ragged_arg_0_addr[%cst0_i32_2, %202] : !spirv.ptr<!spirv.struct<(!spirv.array<500000 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
          %204 = spirv.Load "StorageBuffer" %203 : f32
          %205 = spirv.Select %201, %204, %cst_f32 : i1, f32
          %cst0_i32_5 = spirv.Constant 0 : i32
          %cst0_i32_6 = spirv.Constant 0 : i32
          %cst1_i32_7 = spirv.Constant 1 : i32
          %206 = spirv.AccessChain %__workgroup_mem__0_addr[%cst0_i32_5, %191] : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>, i32, i32 -> !spirv.ptr<f32, Workgroup>
          spirv.Store "Workgroup" %206, %205 : f32
          %207 = spirv.IAdd %191, %cst256_i32 : i32
          spirv.Branch ^bb1(%207 : i32)
        ^bb3:  // pred: ^bb1
          spirv.mlir.merge
        }
        spirv.mlir.loop {
          spirv.Branch ^bb1(%9 : i32)
        ^bb1(%191: i32):  // 2 preds: ^bb0, ^bb2
          %192 = spirv.SLessThan %191, %cst1024_i32 : i32
          spirv.BranchConditional %192, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          %193 = spirv.UDiv %191, %cst64_i32 : i32
          %194 = spirv.UMod %191, %cst64_i32 : i32
          %195 = spirv.IAdd %140, %193 : i32
          %196 = spirv.IMul %195, %cst750_i32 : i32
          %197 = spirv.IAdd %13, %194 : i32
          %198 = spirv.IAdd %196, %197 : i32
          %199 = spirv.ULessThan %195, %cst500_i32 : i32
          %200 = spirv.ULessThan %197, %cst750_i32 : i32
          %201 = spirv.LogicalAnd %199, %200 : i1
          %202 = spirv.GL.UMin %198, %cst374999_i32 : i32
          %cst0_i32_2 = spirv.Constant 0 : i32
          %cst0_i32_3 = spirv.Constant 0 : i32
          %cst1_i32_4 = spirv.Constant 1 : i32
          %203 = spirv.AccessChain %matmul_relu_ragged_arg_1_addr[%cst0_i32_2, %202] : !spirv.ptr<!spirv.struct<(!spirv.array<375000 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
          %204 = spirv.Load "StorageBuffer" %203 : f32
          %205 = spirv.Select %201, %204, %cst_f32 : i1, f32
          %cst0_i32_5 = spirv.Constant 0 : i32
          %cst0_i32_6 = spirv.Constant 0 : i32
          %cst1_i32_7 = spirv.Constant 1 : i32
          %206 = spirv.AccessChain %__workgroup_mem__1_addr[%cst0_i32_5, %191] : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>, i32, i32 -> !spirv.ptr<f32, Workgroup>
          spirv.Store "Workgroup" %206, %205 : f32
          %207 = spirv.IAdd %191, %cst256_i32 : i32
          spirv.Branch ^bb1(%207 : i32)
        ^bb3:  // pred: ^bb1
          spirv.mlir.merge
        }
        spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
        %158 = spirv.Variable : !spirv.ptr<f32, Function>
        %159 = spirv.Variable : !spirv.ptr<f32, Function>
        %160 = spirv.Variable : !spirv.ptr<f32, Function>
        %161 = spirv.Variable : !spirv.ptr<f32, Function>
        %162 = spirv.Variable : !spirv.ptr<f32, Function>
        %163 = spirv.Variable : !spirv.ptr<f32, Function>
        %164 = spirv.Variable : !spirv.ptr<f32, Function>
        %165 = spirv.Variable : !spirv.ptr<f32, Function>
        %166 = spirv.Variable : !spirv.ptr<f32, Function>
        %167 = spirv.Variable : !spirv.ptr<f32, Function>
        %168 = spirv.Variable : !spirv.ptr<f32, Function>
        %169 = spirv.Variable : !spirv.ptr<f32, Function>
        %170 = spirv.Variable : !spirv.ptr<f32, Function>
        %171 = spirv.Variable : !spirv.ptr<f32, Function>
        %172 = spirv.Variable : !spirv.ptr<f32, Function>
        %173 = spirv.Variable : !spirv.ptr<f32, Function>
        spirv.mlir.loop {
          spirv.Branch ^bb1(%cst0_i32, %141, %142, %143, %144, %145, %146, %147, %148, %149, %150, %151, %152, %153, %154, %155, %156 : i32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)
        ^bb1(%191: i32, %192: f32, %193: f32, %194: f32, %195: f32, %196: f32, %197: f32, %198: f32, %199: f32, %200: f32, %201: f32, %202: f32, %203: f32, %204: f32, %205: f32, %206: f32, %207: f32):  // 2 preds: ^bb0, ^bb2
          %208 = spirv.SLessThan %191, %cst16_i32 : i32
          spirv.BranchConditional %208, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          %209 = spirv.IAdd %14, %191 : i32
          %cst0_i32_2 = spirv.Constant 0 : i32
          %cst0_i32_3 = spirv.Constant 0 : i32
          %cst1_i32_4 = spirv.Constant 1 : i32
          %210 = spirv.AccessChain %__workgroup_mem__0_addr[%cst0_i32_2, %209] : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>, i32, i32 -> !spirv.ptr<f32, Workgroup>
          %211 = spirv.Load "Workgroup" %210 : f32
          %212 = spirv.IAdd %16, %191 : i32
          %cst0_i32_5 = spirv.Constant 0 : i32
          %cst0_i32_6 = spirv.Constant 0 : i32
          %cst1_i32_7 = spirv.Constant 1 : i32
          %213 = spirv.AccessChain %__workgroup_mem__0_addr[%cst0_i32_5, %212] : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>, i32, i32 -> !spirv.ptr<f32, Workgroup>
          %214 = spirv.Load "Workgroup" %213 : f32
          %215 = spirv.IAdd %18, %191 : i32
          %cst0_i32_8 = spirv.Constant 0 : i32
          %cst0_i32_9 = spirv.Constant 0 : i32
          %cst1_i32_10 = spirv.Constant 1 : i32
          %216 = spirv.AccessChain %__workgroup_mem__0_addr[%cst0_i32_8, %215] : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>, i32, i32 -> !spirv.ptr<f32, Workgroup>
          %217 = spirv.Load "Workgroup" %216 : f32
          %218 = spirv.IAdd %20, %191 : i32
          %cst0_i32_11 = spirv.Constant 0 : i32
          %cst0_i32_12 = spirv.Constant 0 : i32
          %cst1_i32_13 = spirv.Constant 1 : i32
          %219 = spirv.AccessChain %__workgroup_mem__0_addr[%cst0_i32_11, %218] : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>, i32, i32 -> !spirv.ptr<f32, Workgroup>
          %220 = spirv.Load "Workgroup" %219 : f32
          %221 = spirv.IMul %191, %cst64_i32 : i32
          %222 = spirv.IAdd %221, %11 : i32
          %cst0_i32_14 = spirv.Constant 0 : i32
          %cst0_i32_15 = spirv.Constant 0 : i32
          %cst1_i32_16 = spirv.Constant 1 : i32
          %223 = spirv.AccessChain %__workgroup_mem__1_addr[%cst0_i32_14, %222] : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>, i32, i32 -> !spirv.ptr<f32, Workgroup>
          %224 = spirv.Load "Workgroup" %223 : f32
          %225 = spirv.IAdd %221, %21 : i32
          %cst0_i32_17 = spirv.Constant 0 : i32
          %cst0_i32_18 = spirv.Constant 0 : i32
          %cst1_i32_19 = spirv.Constant 1 : i32
          %226 = spirv.AccessChain %__workgroup_mem__1_addr[%cst0_i32_17, %225] : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>, i32, i32 -> !spirv.ptr<f32, Workgroup>
          %227 = spirv.Load "Workgroup" %226 : f32
          %228 = spirv.IAdd %221, %22 : i32
          %cst0_i32_20 = spirv.Constant 0 : i32
          %cst0_i32_21 = spirv.Constant 0 : i32
          %cst1_i32_22 = spirv.Constant 1 : i32
          %229 = spirv.AccessChain %__workgroup_mem__1_addr[%cst0_i32_20, %228] : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>, i32, i32 -> !spirv.ptr<f32, Workgroup>
          %230 = spirv.Load "Workgroup" %229 : f32
          %231 = spirv.IAdd %221, %23 : i32
          %cst0_i32_23 = spirv.Constant 0 : i32
          %cst0_i32_24 = spirv.Constant 0 : i32
          %cst1_i32_25 = spirv.Constant 1 : i32
          %232 = spirv.AccessChain %__workgroup_mem__1_addr[%cst0_i32_23, %231] : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>, i32, i32 -> !spirv.ptr<f32, Workgroup>
          %233 = spirv.Load "Workgroup" %232 : f32
          %234 = spirv.FMul %211, %224 : f32
          %235 = spirv.FAdd %192, %234 : f32
          %236 = spirv.FMul %211, %227 : f32
          %237 = spirv.FAdd %193, %236 : f32
          %238 = spirv.FMul %211, %230 : f32
          %239 = spirv.FAdd %194, %238 : f32
          %240 = spirv.FMul %211, %233 : f32
          %241 = spirv.FAdd %195, %240 : f32
          %242 = spirv.FMul %214, %224 : f32
          %243 = spirv.FAdd %196, %242 : f32
          %244 = spirv.FMul %214, %227 : f32
          %245 = spirv.FAdd %197, %244 : f32
          %246 = spirv.FMul %214, %230 : f32
          %247 = spirv.FAdd %198, %246 : f32
          %248 = spirv.FMul %214, %233 : f32
          %249 = spirv.FAdd %199, %248 : f32
          %250 = spirv.FMul %217, %224 : f32
          %251 = spirv.FAdd %200, %250 : f32
          %252 = spirv.FMul %217, %227 : f32
          %253 = spirv.FAdd %201, %252 : f32
          %254 = spirv.FMul %217, %230 : f32
          %255 = spirv.FAdd %202, %254 : f32
          %256 = spirv.FMul %217, %233 : f32
          %257 = spirv.FAdd %203, %256 : f32
          %258 = spirv.FMul %220, %224 : f32
          %259 = spirv.FAdd %204, %258 : f32
          %260 = spirv.FMul %220, %227 : f32
          %261 = spirv.FAdd %205, %260 : f32
          %262 = spirv.FMul %220, %230 : f32
          %263 = spirv.FAdd %206, %262 : f32
          %264 = spirv.FMul %220, %233 : f32
          %265 = spirv.FAdd %207, %264 : f32
          spirv.Store "Function" %158, %235 : f32
          spirv.Store "Function" %159, %237 : f32
          spirv.Store "Function" %160, %239 : f32
          spirv.Store "Function" %161, %241 : f32
          spirv.Store "Function" %162, %243 : f32
          spirv.Store "Function" %163, %245 : f32
          spirv.Store "Function" %164, %247 : f32
          spirv.Store "Function" %165, %249 : f32
          spirv.Store "Function" %166, %251 : f32
          spirv.Store "Function" %167, %253 : f32
          spirv.Store "Function" %168, %255 : f32
          spirv.Store "Function" %169, %257 : f32
          spirv.Store "Function" %170, %259 : f32
          spirv.Store "Function" %171, %261 : f32
          spirv.Store "Function" %172, %263 : f32
          spirv.Store "Function" %173, %265 : f32
          %266 = spirv.IAdd %191, %cst1_i32 : i32
          spirv.Branch ^bb1(%266, %235, %237, %239, %241, %243, %245, %247, %249, %251, %253, %255, %257, %259, %261, %263, %265 : i32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)
        ^bb3:  // pred: ^bb1
          spirv.mlir.merge
        }
        %174 = spirv.Load "Function" %173 : f32
        %175 = spirv.Load "Function" %172 : f32
        %176 = spirv.Load "Function" %171 : f32
        %177 = spirv.Load "Function" %170 : f32
        %178 = spirv.Load "Function" %169 : f32
        %179 = spirv.Load "Function" %168 : f32
        %180 = spirv.Load "Function" %167 : f32
        %181 = spirv.Load "Function" %166 : f32
        %182 = spirv.Load "Function" %165 : f32
        %183 = spirv.Load "Function" %164 : f32
        %184 = spirv.Load "Function" %163 : f32
        %185 = spirv.Load "Function" %162 : f32
        %186 = spirv.Load "Function" %161 : f32
        %187 = spirv.Load "Function" %160 : f32
        %188 = spirv.Load "Function" %159 : f32
        %189 = spirv.Load "Function" %158 : f32
        spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
        spirv.Store "Function" %24, %189 : f32
        spirv.Store "Function" %25, %188 : f32
        spirv.Store "Function" %26, %187 : f32
        spirv.Store "Function" %27, %186 : f32
        spirv.Store "Function" %28, %185 : f32
        spirv.Store "Function" %29, %184 : f32
        spirv.Store "Function" %30, %183 : f32
        spirv.Store "Function" %31, %182 : f32
        spirv.Store "Function" %32, %181 : f32
        spirv.Store "Function" %33, %180 : f32
        spirv.Store "Function" %34, %179 : f32
        spirv.Store "Function" %35, %178 : f32
        spirv.Store "Function" %36, %177 : f32
        spirv.Store "Function" %37, %176 : f32
        spirv.Store "Function" %38, %175 : f32
        spirv.Store "Function" %39, %174 : f32
        %190 = spirv.IAdd %140, %cst16_i32 : i32
        spirv.Branch ^bb1(%190, %189, %188, %187, %186, %185, %184, %183, %182, %181, %180, %179, %178, %177, %176, %175, %174 : i32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)
      ^bb3:  // pred: ^bb1
        spirv.mlir.merge
      }
      %40 = spirv.Load "Function" %39 : f32
      %41 = spirv.Load "Function" %38 : f32
      %42 = spirv.Load "Function" %37 : f32
      %43 = spirv.Load "Function" %36 : f32
      %44 = spirv.Load "Function" %35 : f32
      %45 = spirv.Load "Function" %34 : f32
      %46 = spirv.Load "Function" %33 : f32
      %47 = spirv.Load "Function" %32 : f32
      %48 = spirv.Load "Function" %31 : f32
      %49 = spirv.Load "Function" %30 : f32
      %50 = spirv.Load "Function" %29 : f32
      %51 = spirv.Load "Function" %28 : f32
      %52 = spirv.Load "Function" %27 : f32
      %53 = spirv.Load "Function" %26 : f32
      %54 = spirv.Load "Function" %25 : f32
      %55 = spirv.Load "Function" %24 : f32
      %56 = spirv.IAdd %12, %10 : i32
      %57 = spirv.IMul %56, %cst750_i32 : i32
      %58 = spirv.IAdd %13, %11 : i32
      %59 = spirv.IAdd %57, %58 : i32
      %60 = spirv.FOrdGreaterThan %55, %cst_f32 : f32
      %61 = spirv.Select %60, %55, %cst_f32 : i1, f32
      %62 = spirv.ULessThan %56, %cst1000_i32 : i32
      %63 = spirv.ULessThan %58, %cst750_i32 : i32
      %64 = spirv.LogicalAnd %62, %63 : i1
      spirv.mlir.selection {
        spirv.BranchConditional %64, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        %cst0_i32_2 = spirv.Constant 0 : i32
        %cst0_i32_3 = spirv.Constant 0 : i32
        %cst1_i32_4 = spirv.Constant 1 : i32
        %140 = spirv.AccessChain %matmul_relu_ragged_arg_2_addr[%cst0_i32_2, %59] : !spirv.ptr<!spirv.struct<(!spirv.array<750000 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
        spirv.Store "StorageBuffer" %140, %61 : f32
        spirv.Branch ^bb2
      ^bb2:  // 2 preds: ^bb0, ^bb1
        spirv.mlir.merge
      }
      %65 = spirv.IAdd %13, %21 : i32
      %66 = spirv.IAdd %57, %65 : i32
      %67 = spirv.FOrdGreaterThan %54, %cst_f32 : f32
      %68 = spirv.Select %67, %54, %cst_f32 : i1, f32
      %69 = spirv.ULessThan %65, %cst750_i32 : i32
      %70 = spirv.LogicalAnd %62, %69 : i1
      spirv.mlir.selection {
        spirv.BranchConditional %70, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        %cst0_i32_2 = spirv.Constant 0 : i32
        %cst0_i32_3 = spirv.Constant 0 : i32
        %cst1_i32_4 = spirv.Constant 1 : i32
        %140 = spirv.AccessChain %matmul_relu_ragged_arg_2_addr[%cst0_i32_2, %66] : !spirv.ptr<!spirv.struct<(!spirv.array<750000 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
        spirv.Store "StorageBuffer" %140, %68 : f32
        spirv.Branch ^bb2
      ^bb2:  // 2 preds: ^bb0, ^bb1
        spirv.mlir.merge
      }
      %71 = spirv.IAdd %13, %22 : i32
      %72 = spirv.IAdd %57, %71 : i32
      %73 = spirv.FOrdGreaterThan %53, %cst_f32 : f32
      %74 = spirv.Select %73, %53, %cst_f32 : i1, f32
      %75 = spirv.ULessThan %71, %cst750_i32 : i32
      %76 = spirv.LogicalAnd %62, %75 : i1
      spirv.mlir.selection {
        spirv.BranchConditional %76, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        %cst0_i32_2 = spirv.Constant 0 : i32
        %cst0_i32_3 = spirv.Constant 0 : i32
        %cst1_i32_4 = spirv.Constant 1 : i32
        %140 = spirv.AccessChain %matmul_relu_ragged_arg_2_addr[%cst0_i32_2, %72] : !spirv.ptr<!spirv.struct<(!spirv.array<750000 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
        spirv.Store "StorageBuffer" %140, %74 : f32
        spirv.Branch ^bb2
      ^bb2:  // 2 preds: ^bb0, ^bb1
        spirv.mlir.merge
      }
      %77 = spirv.IAdd %13, %23 : i32
      %78 = spirv.IAdd %57, %77 : i32
      %79 = spirv.FOrdGreaterThan %52, %cst_f32 : f32
      %80 = spirv.Select %79, %52, %cst_f32 : i1, f32
      %81 = spirv.ULessThan %77, %cst750_i32 : i32
      %82 = spirv.LogicalAnd %62, %81 : i1
      spirv.mlir.selection {
        spirv.BranchConditional %82, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        %cst0_i32_2 = spirv.Constant 0 : i32
        %cst0_i32_3 = spirv.Constant 0 : i32
        %cst1_i32_4 = spirv.Constant 1 : i32
        %140 = spirv.AccessChain %matmul_relu_ragged_arg_2_addr[%cst0_i32_2, %78] : !spirv.ptr<!spirv.struct<(!spirv.array<750000 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
        spirv.Store "StorageBuffer" %140, %80 : f32
        spirv.Branch ^bb2
      ^bb2:  // 2 preds: ^bb0, ^bb1
        spirv.mlir.merge
      }
      %83 = spirv.IAdd %12, %15 : i32
      %84 = spirv.IMul %83, %cst750_i32 : i32
      %85 = spirv.IAdd %84, %58 : i32
      %86 = spirv.FOrdGreaterThan %51, %cst_f32 : f32
      %87 = spirv.Select %86, %51, %cst_f32 : i1, f32
      %88 = spirv.ULessThan %83, %cst1000_i32 : i32
      %89 = spirv.LogicalAnd %88, %63 : i1
      spirv.mlir.selection {
        spirv.BranchConditional %89, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        %cst0_i32_2 = spirv.Constant 0 : i32
        %cst0_i32_3 = spirv.Constant 0 : i32
        %cst1_i32_4 = spirv.Constant 1 : i32
        %140 = spirv.AccessChain %matmul_relu_ragged_arg_2_addr[%cst0_i32_2, %85] : !spirv.ptr<!spirv.struct<(!spirv.array<750000 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
        spirv.Store "StorageBuffer" %140, %87 : f32
        spirv.Branch ^bb2
      ^bb2:  // 2 preds: ^bb0, ^bb1
        spirv.mlir.merge
      }
      %90 = spirv.IAdd %84, %65 : i32
      %91 = spirv.FOrdGreaterThan %50, %cst_f32 : f32
      %92 = spirv.Select %91, %50, %cst_f32 : i1, f32
      %93 = spirv.LogicalAnd %88, %69 : i1
      spirv.mlir.selection {
        spirv.BranchConditional %93, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        %cst0_i32_2 = spirv.Constant 0 : i32
        %cst0_i32_3 = spirv.Constant 0 : i32
        %cst1_i32_4 = spirv.Constant 1 : i32
        %140 = spirv.AccessChain %matmul_relu_ragged_arg_2_addr[%cst0_i32_2, %90] : !spirv.ptr<!spirv.struct<(!spirv.array<750000 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
        spirv.Store "StorageBuffer" %140, %92 : f32
        spirv.Branch ^bb2
      ^bb2:  // 2 preds: ^bb0, ^bb1
        spirv.mlir.merge
      }
      %94 = spirv.IAdd %84, %71 : i32
      %95 = spirv.FOrdGreaterThan %49, %cst_f32 : f32
      %96 = spirv.Select %95, %49, %cst_f32 : i1, f32
      %97 = spirv.LogicalAnd %88, %75 : i1
      spirv.mlir.selection {
        spirv.BranchConditional %97, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        %cst0_i32_2 = spirv.Constant 0 : i32
        %cst0_i32_3 = spirv.Constant 0 : i32
        %cst1_i32_4 = spirv.Constant 1 : i32
        %140 = spirv.AccessChain %matmul_relu_ragged_arg_2_addr[%cst0_i32_2, %94] : !spirv.ptr<!spirv.struct<(!spirv.array<750000 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
        spirv.Store "StorageBuffer" %140, %96 : f32
        spirv.Branch ^bb2
      ^bb2:  // 2 preds: ^bb0, ^bb1
        spirv.mlir.merge
      }
      %98 = spirv.IAdd %84, %77 : i32
      %99 = spirv.FOrdGreaterThan %48, %cst_f32 : f32
      %100 = spirv.Select %99, %48, %cst_f32 : i1, f32
      %101 = spirv.LogicalAnd %88, %81 : i1
      spirv.mlir.selection {
        spirv.BranchConditional %101, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        %cst0_i32_2 = spirv.Constant 0 : i32
        %cst0_i32_3 = spirv.Constant 0 : i32
        %cst1_i32_4 = spirv.Constant 1 : i32
        %140 = spirv.AccessChain %matmul_relu_ragged_arg_2_addr[%cst0_i32_2, %98] : !spirv.ptr<!spirv.struct<(!spirv.array<750000 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
        spirv.Store "StorageBuffer" %140, %100 : f32
        spirv.Branch ^bb2
      ^bb2:  // 2 preds: ^bb0, ^bb1
        spirv.mlir.merge
      }
      %102 = spirv.IAdd %12, %17 : i32
      %103 = spirv.IMul %102, %cst750_i32 : i32
      %104 = spirv.IAdd %103, %58 : i32
      %105 = spirv.FOrdGreaterThan %47, %cst_f32 : f32
      %106 = spirv.Select %105, %47, %cst_f32 : i1, f32
      %107 = spirv.ULessThan %102, %cst1000_i32 : i32
      %108 = spirv.LogicalAnd %107, %63 : i1
      spirv.mlir.selection {
        spirv.BranchConditional %108, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        %cst0_i32_2 = spirv.Constant 0 : i32
        %cst0_i32_3 = spirv.Constant 0 : i32
        %cst1_i32_4 = spirv.Constant 1 : i32
        %140 = spirv.AccessChain %matmul_relu_ragged_arg_2_addr[%cst0_i32_2, %104] : !spirv.ptr<!spirv.struct<(!spirv.array<750000 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
        spirv.Store "StorageBuffer" %140, %106 : f32
        spirv.Branch ^bb2
      ^bb2:  // 2 preds: ^bb0, ^bb1
        spirv.mlir.merge
      }
      %109 = spirv.IAdd %103, %65 : i32
      %110 = spirv.FOrdGreaterThan %46, %cst_f32 : f32
      %111 = spirv.Select %110, %46, %cst_f32 : i1, f32
      %112 = spirv.LogicalAnd %107, %69 : i1
      spirv.mlir.selection {
        spirv.BranchConditional %112, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        %cst0_i32_2 = spirv.Constant 0 : i32
        %cst0_i32_3 = spirv.Constant 0 : i32
        %cst1_i32_4 = spirv.Constant 1 : i32
        %140 = spirv.AccessChain %matmul_relu_ragged_arg_2_addr[%cst0_i32_2, %109] : !spirv.ptr<!spirv.struct<(!spirv.array<750000 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
        spirv.Store "StorageBuffer" %140, %111 : f32
        spirv.Branch ^bb2
      ^bb2:  // 2 preds: ^bb0, ^bb1
        spirv.mlir.merge
      }
      %113 = spirv.IAdd %103, %71 : i32
      %114 = spirv.FOrdGreaterThan %45, %cst_f32 : f32
      %115 = spirv.Select %114, %45, %cst_f32 : i1, f32
      %116 = spirv.LogicalAnd %107, %75 : i1
      spirv.mlir.selection {
        spirv.BranchConditional %116, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        %cst0_i32_2 = spirv.Constant 0 : i32
        %cst0_i32_3 = spirv.Constant 0 : i32
        %cst1_i32_4 = spirv.Constant 1 : i32
        %140 = spirv.AccessChain %matmul_relu_ragged_arg_2_addr[%cst0_i32_2, %113] : !spirv.ptr<!spirv.struct<(!spirv.array<750000 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
        spirv.Store "StorageBuffer" %140, %115 : f32
        spirv.Branch ^bb2
      ^bb2:  // 2 preds: ^bb0, ^bb1
        spirv.mlir.merge
      }
      %117 = spirv.IAdd %103, %77 : i32
      %118 = spirv.FOrdGreaterThan %44, %cst_f32 : f32
      %119 = spirv.Select %118, %44, %cst_f32 : i1, f32
      %120 = spirv.LogicalAnd %107, %81 : i1
      spirv.mlir.selection {
        spirv.BranchConditional %120, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        %cst0_i32_2 = spirv.Constant 0 : i32
        %cst0_i32_3 = spirv.Constant 0 : i32
        %cst1_i32_4 = spirv.Constant 1 : i32
        %140 = spirv.AccessChain %matmul_relu_ragged_arg_2_addr[%cst0_i32_2, %117] : !spirv.ptr<!spirv.struct<(!spirv.array<750000 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
        spirv.Store "StorageBuffer" %140, %119 : f32
        spirv.Branch ^bb2
      ^bb2:  // 2 preds: ^bb0, ^bb1
        spirv.mlir.merge
      }
      %121 = spirv.IAdd %12, %19 : i32
      %122 = spirv.IMul %121, %cst750_i32 : i32
      %123 = spirv.IAdd %122, %58 : i32
      %124 = spirv.FOrdGreaterThan %43, %cst_f32 : f32
      %125 = spirv.Select %124, %43, %cst_f32 : i1, f32
      %126 = spirv.ULessThan %121, %cst1000_i32 : i32
      %127 = spirv.LogicalAnd %126, %63 : i1
      spirv.mlir.selection {
        spirv.BranchConditional %127, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        %cst0_i32_2 = spirv.Constant 0 : i32
        %cst0_i32_3 = spirv.Constant 0 : i32
        %cst1_i32_4 = spirv.Constant 1 : i32
        %140 = spirv.AccessChain %matmul_relu_ragged_arg_2_addr[%cst0_i32_2, %123] : !spirv.ptr<!spirv.struct<(!spirv.array<750000 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
        spirv.Store "StorageBuffer" %140, %125 : f32
        spirv.Branch ^bb2
      ^bb2:  // 2 preds: ^bb0, ^bb1
        spirv.mlir.merge
      }
      %128 = spirv.IAdd %122, %65 : i32
      %129 = spirv.FOrdGreaterThan %42, %cst_f32 : f32
      %130 = spirv.Select %129, %42, %cst_f32 : i1, f32
      %131 = spirv.LogicalAnd %126, %69 : i1
      spirv.mlir.selection {
        spirv.BranchConditional %131, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        %cst0_i32_2 = spirv.Constant 0 : i32
        %cst0_i32_3 = spirv.Constant 0 : i32
        %cst1_i32_4 = spirv.Constant 1 : i32
        %140 = spirv.AccessChain %matmul_relu_ragged_arg_2_addr[%cst0_i32_2, %128] : !spirv.ptr<!spirv.struct<(!spirv.array<750000 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
        spirv.Store "StorageBuffer" %140, %130 : f32
        spirv.Branch ^bb2
      ^bb2:  // 2 preds: ^bb0, ^bb1
        spirv.mlir.merge
      }
      %132 = spirv.IAdd %122, %71 : i32
      %133 = spirv.FOrdGreaterThan %41, %cst_f32 : f32
      %134 = spirv.Select %133, %41, %cst_f32 : i1, f32
      %135 = spirv.LogicalAnd %126, %75 : i1
      spirv.mlir.selection {
        spirv.BranchConditional %135, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        %cst0_i32_2 = spirv.Constant 0 : i32
        %cst0_i32_3 = spirv.Constant 0 : i32
        %cst1_i32_4 = spirv.Constant 1 : i32
        %140 = spirv.AccessChain %matmul_relu_ragged_arg_2_addr[%cst0_i32_2, %132] : !spirv.ptr<!spirv.struct<(!spirv.array<750000 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
        spirv.Store "StorageBuffer" %140, %134 : f32
        spirv.Branch ^bb2
      ^bb2:  // 2 preds: ^bb0, ^bb1
        spirv.mlir.merge
      }
      %136 = spirv.IAdd %122, %77 : i32
      %137 = spirv.FOrdGreaterThan %40, %cst_f32 : f32
      %138 = spirv.Select %137, %40, %cst_f32 : i1, f32
      %139 = spirv.LogicalAnd %126, %81 : i1
      spirv.mlir.selection {
        spirv.BranchConditional %139, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        %cst0_i32_2 = spirv.Constant 0 : i32
        %cst0_i32_3 = spirv.Constant 0 : i32
        %cst1_i32_4 = spirv.Constant 1 : i32
        %140 = spirv.AccessChain %matmul_relu_ragged_arg_2_addr[%cst0_i32_2, %136] : !spirv.ptr<!spirv.struct<(!spirv.array<750000 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
        spirv.Store "StorageBuffer" %140, %138 : f32
        spirv.Branch ^bb2
      ^bb2:  // 2 preds: ^bb0, ^bb1
        spirv.mlir.merge
      }
      spirv.Return
    }
    spirv.EntryPoint "GLCompute" @matmul_relu_ragged, @__builtin__WorkgroupId__, @__builtin__LocalInvocationId__
    spirv.ExecutionMode @matmul_relu_ragged "LocalSize", 16, 16, 1
  }
