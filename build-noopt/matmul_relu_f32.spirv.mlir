  spirv.module @__spv__kernels Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Matrix], [SPV_KHR_storage_buffer_storage_class]> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin__LocalInvocationId__ built_in("LocalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
    spirv.GlobalVariable @__builtin__WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi32>, Input>
    spirv.GlobalVariable @__workgroup_mem__1 : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>
    spirv.GlobalVariable @__workgroup_mem__0 : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>
    spirv.GlobalVariable @matmul_relu_f32_arg_0 bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.array<524288 x f32, stride=4> [0])>, StorageBuffer>
    spirv.GlobalVariable @matmul_relu_f32_arg_1 bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.array<393216 x f32, stride=4> [0])>, StorageBuffer>
    spirv.GlobalVariable @matmul_relu_f32_arg_2 bind(0, 2) : !spirv.ptr<!spirv.struct<(!spirv.array<786432 x f32, stride=4> [0])>, StorageBuffer>
    spirv.func @matmul_relu_f32() "None" attributes {workgroup_attributions = 0 : i64} {
      %matmul_relu_f32_arg_0_addr = spirv.mlir.addressof @matmul_relu_f32_arg_0 : !spirv.ptr<!spirv.struct<(!spirv.array<524288 x f32, stride=4> [0])>, StorageBuffer>
      %matmul_relu_f32_arg_1_addr = spirv.mlir.addressof @matmul_relu_f32_arg_1 : !spirv.ptr<!spirv.struct<(!spirv.array<393216 x f32, stride=4> [0])>, StorageBuffer>
      %matmul_relu_f32_arg_2_addr = spirv.mlir.addressof @matmul_relu_f32_arg_2 : !spirv.ptr<!spirv.struct<(!spirv.array<786432 x f32, stride=4> [0])>, StorageBuffer>
      %cst0_i32 = spirv.Constant 0 : i32
      %cst1_i32 = spirv.Constant 1 : i32
      %cst2_i32 = spirv.Constant 2 : i32
      %cst3_i32 = spirv.Constant 3 : i32
      %cst4_i32 = spirv.Constant 4 : i32
      %cst16_i32 = spirv.Constant 16 : i32
      %cst64_i32 = spirv.Constant 64 : i32
      %cst256_i32 = spirv.Constant 256 : i32
      %cst512_i32 = spirv.Constant 512 : i32
      %cst768_i32 = spirv.Constant 768 : i32
      %cst1024_i32 = spirv.Constant 1024 : i32
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
      %14 = spirv.Variable : !spirv.ptr<f32, Function>
      %15 = spirv.Variable : !spirv.ptr<f32, Function>
      %16 = spirv.Variable : !spirv.ptr<f32, Function>
      %17 = spirv.Variable : !spirv.ptr<f32, Function>
      %18 = spirv.Variable : !spirv.ptr<f32, Function>
      %19 = spirv.Variable : !spirv.ptr<f32, Function>
      %20 = spirv.Variable : !spirv.ptr<f32, Function>
      %21 = spirv.Variable : !spirv.ptr<f32, Function>
      %22 = spirv.Variable : !spirv.ptr<f32, Function>
      %23 = spirv.Variable : !spirv.ptr<f32, Function>
      %24 = spirv.Variable : !spirv.ptr<f32, Function>
      %25 = spirv.Variable : !spirv.ptr<f32, Function>
      %26 = spirv.Variable : !spirv.ptr<f32, Function>
      %27 = spirv.Variable : !spirv.ptr<f32, Function>
      %28 = spirv.Variable : !spirv.ptr<f32, Function>
      %29 = spirv.Variable : !spirv.ptr<f32, Function>
      spirv.mlir.loop {
        spirv.Branch ^bb1(%cst0_i32, %cst_f32, %cst_f32, %cst_f32, %cst_f32, %cst_f32, %cst_f32, %cst_f32, %cst_f32, %cst_f32, %cst_f32, %cst_f32, %cst_f32, %cst_f32, %cst_f32, %cst_f32, %cst_f32 : i32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)
      ^bb1(%149: i32, %150: f32, %151: f32, %152: f32, %153: f32, %154: f32, %155: f32, %156: f32, %157: f32, %158: f32, %159: f32, %160: f32, %161: f32, %162: f32, %163: f32, %164: f32, %165: f32):  // 2 preds: ^bb0, ^bb2
        %166 = spirv.SLessThan %149, %cst512_i32 : i32
        spirv.BranchConditional %166, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        spirv.mlir.loop {
          spirv.Branch ^bb1(%9 : i32)
        ^bb1(%200: i32):  // 2 preds: ^bb0, ^bb2
          %201 = spirv.SLessThan %200, %cst1024_i32 : i32
          spirv.BranchConditional %201, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          %202 = spirv.UDiv %200, %cst16_i32 : i32
          %203 = spirv.UMod %200, %cst16_i32 : i32
          %204 = spirv.IAdd %12, %202 : i32
          %205 = spirv.IMul %204, %cst512_i32 : i32
          %206 = spirv.IAdd %149, %203 : i32
          %207 = spirv.IAdd %205, %206 : i32
          %cst0_i32_50 = spirv.Constant 0 : i32
          %cst0_i32_51 = spirv.Constant 0 : i32
          %cst1_i32_52 = spirv.Constant 1 : i32
          %208 = spirv.AccessChain %matmul_relu_f32_arg_0_addr[%cst0_i32_50, %207] : !spirv.ptr<!spirv.struct<(!spirv.array<524288 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
          %209 = spirv.Load "StorageBuffer" %208 : f32
          %cst0_i32_53 = spirv.Constant 0 : i32
          %cst0_i32_54 = spirv.Constant 0 : i32
          %cst1_i32_55 = spirv.Constant 1 : i32
          %210 = spirv.AccessChain %__workgroup_mem__0_addr[%cst0_i32_53, %200] : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>, i32, i32 -> !spirv.ptr<f32, Workgroup>
          spirv.Store "Workgroup" %210, %209 : f32
          %211 = spirv.IAdd %200, %cst256_i32 : i32
          spirv.Branch ^bb1(%211 : i32)
        ^bb3:  // pred: ^bb1
          spirv.mlir.merge
        }
        spirv.mlir.loop {
          spirv.Branch ^bb1(%9 : i32)
        ^bb1(%200: i32):  // 2 preds: ^bb0, ^bb2
          %201 = spirv.SLessThan %200, %cst1024_i32 : i32
          spirv.BranchConditional %201, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          %202 = spirv.UDiv %200, %cst64_i32 : i32
          %203 = spirv.UMod %200, %cst64_i32 : i32
          %204 = spirv.IAdd %149, %202 : i32
          %205 = spirv.IMul %204, %cst768_i32 : i32
          %206 = spirv.IAdd %13, %203 : i32
          %207 = spirv.IAdd %205, %206 : i32
          %cst0_i32_50 = spirv.Constant 0 : i32
          %cst0_i32_51 = spirv.Constant 0 : i32
          %cst1_i32_52 = spirv.Constant 1 : i32
          %208 = spirv.AccessChain %matmul_relu_f32_arg_1_addr[%cst0_i32_50, %207] : !spirv.ptr<!spirv.struct<(!spirv.array<393216 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
          %209 = spirv.Load "StorageBuffer" %208 : f32
          %cst0_i32_53 = spirv.Constant 0 : i32
          %cst0_i32_54 = spirv.Constant 0 : i32
          %cst1_i32_55 = spirv.Constant 1 : i32
          %210 = spirv.AccessChain %__workgroup_mem__1_addr[%cst0_i32_53, %200] : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>, i32, i32 -> !spirv.ptr<f32, Workgroup>
          spirv.Store "Workgroup" %210, %209 : f32
          %211 = spirv.IAdd %200, %cst256_i32 : i32
          spirv.Branch ^bb1(%211 : i32)
        ^bb3:  // pred: ^bb1
          spirv.mlir.merge
        }
        spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
        %167 = spirv.Variable : !spirv.ptr<f32, Function>
        %168 = spirv.Variable : !spirv.ptr<f32, Function>
        %169 = spirv.Variable : !spirv.ptr<f32, Function>
        %170 = spirv.Variable : !spirv.ptr<f32, Function>
        %171 = spirv.Variable : !spirv.ptr<f32, Function>
        %172 = spirv.Variable : !spirv.ptr<f32, Function>
        %173 = spirv.Variable : !spirv.ptr<f32, Function>
        %174 = spirv.Variable : !spirv.ptr<f32, Function>
        %175 = spirv.Variable : !spirv.ptr<f32, Function>
        %176 = spirv.Variable : !spirv.ptr<f32, Function>
        %177 = spirv.Variable : !spirv.ptr<f32, Function>
        %178 = spirv.Variable : !spirv.ptr<f32, Function>
        %179 = spirv.Variable : !spirv.ptr<f32, Function>
        %180 = spirv.Variable : !spirv.ptr<f32, Function>
        %181 = spirv.Variable : !spirv.ptr<f32, Function>
        %182 = spirv.Variable : !spirv.ptr<f32, Function>
        spirv.mlir.loop {
          spirv.Branch ^bb1(%cst0_i32, %150, %151, %152, %153, %154, %155, %156, %157, %158, %159, %160, %161, %162, %163, %164, %165 : i32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)
        ^bb1(%200: i32, %201: f32, %202: f32, %203: f32, %204: f32, %205: f32, %206: f32, %207: f32, %208: f32, %209: f32, %210: f32, %211: f32, %212: f32, %213: f32, %214: f32, %215: f32, %216: f32):  // 2 preds: ^bb0, ^bb2
          %217 = spirv.SLessThan %200, %cst16_i32 : i32
          spirv.BranchConditional %217, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          %218 = spirv.IMul %10, %cst16_i32 : i32
          %219 = spirv.IAdd %218, %200 : i32
          %cst0_i32_50 = spirv.Constant 0 : i32
          %cst0_i32_51 = spirv.Constant 0 : i32
          %cst1_i32_52 = spirv.Constant 1 : i32
          %220 = spirv.AccessChain %__workgroup_mem__0_addr[%cst0_i32_50, %219] : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>, i32, i32 -> !spirv.ptr<f32, Workgroup>
          %221 = spirv.Load "Workgroup" %220 : f32
          %222 = spirv.IAdd %10, %cst1_i32 : i32
          %223 = spirv.IMul %222, %cst16_i32 : i32
          %224 = spirv.IAdd %223, %200 : i32
          %cst0_i32_53 = spirv.Constant 0 : i32
          %cst0_i32_54 = spirv.Constant 0 : i32
          %cst1_i32_55 = spirv.Constant 1 : i32
          %225 = spirv.AccessChain %__workgroup_mem__0_addr[%cst0_i32_53, %224] : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>, i32, i32 -> !spirv.ptr<f32, Workgroup>
          %226 = spirv.Load "Workgroup" %225 : f32
          %227 = spirv.IAdd %10, %cst2_i32 : i32
          %228 = spirv.IMul %227, %cst16_i32 : i32
          %229 = spirv.IAdd %228, %200 : i32
          %cst0_i32_56 = spirv.Constant 0 : i32
          %cst0_i32_57 = spirv.Constant 0 : i32
          %cst1_i32_58 = spirv.Constant 1 : i32
          %230 = spirv.AccessChain %__workgroup_mem__0_addr[%cst0_i32_56, %229] : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>, i32, i32 -> !spirv.ptr<f32, Workgroup>
          %231 = spirv.Load "Workgroup" %230 : f32
          %232 = spirv.IAdd %10, %cst3_i32 : i32
          %233 = spirv.IMul %232, %cst16_i32 : i32
          %234 = spirv.IAdd %233, %200 : i32
          %cst0_i32_59 = spirv.Constant 0 : i32
          %cst0_i32_60 = spirv.Constant 0 : i32
          %cst1_i32_61 = spirv.Constant 1 : i32
          %235 = spirv.AccessChain %__workgroup_mem__0_addr[%cst0_i32_59, %234] : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>, i32, i32 -> !spirv.ptr<f32, Workgroup>
          %236 = spirv.Load "Workgroup" %235 : f32
          %237 = spirv.IMul %200, %cst64_i32 : i32
          %238 = spirv.IAdd %237, %11 : i32
          %cst0_i32_62 = spirv.Constant 0 : i32
          %cst0_i32_63 = spirv.Constant 0 : i32
          %cst1_i32_64 = spirv.Constant 1 : i32
          %239 = spirv.AccessChain %__workgroup_mem__1_addr[%cst0_i32_62, %238] : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>, i32, i32 -> !spirv.ptr<f32, Workgroup>
          %240 = spirv.Load "Workgroup" %239 : f32
          %241 = spirv.IMul %200, %cst64_i32 : i32
          %242 = spirv.IAdd %11, %cst1_i32 : i32
          %243 = spirv.IAdd %241, %242 : i32
          %cst0_i32_65 = spirv.Constant 0 : i32
          %cst0_i32_66 = spirv.Constant 0 : i32
          %cst1_i32_67 = spirv.Constant 1 : i32
          %244 = spirv.AccessChain %__workgroup_mem__1_addr[%cst0_i32_65, %243] : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>, i32, i32 -> !spirv.ptr<f32, Workgroup>
          %245 = spirv.Load "Workgroup" %244 : f32
          %246 = spirv.IMul %200, %cst64_i32 : i32
          %247 = spirv.IAdd %11, %cst2_i32 : i32
          %248 = spirv.IAdd %246, %247 : i32
          %cst0_i32_68 = spirv.Constant 0 : i32
          %cst0_i32_69 = spirv.Constant 0 : i32
          %cst1_i32_70 = spirv.Constant 1 : i32
          %249 = spirv.AccessChain %__workgroup_mem__1_addr[%cst0_i32_68, %248] : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>, i32, i32 -> !spirv.ptr<f32, Workgroup>
          %250 = spirv.Load "Workgroup" %249 : f32
          %251 = spirv.IMul %200, %cst64_i32 : i32
          %252 = spirv.IAdd %11, %cst3_i32 : i32
          %253 = spirv.IAdd %251, %252 : i32
          %cst0_i32_71 = spirv.Constant 0 : i32
          %cst0_i32_72 = spirv.Constant 0 : i32
          %cst1_i32_73 = spirv.Constant 1 : i32
          %254 = spirv.AccessChain %__workgroup_mem__1_addr[%cst0_i32_71, %253] : !spirv.ptr<!spirv.struct<(!spirv.array<1024 x f32>)>, Workgroup>, i32, i32 -> !spirv.ptr<f32, Workgroup>
          %255 = spirv.Load "Workgroup" %254 : f32
          %256 = spirv.FMul %221, %240 : f32
          %257 = spirv.FAdd %201, %256 : f32
          %258 = spirv.FMul %221, %245 : f32
          %259 = spirv.FAdd %202, %258 : f32
          %260 = spirv.FMul %221, %250 : f32
          %261 = spirv.FAdd %203, %260 : f32
          %262 = spirv.FMul %221, %255 : f32
          %263 = spirv.FAdd %204, %262 : f32
          %264 = spirv.FMul %226, %240 : f32
          %265 = spirv.FAdd %205, %264 : f32
          %266 = spirv.FMul %226, %245 : f32
          %267 = spirv.FAdd %206, %266 : f32
          %268 = spirv.FMul %226, %250 : f32
          %269 = spirv.FAdd %207, %268 : f32
          %270 = spirv.FMul %226, %255 : f32
          %271 = spirv.FAdd %208, %270 : f32
          %272 = spirv.FMul %231, %240 : f32
          %273 = spirv.FAdd %209, %272 : f32
          %274 = spirv.FMul %231, %245 : f32
          %275 = spirv.FAdd %210, %274 : f32
          %276 = spirv.FMul %231, %250 : f32
          %277 = spirv.FAdd %211, %276 : f32
          %278 = spirv.FMul %231, %255 : f32
          %279 = spirv.FAdd %212, %278 : f32
          %280 = spirv.FMul %236, %240 : f32
          %281 = spirv.FAdd %213, %280 : f32
          %282 = spirv.FMul %236, %245 : f32
          %283 = spirv.FAdd %214, %282 : f32
          %284 = spirv.FMul %236, %250 : f32
          %285 = spirv.FAdd %215, %284 : f32
          %286 = spirv.FMul %236, %255 : f32
          %287 = spirv.FAdd %216, %286 : f32
          spirv.Store "Function" %167, %257 : f32
          spirv.Store "Function" %168, %259 : f32
          spirv.Store "Function" %169, %261 : f32
          spirv.Store "Function" %170, %263 : f32
          spirv.Store "Function" %171, %265 : f32
          spirv.Store "Function" %172, %267 : f32
          spirv.Store "Function" %173, %269 : f32
          spirv.Store "Function" %174, %271 : f32
          spirv.Store "Function" %175, %273 : f32
          spirv.Store "Function" %176, %275 : f32
          spirv.Store "Function" %177, %277 : f32
          spirv.Store "Function" %178, %279 : f32
          spirv.Store "Function" %179, %281 : f32
          spirv.Store "Function" %180, %283 : f32
          spirv.Store "Function" %181, %285 : f32
          spirv.Store "Function" %182, %287 : f32
          %288 = spirv.IAdd %200, %cst1_i32 : i32
          spirv.Branch ^bb1(%288, %257, %259, %261, %263, %265, %267, %269, %271, %273, %275, %277, %279, %281, %283, %285, %287 : i32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)
        ^bb3:  // pred: ^bb1
          spirv.mlir.merge
        }
        %183 = spirv.Load "Function" %182 : f32
        %184 = spirv.Load "Function" %181 : f32
        %185 = spirv.Load "Function" %180 : f32
        %186 = spirv.Load "Function" %179 : f32
        %187 = spirv.Load "Function" %178 : f32
        %188 = spirv.Load "Function" %177 : f32
        %189 = spirv.Load "Function" %176 : f32
        %190 = spirv.Load "Function" %175 : f32
        %191 = spirv.Load "Function" %174 : f32
        %192 = spirv.Load "Function" %173 : f32
        %193 = spirv.Load "Function" %172 : f32
        %194 = spirv.Load "Function" %171 : f32
        %195 = spirv.Load "Function" %170 : f32
        %196 = spirv.Load "Function" %169 : f32
        %197 = spirv.Load "Function" %168 : f32
        %198 = spirv.Load "Function" %167 : f32
        spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
        spirv.Store "Function" %14, %198 : f32
        spirv.Store "Function" %15, %197 : f32
        spirv.Store "Function" %16, %196 : f32
        spirv.Store "Function" %17, %195 : f32
        spirv.Store "Function" %18, %194 : f32
        spirv.Store "Function" %19, %193 : f32
        spirv.Store "Function" %20, %192 : f32
        spirv.Store "Function" %21, %191 : f32
        spirv.Store "Function" %22, %190 : f32
        spirv.Store "Function" %23, %189 : f32
        spirv.Store "Function" %24, %188 : f32
        spirv.Store "Function" %25, %187 : f32
        spirv.Store "Function" %26, %186 : f32
        spirv.Store "Function" %27, %185 : f32
        spirv.Store "Function" %28, %184 : f32
        spirv.Store "Function" %29, %183 : f32
        %199 = spirv.IAdd %149, %cst16_i32 : i32
        spirv.Branch ^bb1(%199, %198, %197, %196, %195, %194, %193, %192, %191, %190, %189, %188, %187, %186, %185, %184, %183 : i32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)
      ^bb3:  // pred: ^bb1
        spirv.mlir.merge
      }
      %30 = spirv.Load "Function" %29 : f32
      %31 = spirv.Load "Function" %28 : f32
      %32 = spirv.Load "Function" %27 : f32
      %33 = spirv.Load "Function" %26 : f32
      %34 = spirv.Load "Function" %25 : f32
      %35 = spirv.Load "Function" %24 : f32
      %36 = spirv.Load "Function" %23 : f32
      %37 = spirv.Load "Function" %22 : f32
      %38 = spirv.Load "Function" %21 : f32
      %39 = spirv.Load "Function" %20 : f32
      %40 = spirv.Load "Function" %19 : f32
      %41 = spirv.Load "Function" %18 : f32
      %42 = spirv.Load "Function" %17 : f32
      %43 = spirv.Load "Function" %16 : f32
      %44 = spirv.Load "Function" %15 : f32
      %45 = spirv.Load "Function" %14 : f32
      %46 = spirv.IAdd %12, %10 : i32
      %47 = spirv.IMul %46, %cst768_i32 : i32
      %48 = spirv.IAdd %13, %11 : i32
      %49 = spirv.IAdd %47, %48 : i32
      %50 = spirv.FOrdGreaterThan %45, %cst_f32 : f32
      %51 = spirv.Select %50, %45, %cst_f32 : i1, f32
      %cst0_i32_2 = spirv.Constant 0 : i32
      %cst0_i32_3 = spirv.Constant 0 : i32
      %cst1_i32_4 = spirv.Constant 1 : i32
      %52 = spirv.AccessChain %matmul_relu_f32_arg_2_addr[%cst0_i32_2, %49] : !spirv.ptr<!spirv.struct<(!spirv.array<786432 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      spirv.Store "StorageBuffer" %52, %51 : f32
      %53 = spirv.IAdd %11, %cst1_i32 : i32
      %54 = spirv.IAdd %13, %53 : i32
      %55 = spirv.IAdd %47, %54 : i32
      %56 = spirv.FOrdGreaterThan %44, %cst_f32 : f32
      %57 = spirv.Select %56, %44, %cst_f32 : i1, f32
      %cst0_i32_5 = spirv.Constant 0 : i32
      %cst0_i32_6 = spirv.Constant 0 : i32
      %cst1_i32_7 = spirv.Constant 1 : i32
      %58 = spirv.AccessChain %matmul_relu_f32_arg_2_addr[%cst0_i32_5, %55] : !spirv.ptr<!spirv.struct<(!spirv.array<786432 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      spirv.Store "StorageBuffer" %58, %57 : f32
      %59 = spirv.IAdd %11, %cst2_i32 : i32
      %60 = spirv.IAdd %13, %59 : i32
      %61 = spirv.IAdd %47, %60 : i32
      %62 = spirv.FOrdGreaterThan %43, %cst_f32 : f32
      %63 = spirv.Select %62, %43, %cst_f32 : i1, f32
      %cst0_i32_8 = spirv.Constant 0 : i32
      %cst0_i32_9 = spirv.Constant 0 : i32
      %cst1_i32_10 = spirv.Constant 1 : i32
      %64 = spirv.AccessChain %matmul_relu_f32_arg_2_addr[%cst0_i32_8, %61] : !spirv.ptr<!spirv.struct<(!spirv.array<786432 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      spirv.Store "StorageBuffer" %64, %63 : f32
      %65 = spirv.IAdd %11, %cst3_i32 : i32
      %66 = spirv.IAdd %13, %65 : i32
      %67 = spirv.IAdd %47, %66 : i32
      %68 = spirv.FOrdGreaterThan %42, %cst_f32 : f32
      %69 = spirv.Select %68, %42, %cst_f32 : i1, f32
      %cst0_i32_11 = spirv.Constant 0 : i32
      %cst0_i32_12 = spirv.Constant 0 : i32
      %cst1_i32_13 = spirv.Constant 1 : i32
      %70 = spirv.AccessChain %matmul_relu_f32_arg_2_addr[%cst0_i32_11, %67] : !spirv.ptr<!spirv.struct<(!spirv.array<786432 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      spirv.Store "StorageBuffer" %70, %69 : f32
      %71 = spirv.IAdd %10, %cst1_i32 : i32
      %72 = spirv.IAdd %12, %71 : i32
      %73 = spirv.IMul %72, %cst768_i32 : i32
      %74 = spirv.IAdd %13, %11 : i32
      %75 = spirv.IAdd %73, %74 : i32
      %76 = spirv.FOrdGreaterThan %41, %cst_f32 : f32
      %77 = spirv.Select %76, %41, %cst_f32 : i1, f32
      %cst0_i32_14 = spirv.Constant 0 : i32
      %cst0_i32_15 = spirv.Constant 0 : i32
      %cst1_i32_16 = spirv.Constant 1 : i32
      %78 = spirv.AccessChain %matmul_relu_f32_arg_2_addr[%cst0_i32_14, %75] : !spirv.ptr<!spirv.struct<(!spirv.array<786432 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      spirv.Store "StorageBuffer" %78, %77 : f32
      %79 = spirv.IAdd %11, %cst1_i32 : i32
      %80 = spirv.IAdd %13, %79 : i32
      %81 = spirv.IAdd %73, %80 : i32
      %82 = spirv.FOrdGreaterThan %40, %cst_f32 : f32
      %83 = spirv.Select %82, %40, %cst_f32 : i1, f32
      %cst0_i32_17 = spirv.Constant 0 : i32
      %cst0_i32_18 = spirv.Constant 0 : i32
      %cst1_i32_19 = spirv.Constant 1 : i32
      %84 = spirv.AccessChain %matmul_relu_f32_arg_2_addr[%cst0_i32_17, %81] : !spirv.ptr<!spirv.struct<(!spirv.array<786432 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      spirv.Store "StorageBuffer" %84, %83 : f32
      %85 = spirv.IAdd %11, %cst2_i32 : i32
      %86 = spirv.IAdd %13, %85 : i32
      %87 = spirv.IAdd %73, %86 : i32
      %88 = spirv.FOrdGreaterThan %39, %cst_f32 : f32
      %89 = spirv.Select %88, %39, %cst_f32 : i1, f32
      %cst0_i32_20 = spirv.Constant 0 : i32
      %cst0_i32_21 = spirv.Constant 0 : i32
      %cst1_i32_22 = spirv.Constant 1 : i32
      %90 = spirv.AccessChain %matmul_relu_f32_arg_2_addr[%cst0_i32_20, %87] : !spirv.ptr<!spirv.struct<(!spirv.array<786432 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      spirv.Store "StorageBuffer" %90, %89 : f32
      %91 = spirv.IAdd %11, %cst3_i32 : i32
      %92 = spirv.IAdd %13, %91 : i32
      %93 = spirv.IAdd %73, %92 : i32
      %94 = spirv.FOrdGreaterThan %38, %cst_f32 : f32
      %95 = spirv.Select %94, %38, %cst_f32 : i1, f32
      %cst0_i32_23 = spirv.Constant 0 : i32
      %cst0_i32_24 = spirv.Constant 0 : i32
      %cst1_i32_25 = spirv.Constant 1 : i32
      %96 = spirv.AccessChain %matmul_relu_f32_arg_2_addr[%cst0_i32_23, %93] : !spirv.ptr<!spirv.struct<(!spirv.array<786432 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      spirv.Store "StorageBuffer" %96, %95 : f32
      %97 = spirv.IAdd %10, %cst2_i32 : i32
      %98 = spirv.IAdd %12, %97 : i32
      %99 = spirv.IMul %98, %cst768_i32 : i32
      %100 = spirv.IAdd %13, %11 : i32
      %101 = spirv.IAdd %99, %100 : i32
      %102 = spirv.FOrdGreaterThan %37, %cst_f32 : f32
      %103 = spirv.Select %102, %37, %cst_f32 : i1, f32
      %cst0_i32_26 = spirv.Constant 0 : i32
      %cst0_i32_27 = spirv.Constant 0 : i32
      %cst1_i32_28 = spirv.Constant 1 : i32
      %104 = spirv.AccessChain %matmul_relu_f32_arg_2_addr[%cst0_i32_26, %101] : !spirv.ptr<!spirv.struct<(!spirv.array<786432 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      spirv.Store "StorageBuffer" %104, %103 : f32
      %105 = spirv.IAdd %11, %cst1_i32 : i32
      %106 = spirv.IAdd %13, %105 : i32
      %107 = spirv.IAdd %99, %106 : i32
      %108 = spirv.FOrdGreaterThan %36, %cst_f32 : f32
      %109 = spirv.Select %108, %36, %cst_f32 : i1, f32
      %cst0_i32_29 = spirv.Constant 0 : i32
      %cst0_i32_30 = spirv.Constant 0 : i32
      %cst1_i32_31 = spirv.Constant 1 : i32
      %110 = spirv.AccessChain %matmul_relu_f32_arg_2_addr[%cst0_i32_29, %107] : !spirv.ptr<!spirv.struct<(!spirv.array<786432 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      spirv.Store "StorageBuffer" %110, %109 : f32
      %111 = spirv.IAdd %11, %cst2_i32 : i32
      %112 = spirv.IAdd %13, %111 : i32
      %113 = spirv.IAdd %99, %112 : i32
      %114 = spirv.FOrdGreaterThan %35, %cst_f32 : f32
      %115 = spirv.Select %114, %35, %cst_f32 : i1, f32
      %cst0_i32_32 = spirv.Constant 0 : i32
      %cst0_i32_33 = spirv.Constant 0 : i32
      %cst1_i32_34 = spirv.Constant 1 : i32
      %116 = spirv.AccessChain %matmul_relu_f32_arg_2_addr[%cst0_i32_32, %113] : !spirv.ptr<!spirv.struct<(!spirv.array<786432 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      spirv.Store "StorageBuffer" %116, %115 : f32
      %117 = spirv.IAdd %11, %cst3_i32 : i32
      %118 = spirv.IAdd %13, %117 : i32
      %119 = spirv.IAdd %99, %118 : i32
      %120 = spirv.FOrdGreaterThan %34, %cst_f32 : f32
      %121 = spirv.Select %120, %34, %cst_f32 : i1, f32
      %cst0_i32_35 = spirv.Constant 0 : i32
      %cst0_i32_36 = spirv.Constant 0 : i32
      %cst1_i32_37 = spirv.Constant 1 : i32
      %122 = spirv.AccessChain %matmul_relu_f32_arg_2_addr[%cst0_i32_35, %119] : !spirv.ptr<!spirv.struct<(!spirv.array<786432 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      spirv.Store "StorageBuffer" %122, %121 : f32
      %123 = spirv.IAdd %10, %cst3_i32 : i32
      %124 = spirv.IAdd %12, %123 : i32
      %125 = spirv.IMul %124, %cst768_i32 : i32
      %126 = spirv.IAdd %13, %11 : i32
      %127 = spirv.IAdd %125, %126 : i32
      %128 = spirv.FOrdGreaterThan %33, %cst_f32 : f32
      %129 = spirv.Select %128, %33, %cst_f32 : i1, f32
      %cst0_i32_38 = spirv.Constant 0 : i32
      %cst0_i32_39 = spirv.Constant 0 : i32
      %cst1_i32_40 = spirv.Constant 1 : i32
      %130 = spirv.AccessChain %matmul_relu_f32_arg_2_addr[%cst0_i32_38, %127] : !spirv.ptr<!spirv.struct<(!spirv.array<786432 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      spirv.Store "StorageBuffer" %130, %129 : f32
      %131 = spirv.IAdd %11, %cst1_i32 : i32
      %132 = spirv.IAdd %13, %131 : i32
      %133 = spirv.IAdd %125, %132 : i32
      %134 = spirv.FOrdGreaterThan %32, %cst_f32 : f32
      %135 = spirv.Select %134, %32, %cst_f32 : i1, f32
      %cst0_i32_41 = spirv.Constant 0 : i32
      %cst0_i32_42 = spirv.Constant 0 : i32
      %cst1_i32_43 = spirv.Constant 1 : i32
      %136 = spirv.AccessChain %matmul_relu_f32_arg_2_addr[%cst0_i32_41, %133] : !spirv.ptr<!spirv.struct<(!spirv.array<786432 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      spirv.Store "StorageBuffer" %136, %135 : f32
      %137 = spirv.IAdd %11, %cst2_i32 : i32
      %138 = spirv.IAdd %13, %137 : i32
      %139 = spirv.IAdd %125, %138 : i32
      %140 = spirv.FOrdGreaterThan %31, %cst_f32 : f32
      %141 = spirv.Select %140, %31, %cst_f32 : i1, f32
      %cst0_i32_44 = spirv.Constant 0 : i32
      %cst0_i32_45 = spirv.Constant 0 : i32
      %cst1_i32_46 = spirv.Constant 1 : i32
      %142 = spirv.AccessChain %matmul_relu_f32_arg_2_addr[%cst0_i32_44, %139] : !spirv.ptr<!spirv.struct<(!spirv.array<786432 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      spirv.Store "StorageBuffer" %142, %141 : f32
      %143 = spirv.IAdd %11, %cst3_i32 : i32
      %144 = spirv.IAdd %13, %143 : i32
      %145 = spirv.IAdd %125, %144 : i32
      %146 = spirv.FOrdGreaterThan %30, %cst_f32 : f32
      %147 = spirv.Select %146, %30, %cst_f32 : i1, f32
      %cst0_i32_47 = spirv.Constant 0 : i32
      %cst0_i32_48 = spirv.Constant 0 : i32
      %cst1_i32_49 = spirv.Constant 1 : i32
      %148 = spirv.AccessChain %matmul_relu_f32_arg_2_addr[%cst0_i32_47, %145] : !spirv.ptr<!spirv.struct<(!spirv.array<786432 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      spirv.Store "StorageBuffer" %148, %147 : f32
      spirv.Return
    }
    spirv.EntryPoint "GLCompute" @matmul_relu_f32, @__builtin__WorkgroupId__, @__builtin__LocalInvocationId__
    spirv.ExecutionMode @matmul_relu_f32 "LocalSize", 16, 16, 1
  }
