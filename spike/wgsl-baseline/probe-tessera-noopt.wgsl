struct type_5 {
    member: array<f32, 1024>,
}

struct type_8 {
    member: array<f32, 524288>,
}

struct type_11 {
    member: array<f32, 393216>,
}

struct type_14 {
    member: array<f32, 786432>,
}

var<private> _builtin_LocalInvocationId_1: vec3<u32>;
var<private> _builtin_WorkgroupId_1: vec3<u32>;
var<workgroup> _workgroup_mem_1_: type_5;
var<workgroup> _workgroup_mem_0_: type_5;
@group(0) @binding(0) 
var<storage, read_write> matmul_relu_f32_arg_0_: type_8;
@group(0) @binding(1) 
var<storage, read_write> matmul_relu_f32_arg_1_: type_11;
@group(0) @binding(2) 
var<storage, read_write> matmul_relu_f32_arg_2_: type_14;

fn matmul_relu_f32_1() {
    var local: f32;
    var local_1: f32;
    var local_2: f32;
    var local_3: f32;
    var local_4: f32;
    var local_5: f32;
    var local_6: f32;
    var local_7: f32;
    var local_8: f32;
    var local_9: f32;
    var local_10: f32;
    var local_11: f32;
    var local_12: f32;
    var local_13: f32;
    var local_14: f32;
    var local_15: f32;
    var local_16: f32;
    var local_17: f32;
    var local_18: f32;
    var local_19: f32;
    var local_20: f32;
    var local_21: f32;
    var local_22: f32;
    var local_23: f32;
    var local_24: f32;
    var local_25: f32;
    var local_26: f32;
    var local_27: f32;
    var local_28: f32;
    var local_29: f32;
    var local_30: f32;
    var local_31: f32;
    var phi_77_: u32;
    var phi_79_: f32;
    var phi_80_: f32;
    var phi_81_: f32;
    var phi_82_: f32;
    var phi_83_: f32;
    var phi_84_: f32;
    var phi_85_: f32;
    var phi_86_: f32;
    var phi_87_: f32;
    var phi_88_: f32;
    var phi_89_: f32;
    var phi_90_: f32;
    var phi_91_: f32;
    var phi_92_: f32;
    var phi_93_: f32;
    var phi_94_: f32;
    var phi_100_: u32;
    var phi_117_: u32;
    var phi_148_: u32;
    var phi_149_: f32;
    var phi_150_: f32;
    var phi_151_: f32;
    var phi_152_: f32;
    var phi_153_: f32;
    var phi_154_: f32;
    var phi_155_: f32;
    var phi_156_: f32;
    var phi_157_: f32;
    var phi_158_: f32;
    var phi_159_: f32;
    var phi_160_: f32;
    var phi_161_: f32;
    var phi_162_: f32;
    var phi_163_: f32;
    var phi_164_: f32;

    let _e55 = _builtin_WorkgroupId_1;
    let _e57 = _builtin_WorkgroupId_1;
    let _e59 = _builtin_LocalInvocationId_1;
    let _e61 = _builtin_LocalInvocationId_1;
    let _e64 = ((_e61.y * 16u) + _e59.x);
    let _e65 = (_e61.y * 4u);
    let _e66 = (_e59.x * 4u);
    let _e67 = (_e57.y * 64u);
    let _e68 = (_e55.x * 64u);
    phi_77_ = 0u;
    phi_79_ = 0f;
    phi_80_ = 0f;
    phi_81_ = 0f;
    phi_82_ = 0f;
    phi_83_ = 0f;
    phi_84_ = 0f;
    phi_85_ = 0f;
    phi_86_ = 0f;
    phi_87_ = 0f;
    phi_88_ = 0f;
    phi_89_ = 0f;
    phi_90_ = 0f;
    phi_91_ = 0f;
    phi_92_ = 0f;
    phi_93_ = 0f;
    phi_94_ = 0f;
    loop {
        let _e70 = phi_77_;
        let _e72 = phi_79_;
        let _e74 = phi_80_;
        let _e76 = phi_81_;
        let _e78 = phi_82_;
        let _e80 = phi_83_;
        let _e82 = phi_84_;
        let _e84 = phi_85_;
        let _e86 = phi_86_;
        let _e88 = phi_87_;
        let _e90 = phi_88_;
        let _e92 = phi_89_;
        let _e94 = phi_90_;
        let _e96 = phi_91_;
        let _e98 = phi_92_;
        let _e100 = phi_93_;
        let _e102 = phi_94_;
        if (bitcast<i32>(_e70) < bitcast<i32>(512u)) {
            continue;
        } else {
            break;
        }
        continuing {
            phi_100_ = _e64;
            loop {
                let _e107 = phi_100_;
                if (bitcast<i32>(_e107) < bitcast<i32>(1024u)) {
                    continue;
                } else {
                    break;
                }
                continuing {
                    let _e119 = matmul_relu_f32_arg_0_.member[(((_e67 + (_e107 / 16u)) * 512u) + (_e70 + (_e107 % 16u)))];
                    _workgroup_mem_0_.member[_e107] = _e119;
                    phi_100_ = (_e107 + 256u);
                }
            }
            phi_117_ = _e64;
            loop {
                let _e124 = phi_117_;
                if (bitcast<i32>(_e124) < bitcast<i32>(1024u)) {
                    continue;
                } else {
                    break;
                }
                continuing {
                    let _e136 = matmul_relu_f32_arg_1_.member[(((_e70 + (_e124 / 64u)) * 768u) + (_e68 + (_e124 % 64u)))];
                    _workgroup_mem_1_.member[_e124] = _e136;
                    phi_117_ = (_e124 + 256u);
                }
            }
            workgroupBarrier();
            phi_148_ = 0u;
            phi_149_ = _e72;
            phi_150_ = _e74;
            phi_151_ = _e76;
            phi_152_ = _e78;
            phi_153_ = _e80;
            phi_154_ = _e82;
            phi_155_ = _e84;
            phi_156_ = _e86;
            phi_157_ = _e88;
            phi_158_ = _e90;
            phi_159_ = _e92;
            phi_160_ = _e94;
            phi_161_ = _e96;
            phi_162_ = _e98;
            phi_163_ = _e100;
            phi_164_ = _e102;
            loop {
                let _e141 = phi_148_;
                let _e143 = phi_149_;
                let _e145 = phi_150_;
                let _e147 = phi_151_;
                let _e149 = phi_152_;
                let _e151 = phi_153_;
                let _e153 = phi_154_;
                let _e155 = phi_155_;
                let _e157 = phi_156_;
                let _e159 = phi_157_;
                let _e161 = phi_158_;
                let _e163 = phi_159_;
                let _e165 = phi_160_;
                let _e167 = phi_161_;
                let _e169 = phi_162_;
                let _e171 = phi_163_;
                let _e173 = phi_164_;
                if (bitcast<i32>(_e141) < bitcast<i32>(16u)) {
                    continue;
                } else {
                    break;
                }
                continuing {
                    let _e181 = _workgroup_mem_0_.member[((_e65 * 16u) + _e141)];
                    let _e187 = _workgroup_mem_0_.member[(((_e65 + 1u) * 16u) + _e141)];
                    let _e193 = _workgroup_mem_0_.member[(((_e65 + 2u) * 16u) + _e141)];
                    let _e199 = _workgroup_mem_0_.member[(((_e65 + 3u) * 16u) + _e141)];
                    let _e204 = _workgroup_mem_1_.member[((_e141 * 64u) + _e66)];
                    let _e210 = _workgroup_mem_1_.member[((_e141 * 64u) + (_e66 + 1u))];
                    let _e216 = _workgroup_mem_1_.member[((_e141 * 64u) + (_e66 + 2u))];
                    let _e222 = _workgroup_mem_1_.member[((_e141 * 64u) + (_e66 + 3u))];
                    let _e224 = (_e143 + (_e181 * _e204));
                    let _e226 = (_e145 + (_e181 * _e210));
                    let _e228 = (_e147 + (_e181 * _e216));
                    let _e230 = (_e149 + (_e181 * _e222));
                    let _e232 = (_e151 + (_e187 * _e204));
                    let _e234 = (_e153 + (_e187 * _e210));
                    let _e236 = (_e155 + (_e187 * _e216));
                    let _e238 = (_e157 + (_e187 * _e222));
                    let _e240 = (_e159 + (_e193 * _e204));
                    let _e242 = (_e161 + (_e193 * _e210));
                    let _e244 = (_e163 + (_e193 * _e216));
                    let _e246 = (_e165 + (_e193 * _e222));
                    let _e248 = (_e167 + (_e199 * _e204));
                    let _e250 = (_e169 + (_e199 * _e210));
                    let _e252 = (_e171 + (_e199 * _e216));
                    let _e254 = (_e173 + (_e199 * _e222));
                    local_16 = _e224;
                    local_17 = _e226;
                    local_18 = _e228;
                    local_19 = _e230;
                    local_20 = _e232;
                    local_21 = _e234;
                    local_22 = _e236;
                    local_23 = _e238;
                    local_24 = _e240;
                    local_25 = _e242;
                    local_26 = _e244;
                    local_27 = _e246;
                    local_28 = _e248;
                    local_29 = _e250;
                    local_30 = _e252;
                    local_31 = _e254;
                    phi_148_ = (_e141 + 1u);
                    phi_149_ = _e224;
                    phi_150_ = _e226;
                    phi_151_ = _e228;
                    phi_152_ = _e230;
                    phi_153_ = _e232;
                    phi_154_ = _e234;
                    phi_155_ = _e236;
                    phi_156_ = _e238;
                    phi_157_ = _e240;
                    phi_158_ = _e242;
                    phi_159_ = _e244;
                    phi_160_ = _e246;
                    phi_161_ = _e248;
                    phi_162_ = _e250;
                    phi_163_ = _e252;
                    phi_164_ = _e254;
                }
            }
            let _e256 = local_31;
            let _e257 = local_30;
            let _e258 = local_29;
            let _e259 = local_28;
            let _e260 = local_27;
            let _e261 = local_26;
            let _e262 = local_25;
            let _e263 = local_24;
            let _e264 = local_23;
            let _e265 = local_22;
            let _e266 = local_21;
            let _e267 = local_20;
            let _e268 = local_19;
            let _e269 = local_18;
            let _e270 = local_17;
            let _e271 = local_16;
            workgroupBarrier();
            local = _e271;
            local_1 = _e270;
            local_2 = _e269;
            local_3 = _e268;
            local_4 = _e267;
            local_5 = _e266;
            local_6 = _e265;
            local_7 = _e264;
            local_8 = _e263;
            local_9 = _e262;
            local_10 = _e261;
            local_11 = _e260;
            local_12 = _e259;
            local_13 = _e258;
            local_14 = _e257;
            local_15 = _e256;
            phi_77_ = (_e70 + 16u);
            phi_79_ = _e271;
            phi_80_ = _e270;
            phi_81_ = _e269;
            phi_82_ = _e268;
            phi_83_ = _e267;
            phi_84_ = _e266;
            phi_85_ = _e265;
            phi_86_ = _e264;
            phi_87_ = _e263;
            phi_88_ = _e262;
            phi_89_ = _e261;
            phi_90_ = _e260;
            phi_91_ = _e259;
            phi_92_ = _e258;
            phi_93_ = _e257;
            phi_94_ = _e256;
        }
    }
    let _e273 = local_15;
    let _e274 = local_14;
    let _e275 = local_13;
    let _e276 = local_12;
    let _e277 = local_11;
    let _e278 = local_10;
    let _e279 = local_9;
    let _e280 = local_8;
    let _e281 = local_7;
    let _e282 = local_6;
    let _e283 = local_5;
    let _e284 = local_4;
    let _e285 = local_3;
    let _e286 = local_2;
    let _e287 = local_1;
    let _e288 = local;
    let _e290 = ((_e67 + _e65) * 768u);
    matmul_relu_f32_arg_2_.member[(_e290 + (_e68 + _e66))] = select(0f, _e288, (_e288 > 0f));
    matmul_relu_f32_arg_2_.member[(_e290 + (_e68 + (_e66 + 1u)))] = select(0f, _e287, (_e287 > 0f));
    matmul_relu_f32_arg_2_.member[(_e290 + (_e68 + (_e66 + 2u)))] = select(0f, _e286, (_e286 > 0f));
    matmul_relu_f32_arg_2_.member[(_e290 + (_e68 + (_e66 + 3u)))] = select(0f, _e285, (_e285 > 0f));
    let _e320 = ((_e67 + (_e65 + 1u)) * 768u);
    matmul_relu_f32_arg_2_.member[(_e320 + (_e68 + _e66))] = select(0f, _e284, (_e284 > 0f));
    matmul_relu_f32_arg_2_.member[(_e320 + (_e68 + (_e66 + 1u)))] = select(0f, _e283, (_e283 > 0f));
    matmul_relu_f32_arg_2_.member[(_e320 + (_e68 + (_e66 + 2u)))] = select(0f, _e282, (_e282 > 0f));
    matmul_relu_f32_arg_2_.member[(_e320 + (_e68 + (_e66 + 3u)))] = select(0f, _e281, (_e281 > 0f));
    let _e350 = ((_e67 + (_e65 + 2u)) * 768u);
    matmul_relu_f32_arg_2_.member[(_e350 + (_e68 + _e66))] = select(0f, _e280, (_e280 > 0f));
    matmul_relu_f32_arg_2_.member[(_e350 + (_e68 + (_e66 + 1u)))] = select(0f, _e279, (_e279 > 0f));
    matmul_relu_f32_arg_2_.member[(_e350 + (_e68 + (_e66 + 2u)))] = select(0f, _e278, (_e278 > 0f));
    matmul_relu_f32_arg_2_.member[(_e350 + (_e68 + (_e66 + 3u)))] = select(0f, _e277, (_e277 > 0f));
    let _e380 = ((_e67 + (_e65 + 3u)) * 768u);
    matmul_relu_f32_arg_2_.member[(_e380 + (_e68 + _e66))] = select(0f, _e276, (_e276 > 0f));
    matmul_relu_f32_arg_2_.member[(_e380 + (_e68 + (_e66 + 1u)))] = select(0f, _e275, (_e275 > 0f));
    matmul_relu_f32_arg_2_.member[(_e380 + (_e68 + (_e66 + 2u)))] = select(0f, _e274, (_e274 > 0f));
    matmul_relu_f32_arg_2_.member[(_e380 + (_e68 + (_e66 + 3u)))] = select(0f, _e273, (_e273 > 0f));
    return;
}

@compute @workgroup_size(16, 16, 1) 
fn matmul_relu_f32_(@builtin(workgroup_id) _builtin_WorkgroupId: vec3<u32>, @builtin(local_invocation_id) _builtin_LocalInvocationId: vec3<u32>) {
    _builtin_WorkgroupId_1 = _builtin_WorkgroupId;
    _builtin_LocalInvocationId_1 = _builtin_LocalInvocationId;
    matmul_relu_f32_1();
}
