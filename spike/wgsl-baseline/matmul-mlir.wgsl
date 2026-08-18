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
    var phi_87_: u32;
    var phi_89_: f32;
    var phi_90_: f32;
    var phi_91_: f32;
    var phi_92_: f32;
    var phi_93_: f32;
    var phi_94_: f32;
    var phi_95_: f32;
    var phi_96_: f32;
    var phi_97_: f32;
    var phi_98_: f32;
    var phi_99_: f32;
    var phi_100_: f32;
    var phi_101_: f32;
    var phi_102_: f32;
    var phi_103_: f32;
    var phi_104_: f32;
    var phi_110_: u32;
    var phi_127_: u32;
    var phi_158_: u32;
    var phi_159_: f32;
    var phi_160_: f32;
    var phi_161_: f32;
    var phi_162_: f32;
    var phi_163_: f32;
    var phi_164_: f32;
    var phi_165_: f32;
    var phi_166_: f32;
    var phi_167_: f32;
    var phi_168_: f32;
    var phi_169_: f32;
    var phi_170_: f32;
    var phi_171_: f32;
    var phi_172_: f32;
    var phi_173_: f32;
    var phi_174_: f32;

    let _e55 = _builtin_WorkgroupId_1;
    let _e57 = _builtin_WorkgroupId_1;
    let _e59 = _builtin_LocalInvocationId_1;
    let _e61 = _builtin_LocalInvocationId_1;
    let _e64 = ((_e61.y * 16u) + _e59.x);
    let _e65 = (_e61.y * 4u);
    let _e66 = (_e59.x * 4u);
    let _e67 = (_e57.y * 64u);
    let _e68 = (_e55.x * 64u);
    let _e70 = (_e65 + 1u);
    let _e72 = (_e65 + 2u);
    let _e74 = (_e65 + 3u);
    let _e76 = (_e66 + 1u);
    let _e77 = (_e66 + 2u);
    let _e78 = (_e66 + 3u);
    phi_87_ = 0u;
    phi_89_ = 0f;
    phi_90_ = 0f;
    phi_91_ = 0f;
    phi_92_ = 0f;
    phi_93_ = 0f;
    phi_94_ = 0f;
    phi_95_ = 0f;
    phi_96_ = 0f;
    phi_97_ = 0f;
    phi_98_ = 0f;
    phi_99_ = 0f;
    phi_100_ = 0f;
    phi_101_ = 0f;
    phi_102_ = 0f;
    phi_103_ = 0f;
    phi_104_ = 0f;
    loop {
        let _e80 = phi_87_;
        let _e82 = phi_89_;
        let _e84 = phi_90_;
        let _e86 = phi_91_;
        let _e88 = phi_92_;
        let _e90 = phi_93_;
        let _e92 = phi_94_;
        let _e94 = phi_95_;
        let _e96 = phi_96_;
        let _e98 = phi_97_;
        let _e100 = phi_98_;
        let _e102 = phi_99_;
        let _e104 = phi_100_;
        let _e106 = phi_101_;
        let _e108 = phi_102_;
        let _e110 = phi_103_;
        let _e112 = phi_104_;
        if (bitcast<i32>(_e80) < bitcast<i32>(512u)) {
            continue;
        } else {
            break;
        }
        continuing {
            phi_110_ = _e64;
            loop {
                let _e117 = phi_110_;
                if (bitcast<i32>(_e117) < bitcast<i32>(1024u)) {
                    continue;
                } else {
                    break;
                }
                continuing {
                    let _e129 = matmul_relu_f32_arg_0_.member[(((_e67 + (_e117 / 16u)) * 512u) + (_e80 + (_e117 % 16u)))];
                    _workgroup_mem_0_.member[_e117] = _e129;
                    phi_110_ = (_e117 + 256u);
                }
            }
            phi_127_ = _e64;
            loop {
                let _e134 = phi_127_;
                if (bitcast<i32>(_e134) < bitcast<i32>(1024u)) {
                    continue;
                } else {
                    break;
                }
                continuing {
                    let _e146 = matmul_relu_f32_arg_1_.member[(((_e80 + (_e134 / 64u)) * 768u) + (_e68 + (_e134 % 64u)))];
                    _workgroup_mem_1_.member[_e134] = _e146;
                    phi_127_ = (_e134 + 256u);
                }
            }
            workgroupBarrier();
            phi_158_ = 0u;
            phi_159_ = _e82;
            phi_160_ = _e84;
            phi_161_ = _e86;
            phi_162_ = _e88;
            phi_163_ = _e90;
            phi_164_ = _e92;
            phi_165_ = _e94;
            phi_166_ = _e96;
            phi_167_ = _e98;
            phi_168_ = _e100;
            phi_169_ = _e102;
            phi_170_ = _e104;
            phi_171_ = _e106;
            phi_172_ = _e108;
            phi_173_ = _e110;
            phi_174_ = _e112;
            loop {
                let _e151 = phi_158_;
                let _e153 = phi_159_;
                let _e155 = phi_160_;
                let _e157 = phi_161_;
                let _e159 = phi_162_;
                let _e161 = phi_163_;
                let _e163 = phi_164_;
                let _e165 = phi_165_;
                let _e167 = phi_166_;
                let _e169 = phi_167_;
                let _e171 = phi_168_;
                let _e173 = phi_169_;
                let _e175 = phi_170_;
                let _e177 = phi_171_;
                let _e179 = phi_172_;
                let _e181 = phi_173_;
                let _e183 = phi_174_;
                if (bitcast<i32>(_e151) < bitcast<i32>(16u)) {
                    continue;
                } else {
                    break;
                }
                continuing {
                    let _e190 = _workgroup_mem_0_.member[((_e61.y * 64u) + _e151)];
                    let _e194 = _workgroup_mem_0_.member[((_e70 * 16u) + _e151)];
                    let _e198 = _workgroup_mem_0_.member[((_e72 * 16u) + _e151)];
                    let _e202 = _workgroup_mem_0_.member[((_e74 * 16u) + _e151)];
                    let _e203 = (_e151 * 64u);
                    let _e207 = _workgroup_mem_1_.member[(_e203 + _e66)];
                    let _e211 = _workgroup_mem_1_.member[(_e203 + _e76)];
                    let _e215 = _workgroup_mem_1_.member[(_e203 + _e77)];
                    let _e219 = _workgroup_mem_1_.member[(_e203 + _e78)];
                    let _e221 = (_e153 + (_e190 * _e207));
                    let _e223 = (_e155 + (_e190 * _e211));
                    let _e225 = (_e157 + (_e190 * _e215));
                    let _e227 = (_e159 + (_e190 * _e219));
                    let _e229 = (_e161 + (_e194 * _e207));
                    let _e231 = (_e163 + (_e194 * _e211));
                    let _e233 = (_e165 + (_e194 * _e215));
                    let _e235 = (_e167 + (_e194 * _e219));
                    let _e237 = (_e169 + (_e198 * _e207));
                    let _e239 = (_e171 + (_e198 * _e211));
                    let _e241 = (_e173 + (_e198 * _e215));
                    let _e243 = (_e175 + (_e198 * _e219));
                    let _e245 = (_e177 + (_e202 * _e207));
                    let _e247 = (_e179 + (_e202 * _e211));
                    let _e249 = (_e181 + (_e202 * _e215));
                    let _e251 = (_e183 + (_e202 * _e219));
                    local_16 = _e221;
                    local_17 = _e223;
                    local_18 = _e225;
                    local_19 = _e227;
                    local_20 = _e229;
                    local_21 = _e231;
                    local_22 = _e233;
                    local_23 = _e235;
                    local_24 = _e237;
                    local_25 = _e239;
                    local_26 = _e241;
                    local_27 = _e243;
                    local_28 = _e245;
                    local_29 = _e247;
                    local_30 = _e249;
                    local_31 = _e251;
                    phi_158_ = (_e151 + 1u);
                    phi_159_ = _e221;
                    phi_160_ = _e223;
                    phi_161_ = _e225;
                    phi_162_ = _e227;
                    phi_163_ = _e229;
                    phi_164_ = _e231;
                    phi_165_ = _e233;
                    phi_166_ = _e235;
                    phi_167_ = _e237;
                    phi_168_ = _e239;
                    phi_169_ = _e241;
                    phi_170_ = _e243;
                    phi_171_ = _e245;
                    phi_172_ = _e247;
                    phi_173_ = _e249;
                    phi_174_ = _e251;
                }
            }
            let _e253 = local_31;
            let _e254 = local_30;
            let _e255 = local_29;
            let _e256 = local_28;
            let _e257 = local_27;
            let _e258 = local_26;
            let _e259 = local_25;
            let _e260 = local_24;
            let _e261 = local_23;
            let _e262 = local_22;
            let _e263 = local_21;
            let _e264 = local_20;
            let _e265 = local_19;
            let _e266 = local_18;
            let _e267 = local_17;
            let _e268 = local_16;
            workgroupBarrier();
            local = _e268;
            local_1 = _e267;
            local_2 = _e266;
            local_3 = _e265;
            local_4 = _e264;
            local_5 = _e263;
            local_6 = _e262;
            local_7 = _e261;
            local_8 = _e260;
            local_9 = _e259;
            local_10 = _e258;
            local_11 = _e257;
            local_12 = _e256;
            local_13 = _e255;
            local_14 = _e254;
            local_15 = _e253;
            phi_87_ = (_e80 + 16u);
            phi_89_ = _e268;
            phi_90_ = _e267;
            phi_91_ = _e266;
            phi_92_ = _e265;
            phi_93_ = _e264;
            phi_94_ = _e263;
            phi_95_ = _e262;
            phi_96_ = _e261;
            phi_97_ = _e260;
            phi_98_ = _e259;
            phi_99_ = _e258;
            phi_100_ = _e257;
            phi_101_ = _e256;
            phi_102_ = _e255;
            phi_103_ = _e254;
            phi_104_ = _e253;
        }
    }
    let _e270 = local_15;
    let _e271 = local_14;
    let _e272 = local_13;
    let _e273 = local_12;
    let _e274 = local_11;
    let _e275 = local_10;
    let _e276 = local_9;
    let _e277 = local_8;
    let _e278 = local_7;
    let _e279 = local_6;
    let _e280 = local_5;
    let _e281 = local_4;
    let _e282 = local_3;
    let _e283 = local_2;
    let _e284 = local_1;
    let _e285 = local;
    let _e287 = ((_e67 + _e65) * 768u);
    let _e288 = (_e68 + _e66);
    matmul_relu_f32_arg_2_.member[(_e287 + _e288)] = select(0f, _e285, (_e285 > 0f));
    let _e294 = (_e68 + _e76);
    matmul_relu_f32_arg_2_.member[(_e287 + _e294)] = select(0f, _e284, (_e284 > 0f));
    let _e300 = (_e68 + _e77);
    matmul_relu_f32_arg_2_.member[(_e287 + _e300)] = select(0f, _e283, (_e283 > 0f));
    let _e306 = (_e68 + _e78);
    matmul_relu_f32_arg_2_.member[(_e287 + _e306)] = select(0f, _e282, (_e282 > 0f));
    let _e313 = ((_e67 + _e70) * 768u);
    matmul_relu_f32_arg_2_.member[(_e313 + _e288)] = select(0f, _e281, (_e281 > 0f));
    matmul_relu_f32_arg_2_.member[(_e313 + _e294)] = select(0f, _e280, (_e280 > 0f));
    matmul_relu_f32_arg_2_.member[(_e313 + _e300)] = select(0f, _e279, (_e279 > 0f));
    matmul_relu_f32_arg_2_.member[(_e313 + _e306)] = select(0f, _e278, (_e278 > 0f));
    let _e335 = ((_e67 + _e72) * 768u);
    matmul_relu_f32_arg_2_.member[(_e335 + _e288)] = select(0f, _e277, (_e277 > 0f));
    matmul_relu_f32_arg_2_.member[(_e335 + _e294)] = select(0f, _e276, (_e276 > 0f));
    matmul_relu_f32_arg_2_.member[(_e335 + _e300)] = select(0f, _e275, (_e275 > 0f));
    matmul_relu_f32_arg_2_.member[(_e335 + _e306)] = select(0f, _e274, (_e274 > 0f));
    let _e357 = ((_e67 + _e74) * 768u);
    matmul_relu_f32_arg_2_.member[(_e357 + _e288)] = select(0f, _e273, (_e273 > 0f));
    matmul_relu_f32_arg_2_.member[(_e357 + _e294)] = select(0f, _e272, (_e272 > 0f));
    matmul_relu_f32_arg_2_.member[(_e357 + _e300)] = select(0f, _e271, (_e271 > 0f));
    matmul_relu_f32_arg_2_.member[(_e357 + _e306)] = select(0f, _e270, (_e270 > 0f));
    return;
}

@compute @workgroup_size(16, 16, 1) 
fn matmul_relu_f32_(@builtin(workgroup_id) _builtin_WorkgroupId: vec3<u32>, @builtin(local_invocation_id) _builtin_LocalInvocationId: vec3<u32>) {
    _builtin_WorkgroupId_1 = _builtin_WorkgroupId;
    _builtin_LocalInvocationId_1 = _builtin_LocalInvocationId;
    matmul_relu_f32_1();
}
