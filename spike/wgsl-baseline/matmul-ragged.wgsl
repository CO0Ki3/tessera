struct type_5 {
    member: array<f32, 1024>,
}

struct type_8 {
    member: array<f32, 500000>,
}

struct type_11 {
    member: array<f32, 375000>,
}

struct type_14 {
    member: array<f32, 750000>,
}

var<private> _builtin_LocalInvocationId_1: vec3<u32>;
var<private> _builtin_WorkgroupId_1: vec3<u32>;
var<workgroup> _workgroup_mem_1_: type_5;
var<workgroup> _workgroup_mem_0_: type_5;
@group(0) @binding(0) 
var<storage, read_write> matmul_relu_ragged_arg_0_: type_8;
@group(0) @binding(1) 
var<storage, read_write> matmul_relu_ragged_arg_1_: type_11;
@group(0) @binding(2) 
var<storage, read_write> matmul_relu_ragged_arg_2_: type_14;

fn matmul_relu_ragged_1() {
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
    var phi_90_: u32;
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
    var phi_105_: f32;
    var phi_106_: f32;
    var phi_107_: f32;
    var phi_113_: u32;
    var phi_136_: u32;
    var phi_172_: u32;
    var phi_173_: f32;
    var phi_174_: f32;
    var phi_175_: f32;
    var phi_176_: f32;
    var phi_177_: f32;
    var phi_178_: f32;
    var phi_179_: f32;
    var phi_180_: f32;
    var phi_181_: f32;
    var phi_182_: f32;
    var phi_183_: f32;
    var phi_184_: f32;
    var phi_185_: f32;
    var phi_186_: f32;
    var phi_187_: f32;
    var phi_188_: f32;

    let _e58 = _builtin_WorkgroupId_1;
    let _e60 = _builtin_WorkgroupId_1;
    let _e62 = _builtin_LocalInvocationId_1;
    let _e64 = _builtin_LocalInvocationId_1;
    let _e67 = ((_e64.y * 16u) + _e62.x);
    let _e68 = (_e64.y * 4u);
    let _e69 = (_e62.x * 4u);
    let _e70 = (_e60.y * 64u);
    let _e71 = (_e58.x * 64u);
    let _e73 = (_e68 + 1u);
    let _e75 = (_e68 + 2u);
    let _e77 = (_e68 + 3u);
    let _e79 = (_e69 + 1u);
    let _e80 = (_e69 + 2u);
    let _e81 = (_e69 + 3u);
    phi_90_ = 0u;
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
    phi_105_ = 0f;
    phi_106_ = 0f;
    phi_107_ = 0f;
    loop {
        let _e83 = phi_90_;
        let _e85 = phi_92_;
        let _e87 = phi_93_;
        let _e89 = phi_94_;
        let _e91 = phi_95_;
        let _e93 = phi_96_;
        let _e95 = phi_97_;
        let _e97 = phi_98_;
        let _e99 = phi_99_;
        let _e101 = phi_100_;
        let _e103 = phi_101_;
        let _e105 = phi_102_;
        let _e107 = phi_103_;
        let _e109 = phi_104_;
        let _e111 = phi_105_;
        let _e113 = phi_106_;
        let _e115 = phi_107_;
        if (bitcast<i32>(_e83) < bitcast<i32>(500u)) {
            continue;
        } else {
            break;
        }
        continuing {
            phi_113_ = _e67;
            loop {
                let _e120 = phi_113_;
                if (bitcast<i32>(_e120) < bitcast<i32>(1024u)) {
                    continue;
                } else {
                    break;
                }
                continuing {
                    let _e126 = (_e70 + (_e120 / 16u));
                    let _e128 = (_e83 + (_e120 % 16u));
                    let _e136 = matmul_relu_ragged_arg_0_.member[min(((_e126 * 500u) + _e128), 499999u)];
                    _workgroup_mem_0_.member[_e120] = select(0f, _e136, ((_e126 < 1000u) && (_e128 < 500u)));
                    phi_113_ = (_e120 + 256u);
                }
            }
            phi_136_ = _e67;
            loop {
                let _e142 = phi_136_;
                if (bitcast<i32>(_e142) < bitcast<i32>(1024u)) {
                    continue;
                } else {
                    break;
                }
                continuing {
                    let _e148 = (_e83 + (_e142 / 64u));
                    let _e150 = (_e71 + (_e142 % 64u));
                    let _e158 = matmul_relu_ragged_arg_1_.member[min(((_e148 * 750u) + _e150), 374999u)];
                    _workgroup_mem_1_.member[_e142] = select(0f, _e158, ((_e148 < 500u) && (_e150 < 750u)));
                    phi_136_ = (_e142 + 256u);
                }
            }
            workgroupBarrier();
            phi_172_ = 0u;
            phi_173_ = _e85;
            phi_174_ = _e87;
            phi_175_ = _e89;
            phi_176_ = _e91;
            phi_177_ = _e93;
            phi_178_ = _e95;
            phi_179_ = _e97;
            phi_180_ = _e99;
            phi_181_ = _e101;
            phi_182_ = _e103;
            phi_183_ = _e105;
            phi_184_ = _e107;
            phi_185_ = _e109;
            phi_186_ = _e111;
            phi_187_ = _e113;
            phi_188_ = _e115;
            loop {
                let _e164 = phi_172_;
                let _e166 = phi_173_;
                let _e168 = phi_174_;
                let _e170 = phi_175_;
                let _e172 = phi_176_;
                let _e174 = phi_177_;
                let _e176 = phi_178_;
                let _e178 = phi_179_;
                let _e180 = phi_180_;
                let _e182 = phi_181_;
                let _e184 = phi_182_;
                let _e186 = phi_183_;
                let _e188 = phi_184_;
                let _e190 = phi_185_;
                let _e192 = phi_186_;
                let _e194 = phi_187_;
                let _e196 = phi_188_;
                if (bitcast<i32>(_e164) < bitcast<i32>(16u)) {
                    continue;
                } else {
                    break;
                }
                continuing {
                    let _e203 = _workgroup_mem_0_.member[((_e64.y * 64u) + _e164)];
                    let _e207 = _workgroup_mem_0_.member[((_e73 * 16u) + _e164)];
                    let _e211 = _workgroup_mem_0_.member[((_e75 * 16u) + _e164)];
                    let _e215 = _workgroup_mem_0_.member[((_e77 * 16u) + _e164)];
                    let _e216 = (_e164 * 64u);
                    let _e220 = _workgroup_mem_1_.member[(_e216 + _e69)];
                    let _e224 = _workgroup_mem_1_.member[(_e216 + _e79)];
                    let _e228 = _workgroup_mem_1_.member[(_e216 + _e80)];
                    let _e232 = _workgroup_mem_1_.member[(_e216 + _e81)];
                    let _e234 = (_e166 + (_e203 * _e220));
                    let _e236 = (_e168 + (_e203 * _e224));
                    let _e238 = (_e170 + (_e203 * _e228));
                    let _e240 = (_e172 + (_e203 * _e232));
                    let _e242 = (_e174 + (_e207 * _e220));
                    let _e244 = (_e176 + (_e207 * _e224));
                    let _e246 = (_e178 + (_e207 * _e228));
                    let _e248 = (_e180 + (_e207 * _e232));
                    let _e250 = (_e182 + (_e211 * _e220));
                    let _e252 = (_e184 + (_e211 * _e224));
                    let _e254 = (_e186 + (_e211 * _e228));
                    let _e256 = (_e188 + (_e211 * _e232));
                    let _e258 = (_e190 + (_e215 * _e220));
                    let _e260 = (_e192 + (_e215 * _e224));
                    let _e262 = (_e194 + (_e215 * _e228));
                    let _e264 = (_e196 + (_e215 * _e232));
                    local_16 = _e234;
                    local_17 = _e236;
                    local_18 = _e238;
                    local_19 = _e240;
                    local_20 = _e242;
                    local_21 = _e244;
                    local_22 = _e246;
                    local_23 = _e248;
                    local_24 = _e250;
                    local_25 = _e252;
                    local_26 = _e254;
                    local_27 = _e256;
                    local_28 = _e258;
                    local_29 = _e260;
                    local_30 = _e262;
                    local_31 = _e264;
                    phi_172_ = (_e164 + 1u);
                    phi_173_ = _e234;
                    phi_174_ = _e236;
                    phi_175_ = _e238;
                    phi_176_ = _e240;
                    phi_177_ = _e242;
                    phi_178_ = _e244;
                    phi_179_ = _e246;
                    phi_180_ = _e248;
                    phi_181_ = _e250;
                    phi_182_ = _e252;
                    phi_183_ = _e254;
                    phi_184_ = _e256;
                    phi_185_ = _e258;
                    phi_186_ = _e260;
                    phi_187_ = _e262;
                    phi_188_ = _e264;
                }
            }
            let _e266 = local_31;
            let _e267 = local_30;
            let _e268 = local_29;
            let _e269 = local_28;
            let _e270 = local_27;
            let _e271 = local_26;
            let _e272 = local_25;
            let _e273 = local_24;
            let _e274 = local_23;
            let _e275 = local_22;
            let _e276 = local_21;
            let _e277 = local_20;
            let _e278 = local_19;
            let _e279 = local_18;
            let _e280 = local_17;
            let _e281 = local_16;
            workgroupBarrier();
            local = _e281;
            local_1 = _e280;
            local_2 = _e279;
            local_3 = _e278;
            local_4 = _e277;
            local_5 = _e276;
            local_6 = _e275;
            local_7 = _e274;
            local_8 = _e273;
            local_9 = _e272;
            local_10 = _e271;
            local_11 = _e270;
            local_12 = _e269;
            local_13 = _e268;
            local_14 = _e267;
            local_15 = _e266;
            phi_90_ = (_e83 + 16u);
            phi_92_ = _e281;
            phi_93_ = _e280;
            phi_94_ = _e279;
            phi_95_ = _e278;
            phi_96_ = _e277;
            phi_97_ = _e276;
            phi_98_ = _e275;
            phi_99_ = _e274;
            phi_100_ = _e273;
            phi_101_ = _e272;
            phi_102_ = _e271;
            phi_103_ = _e270;
            phi_104_ = _e269;
            phi_105_ = _e268;
            phi_106_ = _e267;
            phi_107_ = _e266;
        }
    }
    let _e283 = local_15;
    let _e284 = local_14;
    let _e285 = local_13;
    let _e286 = local_12;
    let _e287 = local_11;
    let _e288 = local_10;
    let _e289 = local_9;
    let _e290 = local_8;
    let _e291 = local_7;
    let _e292 = local_6;
    let _e293 = local_5;
    let _e294 = local_4;
    let _e295 = local_3;
    let _e296 = local_2;
    let _e297 = local_1;
    let _e298 = local;
    let _e299 = (_e70 + _e68);
    let _e300 = (_e299 * 750u);
    let _e301 = (_e71 + _e69);
    let _e305 = (_e299 < 1000u);
    let _e306 = (_e301 < 750u);
    if (_e305 && _e306) {
        matmul_relu_ragged_arg_2_.member[(_e300 + _e301)] = select(0f, _e298, (_e298 > 0f));
    }
    let _e310 = (_e71 + _e79);
    let _e314 = (_e310 < 750u);
    if (_e305 && _e314) {
        matmul_relu_ragged_arg_2_.member[(_e300 + _e310)] = select(0f, _e297, (_e297 > 0f));
    }
    let _e318 = (_e71 + _e80);
    let _e322 = (_e318 < 750u);
    if (_e305 && _e322) {
        matmul_relu_ragged_arg_2_.member[(_e300 + _e318)] = select(0f, _e296, (_e296 > 0f));
    }
    let _e326 = (_e71 + _e81);
    let _e330 = (_e326 < 750u);
    if (_e305 && _e330) {
        matmul_relu_ragged_arg_2_.member[(_e300 + _e326)] = select(0f, _e295, (_e295 > 0f));
    }
    let _e334 = (_e70 + _e73);
    let _e335 = (_e334 * 750u);
    let _e339 = (_e334 < 1000u);
    if (_e339 && _e306) {
        matmul_relu_ragged_arg_2_.member[(_e335 + _e301)] = select(0f, _e294, (_e294 > 0f));
    }
    if (_e339 && _e314) {
        matmul_relu_ragged_arg_2_.member[(_e335 + _e310)] = select(0f, _e293, (_e293 > 0f));
    }
    if (_e339 && _e322) {
        matmul_relu_ragged_arg_2_.member[(_e335 + _e318)] = select(0f, _e292, (_e292 > 0f));
    }
    if (_e339 && _e330) {
        matmul_relu_ragged_arg_2_.member[(_e335 + _e326)] = select(0f, _e291, (_e291 > 0f));
    }
    let _e361 = (_e70 + _e75);
    let _e362 = (_e361 * 750u);
    let _e366 = (_e361 < 1000u);
    if (_e366 && _e306) {
        matmul_relu_ragged_arg_2_.member[(_e362 + _e301)] = select(0f, _e290, (_e290 > 0f));
    }
    if (_e366 && _e314) {
        matmul_relu_ragged_arg_2_.member[(_e362 + _e310)] = select(0f, _e289, (_e289 > 0f));
    }
    if (_e366 && _e322) {
        matmul_relu_ragged_arg_2_.member[(_e362 + _e318)] = select(0f, _e288, (_e288 > 0f));
    }
    if (_e366 && _e330) {
        matmul_relu_ragged_arg_2_.member[(_e362 + _e326)] = select(0f, _e287, (_e287 > 0f));
    }
    let _e388 = (_e70 + _e77);
    let _e389 = (_e388 * 750u);
    let _e393 = (_e388 < 1000u);
    if (_e393 && _e306) {
        matmul_relu_ragged_arg_2_.member[(_e389 + _e301)] = select(0f, _e286, (_e286 > 0f));
    }
    if (_e393 && _e314) {
        matmul_relu_ragged_arg_2_.member[(_e389 + _e310)] = select(0f, _e285, (_e285 > 0f));
    }
    if (_e393 && _e322) {
        matmul_relu_ragged_arg_2_.member[(_e389 + _e318)] = select(0f, _e284, (_e284 > 0f));
    }
    if (_e393 && _e330) {
        matmul_relu_ragged_arg_2_.member[(_e389 + _e326)] = select(0f, _e283, (_e283 > 0f));
    }
    return;
}

@compute @workgroup_size(16, 16, 1) 
fn matmul_relu_ragged(@builtin(workgroup_id) _builtin_WorkgroupId: vec3<u32>, @builtin(local_invocation_id) _builtin_LocalInvocationId: vec3<u32>) {
    _builtin_WorkgroupId_1 = _builtin_WorkgroupId;
    _builtin_LocalInvocationId_1 = _builtin_LocalInvocationId;
    matmul_relu_ragged_1();
}
