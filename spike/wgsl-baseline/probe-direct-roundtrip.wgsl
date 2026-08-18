struct type_6 {
    member: array<f32>,
}

@group(0) @binding(0) 
var<storage> global: type_6;
@group(0) @binding(1) 
var<storage> global_1: type_6;
@group(0) @binding(2) 
var<storage, read_write> global_2: type_6;
var<workgroup> global_3: array<f32, 1024>;
var<workgroup> global_4: array<f32, 1024>;
var<private> global_5: vec3<u32>;
var<private> global_6: vec3<u32>;
var<private> global_7: u32;

fn function_(param_3: u32, param_4: u32) -> u32 {
    return (param_3 / select(param_4, 1u, (param_4 == 0u)));
}

fn function_1(param_5: u32, param_6: u32) -> u32 {
    return (param_5 % select(param_6, 1u, (param_6 == 0u)));
}

fn function_2() {
    var local: u32 = u32();
    var local_1: u32 = 0u;
    var local_2: u32 = u32();
    var local_3: array<f32, 16> = array<f32, 16>();
    var local_4: u32 = u32();
    var local_5: vec2<u32> = vec2<u32>(4294967295u, 4294967295u);
    var local_6: vec2<u32> = vec2<u32>(4294967295u, 4294967295u);
    var local_7: vec2<u32> = vec2<u32>(4294967295u, 4294967295u);
    var local_8: vec2<u32> = vec2<u32>(4294967295u, 4294967295u);

    let _e58 = global_5;
    let _e59 = global_6;
    let _e63 = global_7;
    let _e70 = ((_e59.y * 16u) + _e59.x);
    let _e71 = (_e58.y * 64u);
    let _e72 = (_e58.x * 64u);
    loop {
        let _e73 = local_5;
        if all((vec2<u32>(0u, 0u) == _e73)) {
            break;
        }
        local_5 = (_e73 - vec2<u32>(select(0u, 1u, (_e73.y == 0u)), 1u));
        let _e81 = local_1;
        if (_e81 < 512u) {
        } else {
            break;
        }
        local_4 = _e70;
        loop {
            let _e83 = local_6;
            if all((vec2<u32>(0u, 0u) == _e83)) {
                break;
            }
            local_6 = (_e83 - vec2<u32>(select(0u, 1u, (_e83.y == 0u)), 1u));
            let _e91 = local_4;
            if (_e91 < 1024u) {
            } else {
                break;
            }
            let _e93 = local_4;
            let _e94 = function_(_e93, 16u);
            let _e95 = local_4;
            let _e96 = function_1(_e95, 16u);
            let _e98 = local_1;
            let _e102 = local_4;
            let _e104 = global.member[(((_e71 + _e94) * 512u) + (_e98 + _e96))];
            global_3[_e102] = _e104;
            continue;
            continuing {
                let _e106 = local_4;
                local_4 = (_e106 + 256u);
            }
        }
        local_2 = _e70;
        loop {
            let _e108 = local_7;
            if all((vec2<u32>(0u, 0u) == _e108)) {
                break;
            }
            local_7 = (_e108 - vec2<u32>(select(0u, 1u, (_e108.y == 0u)), 1u));
            let _e116 = local_2;
            if (_e116 < 1024u) {
            } else {
                break;
            }
            let _e118 = local_2;
            let _e119 = function_(_e118, 64u);
            let _e120 = local_2;
            let _e121 = function_1(_e120, 64u);
            let _e122 = local_1;
            let _e127 = local_2;
            let _e129 = global_1.member[(((_e122 + _e119) * 768u) + (_e72 + _e121))];
            global_4[_e127] = _e129;
            continue;
            continuing {
                let _e131 = local_2;
                local_2 = (_e131 + 256u);
            }
        }
        workgroupBarrier();
        local = 0u;
        loop {
            let _e133 = local_8;
            if all((vec2<u32>(0u, 0u) == _e133)) {
                break;
            }
            local_8 = (_e133 - vec2<u32>(select(0u, 1u, (_e133.y == 0u)), 1u));
            let _e141 = local;
            if (_e141 < 16u) {
            } else {
                break;
            }
            let _e146 = local;
            let _e149 = global_3[((((_e59.y * 4u) + 0u) * 16u) + _e146)];
            let _e153 = local;
            let _e156 = global_3[((((_e59.y * 4u) + 1u) * 16u) + _e153)];
            let _e160 = local;
            let _e163 = global_3[((((_e59.y * 4u) + 2u) * 16u) + _e160)];
            let _e167 = local;
            let _e170 = global_3[((((_e59.y * 4u) + 3u) * 16u) + _e167)];
            let _e171 = local;
            let _e177 = global_4[((_e171 * 64u) + ((_e59.x * 4u) + 0u))];
            let _e178 = local;
            let _e184 = global_4[((_e178 * 64u) + ((_e59.x * 4u) + 1u))];
            let _e185 = local;
            let _e191 = global_4[((_e185 * 64u) + ((_e59.x * 4u) + 2u))];
            let _e192 = local;
            let _e198 = global_4[((_e192 * 64u) + ((_e59.x * 4u) + 3u))];
            let _e200 = local_3[0u];
            local_3[0u] = (_e200 + (_e149 * _e177));
            let _e205 = local_3[1u];
            local_3[1u] = (_e205 + (_e149 * _e184));
            let _e210 = local_3[2u];
            local_3[2u] = (_e210 + (_e149 * _e191));
            let _e215 = local_3[3u];
            local_3[3u] = (_e215 + (_e149 * _e198));
            let _e220 = local_3[4u];
            local_3[4u] = (_e220 + (_e156 * _e177));
            let _e225 = local_3[5u];
            local_3[5u] = (_e225 + (_e156 * _e184));
            let _e230 = local_3[6u];
            local_3[6u] = (_e230 + (_e156 * _e191));
            let _e235 = local_3[7u];
            local_3[7u] = (_e235 + (_e156 * _e198));
            let _e240 = local_3[8u];
            local_3[8u] = (_e240 + (_e163 * _e177));
            let _e245 = local_3[9u];
            local_3[9u] = (_e245 + (_e163 * _e184));
            let _e250 = local_3[10u];
            local_3[10u] = (_e250 + (_e163 * _e191));
            let _e255 = local_3[11u];
            local_3[11u] = (_e255 + (_e163 * _e198));
            let _e260 = local_3[12u];
            local_3[12u] = (_e260 + (_e170 * _e177));
            let _e265 = local_3[13u];
            local_3[13u] = (_e265 + (_e170 * _e184));
            let _e270 = local_3[14u];
            local_3[14u] = (_e270 + (_e170 * _e191));
            let _e275 = local_3[15u];
            local_3[15u] = (_e275 + (_e170 * _e198));
            continue;
            continuing {
                let _e279 = local;
                local = (_e279 + 1u);
            }
        }
        workgroupBarrier();
        continue;
        continuing {
            let _e281 = local_1;
            local_1 = (_e281 + 16u);
        }
    }
    let _e285 = ((_e71 + (_e59.y * 4u)) + 0u);
    let _e292 = local_3[0u];
    let _e294 = local_3[0u];
    global_2.member[((_e285 * 768u) + ((_e72 + (_e59.x * 4u)) + 0u))] = select(0f, _e292, (_e294 > 0f));
    let _e304 = local_3[1u];
    let _e306 = local_3[1u];
    global_2.member[((_e285 * 768u) + ((_e72 + (_e59.x * 4u)) + 1u))] = select(0f, _e304, (_e306 > 0f));
    let _e316 = local_3[2u];
    let _e318 = local_3[2u];
    global_2.member[((_e285 * 768u) + ((_e72 + (_e59.x * 4u)) + 2u))] = select(0f, _e316, (_e318 > 0f));
    let _e328 = local_3[3u];
    let _e330 = local_3[3u];
    global_2.member[((_e285 * 768u) + ((_e72 + (_e59.x * 4u)) + 3u))] = select(0f, _e328, (_e330 > 0f));
    let _e336 = ((_e71 + (_e59.y * 4u)) + 1u);
    let _e343 = local_3[4u];
    let _e345 = local_3[4u];
    global_2.member[((_e336 * 768u) + ((_e72 + (_e59.x * 4u)) + 0u))] = select(0f, _e343, (_e345 > 0f));
    let _e355 = local_3[5u];
    let _e357 = local_3[5u];
    global_2.member[((_e336 * 768u) + ((_e72 + (_e59.x * 4u)) + 1u))] = select(0f, _e355, (_e357 > 0f));
    let _e367 = local_3[6u];
    let _e369 = local_3[6u];
    global_2.member[((_e336 * 768u) + ((_e72 + (_e59.x * 4u)) + 2u))] = select(0f, _e367, (_e369 > 0f));
    let _e379 = local_3[7u];
    let _e381 = local_3[7u];
    global_2.member[((_e336 * 768u) + ((_e72 + (_e59.x * 4u)) + 3u))] = select(0f, _e379, (_e381 > 0f));
    let _e387 = ((_e71 + (_e59.y * 4u)) + 2u);
    let _e394 = local_3[8u];
    let _e396 = local_3[8u];
    global_2.member[((_e387 * 768u) + ((_e72 + (_e59.x * 4u)) + 0u))] = select(0f, _e394, (_e396 > 0f));
    let _e406 = local_3[9u];
    let _e408 = local_3[9u];
    global_2.member[((_e387 * 768u) + ((_e72 + (_e59.x * 4u)) + 1u))] = select(0f, _e406, (_e408 > 0f));
    let _e418 = local_3[10u];
    let _e420 = local_3[10u];
    global_2.member[((_e387 * 768u) + ((_e72 + (_e59.x * 4u)) + 2u))] = select(0f, _e418, (_e420 > 0f));
    let _e430 = local_3[11u];
    let _e432 = local_3[11u];
    global_2.member[((_e387 * 768u) + ((_e72 + (_e59.x * 4u)) + 3u))] = select(0f, _e430, (_e432 > 0f));
    let _e438 = ((_e71 + (_e59.y * 4u)) + 3u);
    let _e445 = local_3[12u];
    let _e447 = local_3[12u];
    global_2.member[((_e438 * 768u) + ((_e72 + (_e59.x * 4u)) + 0u))] = select(0f, _e445, (_e447 > 0f));
    let _e457 = local_3[13u];
    let _e459 = local_3[13u];
    global_2.member[((_e438 * 768u) + ((_e72 + (_e59.x * 4u)) + 1u))] = select(0f, _e457, (_e459 > 0f));
    let _e469 = local_3[14u];
    let _e471 = local_3[14u];
    global_2.member[((_e438 * 768u) + ((_e72 + (_e59.x * 4u)) + 2u))] = select(0f, _e469, (_e471 > 0f));
    let _e481 = local_3[15u];
    let _e483 = local_3[15u];
    global_2.member[((_e438 * 768u) + ((_e72 + (_e59.x * 4u)) + 3u))] = select(0f, _e481, (_e483 > 0f));
    return;
}

@compute @workgroup_size(16, 16, 1) 
fn matmul_relu_f32_(@builtin(workgroup_id) param: vec3<u32>, @builtin(local_invocation_id) param_1: vec3<u32>, @builtin(local_invocation_index) param_2: u32) {
    global_5 = param;
    global_6 = param_1;
    global_7 = param_2;
    function_2();
}
