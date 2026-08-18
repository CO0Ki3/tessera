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
    let _e68 = ((_e59.y * 16u) + _e59.x);
    let _e70 = (_e58.y * 64u);
    let _e72 = (_e58.x * 64u);
    local_3[0u] = 0f;
    local_3[1u] = 0f;
    local_3[2u] = 0f;
    local_3[3u] = 0f;
    local_3[4u] = 0f;
    local_3[5u] = 0f;
    local_3[6u] = 0f;
    local_3[7u] = 0f;
    local_3[8u] = 0f;
    local_3[9u] = 0f;
    local_3[10u] = 0f;
    local_3[11u] = 0f;
    local_3[12u] = 0f;
    local_3[13u] = 0f;
    local_3[14u] = 0f;
    local_3[15u] = 0f;
    loop {
        let _e89 = local_5;
        if all((vec2<u32>(0u, 0u) == _e89)) {
            break;
        }
        local_5 = (_e89 - vec2<u32>(select(0u, 1u, (_e89.y == 0u)), 1u));
        let _e97 = local_1;
        if (_e97 < 512u) {
        } else {
            break;
        }
        local_4 = _e68;
        loop {
            let _e99 = local_6;
            if all((vec2<u32>(0u, 0u) == _e99)) {
                break;
            }
            local_6 = (_e99 - vec2<u32>(select(0u, 1u, (_e99.y == 0u)), 1u));
            let _e107 = local_4;
            if (_e107 < 1024u) {
            } else {
                break;
            }
            let _e109 = local_4;
            let _e110 = function_(_e109, 16u);
            let _e111 = local_4;
            let _e112 = function_1(_e111, 16u);
            let _e114 = local_1;
            let _e118 = local_4;
            let _e120 = global.member[(((_e70 + _e110) * 512u) + (_e114 + _e112))];
            global_3[_e118] = _e120;
            continue;
            continuing {
                let _e122 = local_4;
                local_4 = (_e122 + 256u);
            }
        }
        local_2 = _e68;
        loop {
            let _e124 = local_7;
            if all((vec2<u32>(0u, 0u) == _e124)) {
                break;
            }
            local_7 = (_e124 - vec2<u32>(select(0u, 1u, (_e124.y == 0u)), 1u));
            let _e132 = local_2;
            if (_e132 < 1024u) {
            } else {
                break;
            }
            let _e134 = local_2;
            let _e135 = function_(_e134, 64u);
            let _e136 = local_2;
            let _e137 = function_1(_e136, 64u);
            let _e138 = local_1;
            let _e143 = local_2;
            let _e145 = global_1.member[(((_e138 + _e135) * 768u) + (_e72 + _e137))];
            global_4[_e143] = _e145;
            continue;
            continuing {
                let _e147 = local_2;
                local_2 = (_e147 + 256u);
            }
        }
        workgroupBarrier();
        local = 0u;
        loop {
            let _e169 = local_8;
            if all((vec2<u32>(0u, 0u) == _e169)) {
                break;
            }
            local_8 = (_e169 - vec2<u32>(select(0u, 1u, (_e169.y == 0u)), 1u));
            let _e177 = local;
            if (_e177 < 16u) {
            } else {
                break;
            }
            let _e179 = local;
            let _e180 = local_1;
            let _e184 = global_3[((((_e59.y * 4u) + 0u) * 16u) + _e179)];
            let _e187 = global_3[((((_e59.y * 4u) + 1u) * 16u) + _e179)];
            let _e190 = global_3[((((_e59.y * 4u) + 2u) * 16u) + _e179)];
            let _e193 = global_3[((((_e59.y * 4u) + 3u) * 16u) + _e179)];
            let _e197 = global_4[((_e179 * 64u) + ((_e59.x * 4u) + 0u))];
            let _e201 = global_4[((_e179 * 64u) + ((_e59.x * 4u) + 1u))];
            let _e205 = global_4[((_e179 * 64u) + ((_e59.x * 4u) + 2u))];
            let _e209 = global_4[((_e179 * 64u) + ((_e59.x * 4u) + 3u))];
            let _e211 = local_3[0u];
            local_3[0u] = (_e211 + (_e184 * _e197));
            let _e216 = local_3[1u];
            local_3[1u] = (_e216 + (_e184 * _e201));
            let _e221 = local_3[2u];
            local_3[2u] = (_e221 + (_e184 * _e205));
            let _e226 = local_3[3u];
            local_3[3u] = (_e226 + (_e184 * _e209));
            let _e231 = local_3[4u];
            local_3[4u] = (_e231 + (_e187 * _e197));
            let _e236 = local_3[5u];
            local_3[5u] = (_e236 + (_e187 * _e201));
            let _e241 = local_3[6u];
            local_3[6u] = (_e241 + (_e187 * _e205));
            let _e246 = local_3[7u];
            local_3[7u] = (_e246 + (_e187 * _e209));
            let _e251 = local_3[8u];
            local_3[8u] = (_e251 + (_e190 * _e197));
            let _e256 = local_3[9u];
            local_3[9u] = (_e256 + (_e190 * _e201));
            let _e261 = local_3[10u];
            local_3[10u] = (_e261 + (_e190 * _e205));
            let _e266 = local_3[11u];
            local_3[11u] = (_e266 + (_e190 * _e209));
            let _e271 = local_3[12u];
            local_3[12u] = (_e271 + (_e193 * _e197));
            let _e276 = local_3[13u];
            local_3[13u] = (_e276 + (_e193 * _e201));
            let _e281 = local_3[14u];
            local_3[14u] = (_e281 + (_e193 * _e205));
            let _e286 = local_3[15u];
            local_3[15u] = (_e286 + (_e193 * _e209));
            continue;
            continuing {
                let _e290 = local;
                local = (_e290 + 1u);
            }
        }
        workgroupBarrier();
        continue;
        continuing {
            let _e292 = local_1;
            local_1 = (_e292 + 16u);
        }
    }
    let _e303 = local_3[0u];
    let _e305 = local_3[0u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 0u) * 768u) + ((_e72 + (_e59.x * 4u)) + 0u))] = select(0f, _e303, (_e305 > 0f));
    let _e318 = local_3[1u];
    let _e320 = local_3[1u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 0u) * 768u) + ((_e72 + (_e59.x * 4u)) + 1u))] = select(0f, _e318, (_e320 > 0f));
    let _e333 = local_3[2u];
    let _e335 = local_3[2u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 0u) * 768u) + ((_e72 + (_e59.x * 4u)) + 2u))] = select(0f, _e333, (_e335 > 0f));
    let _e348 = local_3[3u];
    let _e350 = local_3[3u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 0u) * 768u) + ((_e72 + (_e59.x * 4u)) + 3u))] = select(0f, _e348, (_e350 > 0f));
    let _e363 = local_3[4u];
    let _e365 = local_3[4u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 1u) * 768u) + ((_e72 + (_e59.x * 4u)) + 0u))] = select(0f, _e363, (_e365 > 0f));
    let _e378 = local_3[5u];
    let _e380 = local_3[5u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 1u) * 768u) + ((_e72 + (_e59.x * 4u)) + 1u))] = select(0f, _e378, (_e380 > 0f));
    let _e393 = local_3[6u];
    let _e395 = local_3[6u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 1u) * 768u) + ((_e72 + (_e59.x * 4u)) + 2u))] = select(0f, _e393, (_e395 > 0f));
    let _e408 = local_3[7u];
    let _e410 = local_3[7u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 1u) * 768u) + ((_e72 + (_e59.x * 4u)) + 3u))] = select(0f, _e408, (_e410 > 0f));
    let _e423 = local_3[8u];
    let _e425 = local_3[8u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 2u) * 768u) + ((_e72 + (_e59.x * 4u)) + 0u))] = select(0f, _e423, (_e425 > 0f));
    let _e438 = local_3[9u];
    let _e440 = local_3[9u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 2u) * 768u) + ((_e72 + (_e59.x * 4u)) + 1u))] = select(0f, _e438, (_e440 > 0f));
    let _e453 = local_3[10u];
    let _e455 = local_3[10u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 2u) * 768u) + ((_e72 + (_e59.x * 4u)) + 2u))] = select(0f, _e453, (_e455 > 0f));
    let _e468 = local_3[11u];
    let _e470 = local_3[11u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 2u) * 768u) + ((_e72 + (_e59.x * 4u)) + 3u))] = select(0f, _e468, (_e470 > 0f));
    let _e483 = local_3[12u];
    let _e485 = local_3[12u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 3u) * 768u) + ((_e72 + (_e59.x * 4u)) + 0u))] = select(0f, _e483, (_e485 > 0f));
    let _e498 = local_3[13u];
    let _e500 = local_3[13u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 3u) * 768u) + ((_e72 + (_e59.x * 4u)) + 1u))] = select(0f, _e498, (_e500 > 0f));
    let _e513 = local_3[14u];
    let _e515 = local_3[14u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 3u) * 768u) + ((_e72 + (_e59.x * 4u)) + 2u))] = select(0f, _e513, (_e515 > 0f));
    let _e528 = local_3[15u];
    let _e530 = local_3[15u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 3u) * 768u) + ((_e72 + (_e59.x * 4u)) + 3u))] = select(0f, _e528, (_e530 > 0f));
    return;
}

@compute @workgroup_size(16, 16, 1) 
fn matmul_relu_f32_(@builtin(workgroup_id) param: vec3<u32>, @builtin(local_invocation_id) param_1: vec3<u32>, @builtin(local_invocation_index) param_2: u32) {
    global_5 = param;
    global_6 = param_1;
    global_7 = param_2;
    function_2();
}
