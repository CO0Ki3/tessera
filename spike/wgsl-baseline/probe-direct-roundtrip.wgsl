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
            let _e149 = local_8;
            if all((vec2<u32>(0u, 0u) == _e149)) {
                break;
            }
            local_8 = (_e149 - vec2<u32>(select(0u, 1u, (_e149.y == 0u)), 1u));
            let _e157 = local;
            if (_e157 < 16u) {
            } else {
                break;
            }
            let _e159 = local_1;
            let _e160 = local;
            let _e161 = (_e159 + _e160);
            let _e165 = local_1;
            let _e169 = global_3[((((_e59.y * 4u) + 0u) * 16u) + (_e161 - _e165))];
            let _e173 = local_1;
            let _e177 = global_3[((((_e59.y * 4u) + 1u) * 16u) + (_e161 - _e173))];
            let _e181 = local_1;
            let _e185 = global_3[((((_e59.y * 4u) + 2u) * 16u) + (_e161 - _e181))];
            let _e189 = local_1;
            let _e193 = global_3[((((_e59.y * 4u) + 3u) * 16u) + (_e161 - _e189))];
            let _e194 = local_1;
            let _e201 = global_4[(((_e161 - _e194) * 64u) + ((_e59.x * 4u) + 0u))];
            let _e202 = local_1;
            let _e209 = global_4[(((_e161 - _e202) * 64u) + ((_e59.x * 4u) + 1u))];
            let _e210 = local_1;
            let _e217 = global_4[(((_e161 - _e210) * 64u) + ((_e59.x * 4u) + 2u))];
            let _e218 = local_1;
            let _e225 = global_4[(((_e161 - _e218) * 64u) + ((_e59.x * 4u) + 3u))];
            let _e227 = local_3[0u];
            local_3[0u] = (_e227 + (_e169 * _e201));
            let _e232 = local_3[1u];
            local_3[1u] = (_e232 + (_e169 * _e209));
            let _e237 = local_3[2u];
            local_3[2u] = (_e237 + (_e169 * _e217));
            let _e242 = local_3[3u];
            local_3[3u] = (_e242 + (_e169 * _e225));
            let _e247 = local_3[4u];
            local_3[4u] = (_e247 + (_e177 * _e201));
            let _e252 = local_3[5u];
            local_3[5u] = (_e252 + (_e177 * _e209));
            let _e257 = local_3[6u];
            local_3[6u] = (_e257 + (_e177 * _e217));
            let _e262 = local_3[7u];
            local_3[7u] = (_e262 + (_e177 * _e225));
            let _e267 = local_3[8u];
            local_3[8u] = (_e267 + (_e185 * _e201));
            let _e272 = local_3[9u];
            local_3[9u] = (_e272 + (_e185 * _e209));
            let _e277 = local_3[10u];
            local_3[10u] = (_e277 + (_e185 * _e217));
            let _e282 = local_3[11u];
            local_3[11u] = (_e282 + (_e185 * _e225));
            let _e287 = local_3[12u];
            local_3[12u] = (_e287 + (_e193 * _e201));
            let _e292 = local_3[13u];
            local_3[13u] = (_e292 + (_e193 * _e209));
            let _e297 = local_3[14u];
            local_3[14u] = (_e297 + (_e193 * _e217));
            let _e302 = local_3[15u];
            local_3[15u] = (_e302 + (_e193 * _e225));
            continue;
            continuing {
                let _e306 = local;
                local = (_e306 + 1u);
            }
        }
        workgroupBarrier();
        continue;
        continuing {
            let _e308 = local_1;
            local_1 = (_e308 + 16u);
        }
    }
    let _e319 = local_3[0u];
    let _e321 = local_3[0u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 0u) * 768u) + ((_e72 + (_e59.x * 4u)) + 0u))] = select(0f, _e319, (_e321 > 0f));
    let _e334 = local_3[1u];
    let _e336 = local_3[1u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 0u) * 768u) + ((_e72 + (_e59.x * 4u)) + 1u))] = select(0f, _e334, (_e336 > 0f));
    let _e349 = local_3[2u];
    let _e351 = local_3[2u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 0u) * 768u) + ((_e72 + (_e59.x * 4u)) + 2u))] = select(0f, _e349, (_e351 > 0f));
    let _e364 = local_3[3u];
    let _e366 = local_3[3u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 0u) * 768u) + ((_e72 + (_e59.x * 4u)) + 3u))] = select(0f, _e364, (_e366 > 0f));
    let _e379 = local_3[4u];
    let _e381 = local_3[4u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 1u) * 768u) + ((_e72 + (_e59.x * 4u)) + 0u))] = select(0f, _e379, (_e381 > 0f));
    let _e394 = local_3[5u];
    let _e396 = local_3[5u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 1u) * 768u) + ((_e72 + (_e59.x * 4u)) + 1u))] = select(0f, _e394, (_e396 > 0f));
    let _e409 = local_3[6u];
    let _e411 = local_3[6u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 1u) * 768u) + ((_e72 + (_e59.x * 4u)) + 2u))] = select(0f, _e409, (_e411 > 0f));
    let _e424 = local_3[7u];
    let _e426 = local_3[7u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 1u) * 768u) + ((_e72 + (_e59.x * 4u)) + 3u))] = select(0f, _e424, (_e426 > 0f));
    let _e439 = local_3[8u];
    let _e441 = local_3[8u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 2u) * 768u) + ((_e72 + (_e59.x * 4u)) + 0u))] = select(0f, _e439, (_e441 > 0f));
    let _e454 = local_3[9u];
    let _e456 = local_3[9u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 2u) * 768u) + ((_e72 + (_e59.x * 4u)) + 1u))] = select(0f, _e454, (_e456 > 0f));
    let _e469 = local_3[10u];
    let _e471 = local_3[10u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 2u) * 768u) + ((_e72 + (_e59.x * 4u)) + 2u))] = select(0f, _e469, (_e471 > 0f));
    let _e484 = local_3[11u];
    let _e486 = local_3[11u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 2u) * 768u) + ((_e72 + (_e59.x * 4u)) + 3u))] = select(0f, _e484, (_e486 > 0f));
    let _e499 = local_3[12u];
    let _e501 = local_3[12u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 3u) * 768u) + ((_e72 + (_e59.x * 4u)) + 0u))] = select(0f, _e499, (_e501 > 0f));
    let _e514 = local_3[13u];
    let _e516 = local_3[13u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 3u) * 768u) + ((_e72 + (_e59.x * 4u)) + 1u))] = select(0f, _e514, (_e516 > 0f));
    let _e529 = local_3[14u];
    let _e531 = local_3[14u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 3u) * 768u) + ((_e72 + (_e59.x * 4u)) + 2u))] = select(0f, _e529, (_e531 > 0f));
    let _e544 = local_3[15u];
    let _e546 = local_3[15u];
    global_2.member[((((_e70 + (_e59.y * 4u)) + 3u) * 768u) + ((_e72 + (_e59.x * 4u)) + 3u))] = select(0f, _e544, (_e546 > 0f));
    return;
}

@compute @workgroup_size(16, 16, 1) 
fn matmul_relu_f32_(@builtin(workgroup_id) param: vec3<u32>, @builtin(local_invocation_id) param_1: vec3<u32>, @builtin(local_invocation_index) param_2: u32) {
    global_5 = param;
    global_6 = param_1;
    global_7 = param_2;
    function_2();
}
