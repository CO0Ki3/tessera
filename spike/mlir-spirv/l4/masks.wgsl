struct type_5 {
    member: array<f32, 64>,
}

var<private> _builtin_LocalInvocationId_1: vec3<u32>;
@group(0) @binding(0) 
var<storage, read_write> masked_arg_0_: type_5;
@group(0) @binding(1) 
var<storage, read_write> masked_arg_1_: type_5;

fn masked_1() {
    let _e9 = _builtin_LocalInvocationId_1;
    let _e11 = (_e9.x < 40u);
    let _e15 = masked_arg_0_.member[min(_e9.x, 63u)];
    if _e11 {
        masked_arg_1_.member[_e9.x] = select(0f, _e15, _e11);
    }
    return;
}

@compute @workgroup_size(16, 1, 1) 
fn masked(@builtin(local_invocation_id) _builtin_LocalInvocationId: vec3<u32>) {
    _builtin_LocalInvocationId_1 = _builtin_LocalInvocationId;
    masked_1();
}
