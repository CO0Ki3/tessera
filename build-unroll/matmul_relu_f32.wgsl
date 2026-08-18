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
    var phi_274_: u32;
    var phi_276_: f32;
    var phi_277_: f32;
    var phi_278_: f32;
    var phi_279_: f32;
    var phi_280_: f32;
    var phi_281_: f32;
    var phi_282_: f32;
    var phi_283_: f32;
    var phi_284_: f32;
    var phi_285_: f32;
    var phi_286_: f32;
    var phi_287_: f32;
    var phi_288_: f32;
    var phi_289_: f32;
    var phi_290_: f32;
    var phi_291_: f32;
    var phi_297_: u32;
    var phi_313_: u32;

    let _e106 = _builtin_WorkgroupId_1;
    let _e108 = _builtin_WorkgroupId_1;
    let _e110 = _builtin_LocalInvocationId_1;
    let _e112 = _builtin_LocalInvocationId_1;
    let _e115 = ((_e112.y * 16u) + _e110.x);
    let _e116 = (_e112.y * 4u);
    let _e117 = (_e110.x * 4u);
    let _e118 = (_e108.y * 64u);
    let _e119 = (_e106.x * 64u);
    let _e120 = (_e112.y * 64u);
    let _e121 = (_e116 + 1u);
    let _e122 = (_e121 * 16u);
    let _e123 = (_e116 + 2u);
    let _e124 = (_e123 * 16u);
    let _e125 = (_e116 + 3u);
    let _e126 = (_e125 * 16u);
    let _e127 = (_e117 + 1u);
    let _e128 = (_e117 + 2u);
    let _e129 = (_e117 + 3u);
    phi_274_ = 0u;
    phi_276_ = 0f;
    phi_277_ = 0f;
    phi_278_ = 0f;
    phi_279_ = 0f;
    phi_280_ = 0f;
    phi_281_ = 0f;
    phi_282_ = 0f;
    phi_283_ = 0f;
    phi_284_ = 0f;
    phi_285_ = 0f;
    phi_286_ = 0f;
    phi_287_ = 0f;
    phi_288_ = 0f;
    phi_289_ = 0f;
    phi_290_ = 0f;
    phi_291_ = 0f;
    loop {
        let _e251 = phi_274_;
        let _e253 = phi_276_;
        let _e255 = phi_277_;
        let _e257 = phi_278_;
        let _e259 = phi_279_;
        let _e261 = phi_280_;
        let _e263 = phi_281_;
        let _e265 = phi_282_;
        let _e267 = phi_283_;
        let _e269 = phi_284_;
        let _e271 = phi_285_;
        let _e273 = phi_286_;
        let _e275 = phi_287_;
        let _e277 = phi_288_;
        let _e279 = phi_289_;
        let _e281 = phi_290_;
        let _e283 = phi_291_;
        if (bitcast<i32>(_e251) < bitcast<i32>(512u)) {
            continue;
        } else {
            break;
        }
        continuing {
            phi_297_ = _e115;
            loop {
                let _e288 = phi_297_;
                if (bitcast<i32>(_e288) < bitcast<i32>(1024u)) {
                    continue;
                } else {
                    break;
                }
                continuing {
                    let _e300 = matmul_relu_f32_arg_0_.member[(((_e118 + (_e288 / 16u)) * 512u) + (_e251 + (_e288 % 16u)))];
                    _workgroup_mem_0_.member[_e288] = _e300;
                    phi_297_ = (_e288 + 256u);
                }
            }
            phi_313_ = _e115;
            loop {
                let _e305 = phi_313_;
                if (bitcast<i32>(_e305) < bitcast<i32>(1024u)) {
                    continue;
                } else {
                    break;
                }
                continuing {
                    let _e317 = matmul_relu_f32_arg_1_.member[(((_e251 + (_e305 / 64u)) * 768u) + (_e119 + (_e305 % 64u)))];
                    _workgroup_mem_1_.member[_e305] = _e317;
                    phi_313_ = (_e305 + 256u);
                }
            }
            workgroupBarrier();
            let _e323 = _workgroup_mem_0_.member[_e120];
            let _e326 = _workgroup_mem_0_.member[_e122];
            let _e329 = _workgroup_mem_0_.member[_e124];
            let _e332 = _workgroup_mem_0_.member[_e126];
            let _e335 = _workgroup_mem_1_.member[_e117];
            let _e338 = _workgroup_mem_1_.member[_e127];
            let _e341 = _workgroup_mem_1_.member[_e128];
            let _e344 = _workgroup_mem_1_.member[_e129];
            let _e379 = _workgroup_mem_0_.member[(_e120 + 1u)];
            let _e382 = _workgroup_mem_0_.member[(_e122 + 1u)];
            let _e385 = _workgroup_mem_0_.member[(_e124 + 1u)];
            let _e388 = _workgroup_mem_0_.member[(_e126 + 1u)];
            let _e391 = _workgroup_mem_1_.member[(_e117 + 64u)];
            let _e394 = _workgroup_mem_1_.member[(_e117 + 65u)];
            let _e397 = _workgroup_mem_1_.member[(_e117 + 66u)];
            let _e400 = _workgroup_mem_1_.member[(_e117 + 67u)];
            let _e435 = _workgroup_mem_0_.member[(_e120 + 2u)];
            let _e438 = _workgroup_mem_0_.member[(_e122 + 2u)];
            let _e441 = _workgroup_mem_0_.member[(_e124 + 2u)];
            let _e444 = _workgroup_mem_0_.member[(_e126 + 2u)];
            let _e447 = _workgroup_mem_1_.member[(_e117 + 128u)];
            let _e450 = _workgroup_mem_1_.member[(_e117 + 129u)];
            let _e453 = _workgroup_mem_1_.member[(_e117 + 130u)];
            let _e456 = _workgroup_mem_1_.member[(_e117 + 131u)];
            let _e491 = _workgroup_mem_0_.member[(_e120 + 3u)];
            let _e494 = _workgroup_mem_0_.member[(_e122 + 3u)];
            let _e497 = _workgroup_mem_0_.member[(_e124 + 3u)];
            let _e500 = _workgroup_mem_0_.member[(_e126 + 3u)];
            let _e503 = _workgroup_mem_1_.member[(_e117 + 192u)];
            let _e506 = _workgroup_mem_1_.member[(_e117 + 193u)];
            let _e509 = _workgroup_mem_1_.member[(_e117 + 194u)];
            let _e512 = _workgroup_mem_1_.member[(_e117 + 195u)];
            let _e547 = _workgroup_mem_0_.member[(_e120 + 4u)];
            let _e550 = _workgroup_mem_0_.member[(_e122 + 4u)];
            let _e553 = _workgroup_mem_0_.member[(_e124 + 4u)];
            let _e556 = _workgroup_mem_0_.member[(_e126 + 4u)];
            let _e559 = _workgroup_mem_1_.member[(_e117 + 256u)];
            let _e562 = _workgroup_mem_1_.member[(_e117 + 257u)];
            let _e565 = _workgroup_mem_1_.member[(_e117 + 258u)];
            let _e568 = _workgroup_mem_1_.member[(_e117 + 259u)];
            let _e603 = _workgroup_mem_0_.member[(_e120 + 5u)];
            let _e606 = _workgroup_mem_0_.member[(_e122 + 5u)];
            let _e609 = _workgroup_mem_0_.member[(_e124 + 5u)];
            let _e612 = _workgroup_mem_0_.member[(_e126 + 5u)];
            let _e615 = _workgroup_mem_1_.member[(_e117 + 320u)];
            let _e618 = _workgroup_mem_1_.member[(_e117 + 321u)];
            let _e621 = _workgroup_mem_1_.member[(_e117 + 322u)];
            let _e624 = _workgroup_mem_1_.member[(_e117 + 323u)];
            let _e659 = _workgroup_mem_0_.member[(_e120 + 6u)];
            let _e662 = _workgroup_mem_0_.member[(_e122 + 6u)];
            let _e665 = _workgroup_mem_0_.member[(_e124 + 6u)];
            let _e668 = _workgroup_mem_0_.member[(_e126 + 6u)];
            let _e671 = _workgroup_mem_1_.member[(_e117 + 384u)];
            let _e674 = _workgroup_mem_1_.member[(_e117 + 385u)];
            let _e677 = _workgroup_mem_1_.member[(_e117 + 386u)];
            let _e680 = _workgroup_mem_1_.member[(_e117 + 387u)];
            let _e715 = _workgroup_mem_0_.member[(_e120 + 7u)];
            let _e718 = _workgroup_mem_0_.member[(_e122 + 7u)];
            let _e721 = _workgroup_mem_0_.member[(_e124 + 7u)];
            let _e724 = _workgroup_mem_0_.member[(_e126 + 7u)];
            let _e727 = _workgroup_mem_1_.member[(_e117 + 448u)];
            let _e730 = _workgroup_mem_1_.member[(_e117 + 449u)];
            let _e733 = _workgroup_mem_1_.member[(_e117 + 450u)];
            let _e736 = _workgroup_mem_1_.member[(_e117 + 451u)];
            let _e771 = _workgroup_mem_0_.member[(_e120 + 8u)];
            let _e774 = _workgroup_mem_0_.member[(_e122 + 8u)];
            let _e777 = _workgroup_mem_0_.member[(_e124 + 8u)];
            let _e780 = _workgroup_mem_0_.member[(_e126 + 8u)];
            let _e783 = _workgroup_mem_1_.member[(_e117 + 512u)];
            let _e786 = _workgroup_mem_1_.member[(_e117 + 513u)];
            let _e789 = _workgroup_mem_1_.member[(_e117 + 514u)];
            let _e792 = _workgroup_mem_1_.member[(_e117 + 515u)];
            let _e827 = _workgroup_mem_0_.member[(_e120 + 9u)];
            let _e830 = _workgroup_mem_0_.member[(_e122 + 9u)];
            let _e833 = _workgroup_mem_0_.member[(_e124 + 9u)];
            let _e836 = _workgroup_mem_0_.member[(_e126 + 9u)];
            let _e839 = _workgroup_mem_1_.member[(_e117 + 576u)];
            let _e842 = _workgroup_mem_1_.member[(_e117 + 577u)];
            let _e845 = _workgroup_mem_1_.member[(_e117 + 578u)];
            let _e848 = _workgroup_mem_1_.member[(_e117 + 579u)];
            let _e883 = _workgroup_mem_0_.member[(_e120 + 10u)];
            let _e886 = _workgroup_mem_0_.member[(_e122 + 10u)];
            let _e889 = _workgroup_mem_0_.member[(_e124 + 10u)];
            let _e892 = _workgroup_mem_0_.member[(_e126 + 10u)];
            let _e895 = _workgroup_mem_1_.member[(_e117 + 640u)];
            let _e898 = _workgroup_mem_1_.member[(_e117 + 641u)];
            let _e901 = _workgroup_mem_1_.member[(_e117 + 642u)];
            let _e904 = _workgroup_mem_1_.member[(_e117 + 643u)];
            let _e939 = _workgroup_mem_0_.member[(_e120 + 11u)];
            let _e942 = _workgroup_mem_0_.member[(_e122 + 11u)];
            let _e945 = _workgroup_mem_0_.member[(_e124 + 11u)];
            let _e948 = _workgroup_mem_0_.member[(_e126 + 11u)];
            let _e951 = _workgroup_mem_1_.member[(_e117 + 704u)];
            let _e954 = _workgroup_mem_1_.member[(_e117 + 705u)];
            let _e957 = _workgroup_mem_1_.member[(_e117 + 706u)];
            let _e960 = _workgroup_mem_1_.member[(_e117 + 707u)];
            let _e995 = _workgroup_mem_0_.member[(_e120 + 12u)];
            let _e998 = _workgroup_mem_0_.member[(_e122 + 12u)];
            let _e1001 = _workgroup_mem_0_.member[(_e124 + 12u)];
            let _e1004 = _workgroup_mem_0_.member[(_e126 + 12u)];
            let _e1007 = _workgroup_mem_1_.member[(_e117 + 768u)];
            let _e1010 = _workgroup_mem_1_.member[(_e117 + 769u)];
            let _e1013 = _workgroup_mem_1_.member[(_e117 + 770u)];
            let _e1016 = _workgroup_mem_1_.member[(_e117 + 771u)];
            let _e1051 = _workgroup_mem_0_.member[(_e120 + 13u)];
            let _e1054 = _workgroup_mem_0_.member[(_e122 + 13u)];
            let _e1057 = _workgroup_mem_0_.member[(_e124 + 13u)];
            let _e1060 = _workgroup_mem_0_.member[(_e126 + 13u)];
            let _e1063 = _workgroup_mem_1_.member[(_e117 + 832u)];
            let _e1066 = _workgroup_mem_1_.member[(_e117 + 833u)];
            let _e1069 = _workgroup_mem_1_.member[(_e117 + 834u)];
            let _e1072 = _workgroup_mem_1_.member[(_e117 + 835u)];
            let _e1107 = _workgroup_mem_0_.member[(_e120 + 14u)];
            let _e1110 = _workgroup_mem_0_.member[(_e122 + 14u)];
            let _e1113 = _workgroup_mem_0_.member[(_e124 + 14u)];
            let _e1116 = _workgroup_mem_0_.member[(_e126 + 14u)];
            let _e1119 = _workgroup_mem_1_.member[(_e117 + 896u)];
            let _e1122 = _workgroup_mem_1_.member[(_e117 + 897u)];
            let _e1125 = _workgroup_mem_1_.member[(_e117 + 898u)];
            let _e1128 = _workgroup_mem_1_.member[(_e117 + 899u)];
            let _e1163 = _workgroup_mem_0_.member[(_e120 + 15u)];
            let _e1166 = _workgroup_mem_0_.member[(_e122 + 15u)];
            let _e1169 = _workgroup_mem_0_.member[(_e124 + 15u)];
            let _e1172 = _workgroup_mem_0_.member[(_e126 + 15u)];
            let _e1175 = _workgroup_mem_1_.member[(_e117 + 960u)];
            let _e1178 = _workgroup_mem_1_.member[(_e117 + 961u)];
            let _e1181 = _workgroup_mem_1_.member[(_e117 + 962u)];
            let _e1184 = _workgroup_mem_1_.member[(_e117 + 963u)];
            let _e1186 = ((((((((((((((((_e253 + (_e323 * _e335)) + (_e379 * _e391)) + (_e435 * _e447)) + (_e491 * _e503)) + (_e547 * _e559)) + (_e603 * _e615)) + (_e659 * _e671)) + (_e715 * _e727)) + (_e771 * _e783)) + (_e827 * _e839)) + (_e883 * _e895)) + (_e939 * _e951)) + (_e995 * _e1007)) + (_e1051 * _e1063)) + (_e1107 * _e1119)) + (_e1163 * _e1175));
            let _e1188 = ((((((((((((((((_e255 + (_e323 * _e338)) + (_e379 * _e394)) + (_e435 * _e450)) + (_e491 * _e506)) + (_e547 * _e562)) + (_e603 * _e618)) + (_e659 * _e674)) + (_e715 * _e730)) + (_e771 * _e786)) + (_e827 * _e842)) + (_e883 * _e898)) + (_e939 * _e954)) + (_e995 * _e1010)) + (_e1051 * _e1066)) + (_e1107 * _e1122)) + (_e1163 * _e1178));
            let _e1190 = ((((((((((((((((_e257 + (_e323 * _e341)) + (_e379 * _e397)) + (_e435 * _e453)) + (_e491 * _e509)) + (_e547 * _e565)) + (_e603 * _e621)) + (_e659 * _e677)) + (_e715 * _e733)) + (_e771 * _e789)) + (_e827 * _e845)) + (_e883 * _e901)) + (_e939 * _e957)) + (_e995 * _e1013)) + (_e1051 * _e1069)) + (_e1107 * _e1125)) + (_e1163 * _e1181));
            let _e1192 = ((((((((((((((((_e259 + (_e323 * _e344)) + (_e379 * _e400)) + (_e435 * _e456)) + (_e491 * _e512)) + (_e547 * _e568)) + (_e603 * _e624)) + (_e659 * _e680)) + (_e715 * _e736)) + (_e771 * _e792)) + (_e827 * _e848)) + (_e883 * _e904)) + (_e939 * _e960)) + (_e995 * _e1016)) + (_e1051 * _e1072)) + (_e1107 * _e1128)) + (_e1163 * _e1184));
            let _e1194 = ((((((((((((((((_e261 + (_e326 * _e335)) + (_e382 * _e391)) + (_e438 * _e447)) + (_e494 * _e503)) + (_e550 * _e559)) + (_e606 * _e615)) + (_e662 * _e671)) + (_e718 * _e727)) + (_e774 * _e783)) + (_e830 * _e839)) + (_e886 * _e895)) + (_e942 * _e951)) + (_e998 * _e1007)) + (_e1054 * _e1063)) + (_e1110 * _e1119)) + (_e1166 * _e1175));
            let _e1196 = ((((((((((((((((_e263 + (_e326 * _e338)) + (_e382 * _e394)) + (_e438 * _e450)) + (_e494 * _e506)) + (_e550 * _e562)) + (_e606 * _e618)) + (_e662 * _e674)) + (_e718 * _e730)) + (_e774 * _e786)) + (_e830 * _e842)) + (_e886 * _e898)) + (_e942 * _e954)) + (_e998 * _e1010)) + (_e1054 * _e1066)) + (_e1110 * _e1122)) + (_e1166 * _e1178));
            let _e1198 = ((((((((((((((((_e265 + (_e326 * _e341)) + (_e382 * _e397)) + (_e438 * _e453)) + (_e494 * _e509)) + (_e550 * _e565)) + (_e606 * _e621)) + (_e662 * _e677)) + (_e718 * _e733)) + (_e774 * _e789)) + (_e830 * _e845)) + (_e886 * _e901)) + (_e942 * _e957)) + (_e998 * _e1013)) + (_e1054 * _e1069)) + (_e1110 * _e1125)) + (_e1166 * _e1181));
            let _e1200 = ((((((((((((((((_e267 + (_e326 * _e344)) + (_e382 * _e400)) + (_e438 * _e456)) + (_e494 * _e512)) + (_e550 * _e568)) + (_e606 * _e624)) + (_e662 * _e680)) + (_e718 * _e736)) + (_e774 * _e792)) + (_e830 * _e848)) + (_e886 * _e904)) + (_e942 * _e960)) + (_e998 * _e1016)) + (_e1054 * _e1072)) + (_e1110 * _e1128)) + (_e1166 * _e1184));
            let _e1202 = ((((((((((((((((_e269 + (_e329 * _e335)) + (_e385 * _e391)) + (_e441 * _e447)) + (_e497 * _e503)) + (_e553 * _e559)) + (_e609 * _e615)) + (_e665 * _e671)) + (_e721 * _e727)) + (_e777 * _e783)) + (_e833 * _e839)) + (_e889 * _e895)) + (_e945 * _e951)) + (_e1001 * _e1007)) + (_e1057 * _e1063)) + (_e1113 * _e1119)) + (_e1169 * _e1175));
            let _e1204 = ((((((((((((((((_e271 + (_e329 * _e338)) + (_e385 * _e394)) + (_e441 * _e450)) + (_e497 * _e506)) + (_e553 * _e562)) + (_e609 * _e618)) + (_e665 * _e674)) + (_e721 * _e730)) + (_e777 * _e786)) + (_e833 * _e842)) + (_e889 * _e898)) + (_e945 * _e954)) + (_e1001 * _e1010)) + (_e1057 * _e1066)) + (_e1113 * _e1122)) + (_e1169 * _e1178));
            let _e1206 = ((((((((((((((((_e273 + (_e329 * _e341)) + (_e385 * _e397)) + (_e441 * _e453)) + (_e497 * _e509)) + (_e553 * _e565)) + (_e609 * _e621)) + (_e665 * _e677)) + (_e721 * _e733)) + (_e777 * _e789)) + (_e833 * _e845)) + (_e889 * _e901)) + (_e945 * _e957)) + (_e1001 * _e1013)) + (_e1057 * _e1069)) + (_e1113 * _e1125)) + (_e1169 * _e1181));
            let _e1208 = ((((((((((((((((_e275 + (_e329 * _e344)) + (_e385 * _e400)) + (_e441 * _e456)) + (_e497 * _e512)) + (_e553 * _e568)) + (_e609 * _e624)) + (_e665 * _e680)) + (_e721 * _e736)) + (_e777 * _e792)) + (_e833 * _e848)) + (_e889 * _e904)) + (_e945 * _e960)) + (_e1001 * _e1016)) + (_e1057 * _e1072)) + (_e1113 * _e1128)) + (_e1169 * _e1184));
            let _e1210 = ((((((((((((((((_e277 + (_e332 * _e335)) + (_e388 * _e391)) + (_e444 * _e447)) + (_e500 * _e503)) + (_e556 * _e559)) + (_e612 * _e615)) + (_e668 * _e671)) + (_e724 * _e727)) + (_e780 * _e783)) + (_e836 * _e839)) + (_e892 * _e895)) + (_e948 * _e951)) + (_e1004 * _e1007)) + (_e1060 * _e1063)) + (_e1116 * _e1119)) + (_e1172 * _e1175));
            let _e1212 = ((((((((((((((((_e279 + (_e332 * _e338)) + (_e388 * _e394)) + (_e444 * _e450)) + (_e500 * _e506)) + (_e556 * _e562)) + (_e612 * _e618)) + (_e668 * _e674)) + (_e724 * _e730)) + (_e780 * _e786)) + (_e836 * _e842)) + (_e892 * _e898)) + (_e948 * _e954)) + (_e1004 * _e1010)) + (_e1060 * _e1066)) + (_e1116 * _e1122)) + (_e1172 * _e1178));
            let _e1214 = ((((((((((((((((_e281 + (_e332 * _e341)) + (_e388 * _e397)) + (_e444 * _e453)) + (_e500 * _e509)) + (_e556 * _e565)) + (_e612 * _e621)) + (_e668 * _e677)) + (_e724 * _e733)) + (_e780 * _e789)) + (_e836 * _e845)) + (_e892 * _e901)) + (_e948 * _e957)) + (_e1004 * _e1013)) + (_e1060 * _e1069)) + (_e1116 * _e1125)) + (_e1172 * _e1181));
            let _e1216 = ((((((((((((((((_e283 + (_e332 * _e344)) + (_e388 * _e400)) + (_e444 * _e456)) + (_e500 * _e512)) + (_e556 * _e568)) + (_e612 * _e624)) + (_e668 * _e680)) + (_e724 * _e736)) + (_e780 * _e792)) + (_e836 * _e848)) + (_e892 * _e904)) + (_e948 * _e960)) + (_e1004 * _e1016)) + (_e1060 * _e1072)) + (_e1116 * _e1128)) + (_e1172 * _e1184));
            workgroupBarrier();
            local = _e1186;
            local_1 = _e1188;
            local_2 = _e1190;
            local_3 = _e1192;
            local_4 = _e1194;
            local_5 = _e1196;
            local_6 = _e1198;
            local_7 = _e1200;
            local_8 = _e1202;
            local_9 = _e1204;
            local_10 = _e1206;
            local_11 = _e1208;
            local_12 = _e1210;
            local_13 = _e1212;
            local_14 = _e1214;
            local_15 = _e1216;
            phi_274_ = (_e251 + 16u);
            phi_276_ = _e1186;
            phi_277_ = _e1188;
            phi_278_ = _e1190;
            phi_279_ = _e1192;
            phi_280_ = _e1194;
            phi_281_ = _e1196;
            phi_282_ = _e1198;
            phi_283_ = _e1200;
            phi_284_ = _e1202;
            phi_285_ = _e1204;
            phi_286_ = _e1206;
            phi_287_ = _e1208;
            phi_288_ = _e1210;
            phi_289_ = _e1212;
            phi_290_ = _e1214;
            phi_291_ = _e1216;
        }
    }
    let _e1218 = local_15;
    let _e1219 = local_14;
    let _e1220 = local_13;
    let _e1221 = local_12;
    let _e1222 = local_11;
    let _e1223 = local_10;
    let _e1224 = local_9;
    let _e1225 = local_8;
    let _e1226 = local_7;
    let _e1227 = local_6;
    let _e1228 = local_5;
    let _e1229 = local_4;
    let _e1230 = local_3;
    let _e1231 = local_2;
    let _e1232 = local_1;
    let _e1233 = local;
    let _e1235 = ((_e118 + _e116) * 768u);
    let _e1236 = (_e119 + _e117);
    matmul_relu_f32_arg_2_.member[(_e1235 + _e1236)] = select(0f, _e1233, (_e1233 > 0f));
    let _e1242 = (_e119 + _e127);
    matmul_relu_f32_arg_2_.member[(_e1235 + _e1242)] = select(0f, _e1232, (_e1232 > 0f));
    let _e1248 = (_e119 + _e128);
    matmul_relu_f32_arg_2_.member[(_e1235 + _e1248)] = select(0f, _e1231, (_e1231 > 0f));
    let _e1254 = (_e119 + _e129);
    matmul_relu_f32_arg_2_.member[(_e1235 + _e1254)] = select(0f, _e1230, (_e1230 > 0f));
    let _e1261 = ((_e118 + _e121) * 768u);
    matmul_relu_f32_arg_2_.member[(_e1261 + _e1236)] = select(0f, _e1229, (_e1229 > 0f));
    matmul_relu_f32_arg_2_.member[(_e1261 + _e1242)] = select(0f, _e1228, (_e1228 > 0f));
    matmul_relu_f32_arg_2_.member[(_e1261 + _e1248)] = select(0f, _e1227, (_e1227 > 0f));
    matmul_relu_f32_arg_2_.member[(_e1261 + _e1254)] = select(0f, _e1226, (_e1226 > 0f));
    let _e1283 = ((_e118 + _e123) * 768u);
    matmul_relu_f32_arg_2_.member[(_e1283 + _e1236)] = select(0f, _e1225, (_e1225 > 0f));
    matmul_relu_f32_arg_2_.member[(_e1283 + _e1242)] = select(0f, _e1224, (_e1224 > 0f));
    matmul_relu_f32_arg_2_.member[(_e1283 + _e1248)] = select(0f, _e1223, (_e1223 > 0f));
    matmul_relu_f32_arg_2_.member[(_e1283 + _e1254)] = select(0f, _e1222, (_e1222 > 0f));
    let _e1305 = ((_e118 + _e125) * 768u);
    matmul_relu_f32_arg_2_.member[(_e1305 + _e1236)] = select(0f, _e1221, (_e1221 > 0f));
    matmul_relu_f32_arg_2_.member[(_e1305 + _e1242)] = select(0f, _e1220, (_e1220 > 0f));
    matmul_relu_f32_arg_2_.member[(_e1305 + _e1248)] = select(0f, _e1219, (_e1219 > 0f));
    matmul_relu_f32_arg_2_.member[(_e1305 + _e1254)] = select(0f, _e1218, (_e1218 > 0f));
    return;
}

@compute @workgroup_size(16, 16, 1) 
fn matmul_relu_f32_(@builtin(workgroup_id) _builtin_WorkgroupId: vec3<u32>, @builtin(local_invocation_id) _builtin_LocalInvocationId: vec3<u32>) {
    _builtin_WorkgroupId_1 = _builtin_WorkgroupId;
    _builtin_LocalInvocationId_1 = _builtin_LocalInvocationId;
    matmul_relu_f32_1();
}
