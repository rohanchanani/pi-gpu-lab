// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s

module {
  vc4kernel.kernel @fragment_pack_unpack(%x : i32) attributes {
    public_name = "fragment_pack_unpack",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : i32 -> vector<16xi32>
    // CHECK: vc4kernel.fragment_unpack
    // CHECK-SAME: layout = #vc4kernel.subword_layout<packed>
    // CHECK-SAME: policy = #vc4kernel.unpack_policy<zero_extend>
    // CHECK-SAME: source = #vc4kernel.subword_type<u8>
    %u8 = vc4kernel.fragment_unpack %v {source = #vc4kernel.subword_type<u8>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.unpack_policy<zero_extend>} : vector<16xi32> -> vector<16xi32>
    // CHECK: vc4kernel.fragment_unpack
    // CHECK-SAME: policy = #vc4kernel.unpack_policy<sign_extend>
    // CHECK-SAME: source = #vc4kernel.subword_type<s16>
    %s16 = vc4kernel.fragment_unpack %v {source = #vc4kernel.subword_type<s16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.unpack_policy<sign_extend>} : vector<16xi32> -> vector<16xi32>
    // CHECK: vc4kernel.fragment_pack
    // CHECK-SAME: dest = #vc4kernel.subword_type<u8>
    // CHECK-SAME: policy = #vc4kernel.pack_policy<truncate>
    %p8 = vc4kernel.fragment_pack %u8 {dest = #vc4kernel.subword_type<u8>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.pack_policy<truncate>} : vector<16xi32> -> vector<16xi32>
    // CHECK: vc4kernel.fragment_pack
    // CHECK-SAME: dest = #vc4kernel.subword_type<u16>
    // CHECK-SAME: policy = #vc4kernel.pack_policy<truncate>
    %p16 = vc4kernel.fragment_pack %s16 {dest = #vc4kernel.subword_type<u16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.pack_policy<truncate>} : vector<16xi32> -> vector<16xi32>
    // CHECK: vc4kernel.fragment_unpack
    // CHECK-SAME: policy = #vc4kernel.unpack_policy<to_f32>
    // CHECK-SAME: source = #vc4kernel.subword_type<f16>
    %f16 = vc4kernel.fragment_unpack %v {source = #vc4kernel.subword_type<f16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.unpack_policy<to_f32>} : vector<16xi32> -> vector<16xf32>
    // CHECK: vc4kernel.fragment_pack
    // CHECK-SAME: dest = #vc4kernel.subword_type<f16>
    // CHECK-SAME: policy = #vc4kernel.pack_policy<from_f32>
    %pf16 = vc4kernel.fragment_pack %f16 {dest = #vc4kernel.subword_type<f16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.pack_policy<from_f32>} : vector<16xf32> -> vector<16xi32>
    vc4kernel.return
  }
}
