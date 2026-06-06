// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @fragment_pack_unpack_lowering
// CHECK: ssavc4.unpack {{.*}} {mode = #vc4.regfile_a_unpack_mode<color8a>} : vector<16xi32> -> vector<16xi32>
// CHECK: ssavc4.unpack {{.*}} {mode = #vc4.regfile_a_unpack_mode<f16a_or_i16a>} : vector<16xi32> -> vector<16xi32>
// CHECK: ssavc4.pack {{.*}} {mode = #vc4.regfile_a_pack_mode<to_8a>} : vector<16xi32> -> vector<16xi32>
// CHECK: ssavc4.pack {{.*}} {mode = #vc4.regfile_a_pack_mode<to_16a>} : vector<16xi32> -> vector<16xi32>
// CHECK-NOT: vc4.qpu
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @fragment_pack_unpack_lowering(%x : i32) attributes {
    public_name = "fragment_pack_unpack_lowering",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : i32 -> vector<16xi32>
    %u8 = vc4kernel.fragment_unpack %v {source = #vc4kernel.subword_type<u8>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.unpack_policy<zero_extend>} : vector<16xi32> -> vector<16xi32>
    %s16 = vc4kernel.fragment_unpack %v {source = #vc4kernel.subword_type<s16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.unpack_policy<sign_extend>} : vector<16xi32> -> vector<16xi32>
    %p8 = vc4kernel.fragment_pack %u8 {dest = #vc4kernel.subword_type<u8>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.pack_policy<truncate>} : vector<16xi32> -> vector<16xi32>
    %p16 = vc4kernel.fragment_pack %s16 {dest = #vc4kernel.subword_type<u16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.pack_policy<truncate>} : vector<16xi32> -> vector<16xi32>
    vc4kernel.return
  }
}
