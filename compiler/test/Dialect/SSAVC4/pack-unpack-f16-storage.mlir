// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: ssavc4.module @pack_unpack_f16_storage
// CHECK: ssavc4.unpack
// CHECK-SAME: f16_storage_conversion
// CHECK-SAME: mode = #vc4.regfile_a_unpack_mode<f16a_or_i16a>
// CHECK-SAME: vector<16xi32> -> vector<16xf32>
// CHECK: ssavc4.pack
// CHECK-SAME: f16_storage_conversion
// CHECK-SAME: mode = #vc4.regfile_a_pack_mode<to_16a>
// CHECK-SAME: vector<16xf32> -> vector<16xi32>
ssavc4.module @pack_unpack_f16_storage {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %carrier = ssavc4.load_imm <splat32> {value = 15360 : i32} : vector<16xi32>
    %f = ssavc4.unpack %carrier {f16_storage_conversion, mode = #vc4.regfile_a_unpack_mode<f16a_or_i16a>} : vector<16xi32> -> vector<16xf32>
    %packed = ssavc4.pack %f {f16_storage_conversion, mode = #vc4.regfile_a_pack_mode<to_16a>} : vector<16xf32> -> vector<16xi32>
    ssavc4.thread_end
  }
}
