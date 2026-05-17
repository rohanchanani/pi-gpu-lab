// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: ssavc4.module @rotate_pack_unpack_roundtrip
// CHECK: ssavc4.rotate
// CHECK-SAME: amount = 8 : i32
// CHECK: ssavc4.pack
// CHECK-SAME: mode = #vc4.regfile_a_pack_mode<to_16a>
// CHECK: ssavc4.unpack
ssavc4.module @rotate_pack_unpack_roundtrip {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %v = ssavc4.load_imm <splat32> {value = 1 : i32} : vector<16xi32>
    %r = ssavc4.rotate %v {amount = 8 : i32} : vector<16xi32> -> vector<16xi32>
    %p = ssavc4.pack %r {mode = #vc4.regfile_a_pack_mode<to_16a>} : vector<16xi32> -> vector<16xi32>
    %u = ssavc4.unpack %p : vector<16xi32> -> vector<16xi32>
    ssavc4.thread_end
  }
}
