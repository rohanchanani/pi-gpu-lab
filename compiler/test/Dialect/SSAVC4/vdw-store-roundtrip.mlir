// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: ssavc4.module @vdw_store_roundtrip
// CHECK: ssavc4.vdw.store
// CHECK-SAME: active_lanes = 16 : i32
// CHECK-SAME: serialize = "mutex"
// CHECK-SAME: subword = #ssavc4.vpm_subword<none>
// CHECK-SAME: vpm_row = 0 : i32
// CHECK-SAME: width = #ssavc4.vpm_elem_width<w32>
ssavc4.module @vdw_store_roundtrip {
  ssavc4.func @store_kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %value = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
    ssavc4.vdw.store %addr, %value {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, active_lanes = 16 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.thread_end
  }
}
