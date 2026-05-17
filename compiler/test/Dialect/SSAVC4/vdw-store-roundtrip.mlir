// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: ssavc4.module @vdw_store_roundtrip
// CHECK: ssavc4.vdw.store
// CHECK-SAME: active_lanes = 16 : i32
// CHECK-SAME: elem_bytes = 4 : i32
// CHECK-SAME: serialize = "mutex"
ssavc4.module @vdw_store_roundtrip {
  ssavc4.func @store_kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %value = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
    ssavc4.vdw.store %addr, %value {elem_bytes = 4 : i32, active_lanes = 16 : i32, vpm_row = 0 : i32, serialize = "mutex", lowering_template = "independent_vector_u32"} : i32, vector<16xi32>
    ssavc4.thread_end
  }
}
