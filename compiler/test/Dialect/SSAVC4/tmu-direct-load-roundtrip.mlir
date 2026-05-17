// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: ssavc4.module @tmu_direct_load_roundtrip
// CHECK: ssavc4.tmu.request
// CHECK-SAME: mode = "direct"
// CHECK-SAME: unit = "tmu0"
// CHECK: ssavc4.tmu.read
// CHECK-SAME: part = "raw32"
ssavc4.module @tmu_direct_load_roundtrip {
  ssavc4.func @tmu_kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
    %tok = ssavc4.tmu.request %addr {unit = "tmu0", mode = "direct"} : vector<16xi32> -> !ssavc4.async.token
    %value = ssavc4.tmu.read %tok {unit = "tmu0", part = "raw32"} : !ssavc4.async.token -> vector<16xf32>
    ssavc4.thread_end
  }
}
