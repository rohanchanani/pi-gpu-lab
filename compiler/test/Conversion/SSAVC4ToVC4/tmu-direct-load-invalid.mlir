// RUN: not vc4-opt %s --convert-ssavc4-to-vc4 -o /dev/null 2>&1 | FileCheck %s

ssavc4.module @tmu_direct_load_invalid_lowering {
  ssavc4.func @bad_tmu_unit() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
    // CHECK: supports only unit = "tmu0"
    %tok = ssavc4.tmu.request %addr {unit = "tmu1", mode = "direct"} : vector<16xi32> -> !ssavc4.async.token
    %value = ssavc4.tmu.read %tok {unit = "tmu0", part = "raw32"} : !ssavc4.async.token -> vector<16xi32>
    ssavc4.thread_end
  }
}
