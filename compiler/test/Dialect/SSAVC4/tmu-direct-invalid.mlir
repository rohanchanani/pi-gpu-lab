// RUN: not vc4-opt %s -o /dev/null 2>&1 | FileCheck %s

ssavc4.module @tmu_direct_invalid {
  ssavc4.func @missing_mode() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
    // CHECK: requires attribute 'mode'
    %tok = ssavc4.tmu.request %addr {unit = "tmu0"} : vector<16xi32> -> !ssavc4.async.token
    %value = ssavc4.tmu.read %tok {unit = "tmu0", part = "raw32"} : !ssavc4.async.token -> vector<16xf32>
    ssavc4.thread_end
  }
}
