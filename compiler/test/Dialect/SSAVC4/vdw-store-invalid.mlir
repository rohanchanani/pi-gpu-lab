// RUN: not vc4-opt %s -o /dev/null 2>&1 | FileCheck %s

ssavc4.module @vdw_store_invalid {
  ssavc4.func @bad_store_kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %value = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
    // CHECK: error:
    ssavc4.vdw.store %addr, %value {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, active_lanes = 16 : i32, vpm_row = 0 : i32, serialize = "mutex"} : vector<16xi32>, i32
    ssavc4.thread_end
  }
}
