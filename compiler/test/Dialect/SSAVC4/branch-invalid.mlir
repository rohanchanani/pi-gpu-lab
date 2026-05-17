// RUN: not vc4-opt %s -o /dev/null 2>&1 | FileCheck %s

ssavc4.module @branch_invalid {
  ssavc4.func @bad_kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    // CHECK: error:
    ssavc4.cond_br %zero, ^done, ^done {cond = #vc4.branch_cond<any_c_set>} : i32
  ^done:
    ssavc4.thread_end
  }
}
