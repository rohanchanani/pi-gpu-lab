// RUN: not vc4-opt %s 2>&1 | FileCheck %s

ssavc4.module @sema_barrier_invalid {
  ssavc4.func @bad_sema_operand() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %v = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
    // CHECK: error:
    ssavc4.sema.release %v {id = 0 : i32} : vector<16xi32>
    ssavc4.thread_end
  }
}
