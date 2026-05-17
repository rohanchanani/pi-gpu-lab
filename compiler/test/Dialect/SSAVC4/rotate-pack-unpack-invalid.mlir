// RUN: not vc4-opt %s -o /dev/null 2>&1 | FileCheck %s

ssavc4.module @rotate_pack_unpack_invalid {
  ssavc4.func @bad_rotate_amount_attr() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %v = ssavc4.load_imm <splat32> {value = 1 : i32} : vector<16xi32>
    // CHECK: error:
    %r = ssavc4.rotate %v {amount = "bad"} : vector<16xi32> -> vector<16xi32>
    ssavc4.thread_end
  }
}
