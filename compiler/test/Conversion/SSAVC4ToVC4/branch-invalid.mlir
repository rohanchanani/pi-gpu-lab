// RUN: not vc4-opt %s --convert-ssavc4-to-vc4 -o /dev/null 2>&1 | FileCheck %s

ssavc4.module @branch_invalid_lowering {
  ssavc4.func @bad_kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %flags = ssavc4.make_flags %zero, %one {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %flags, ^done, ^body {cond = #vc4.branch_cond<any_c_set>} : !ssavc4.flags
  ^body:
    // CHECK: result must be consumed exactly once
    %unused = ssavc4.make_flags %zero, %one {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.br ^done
  ^done:
    ssavc4.thread_end
  }
}
