// RUN: not vc4-opt %s --convert-ssavc4-to-vc4 2>&1 | FileCheck %s

// CHECK: P3 block-argument lowering supports only natural loops with conservative loop-carried data values
ssavc4.module @block_args_irreducible_loop_unsupported {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %flags = ssavc4.make_flags %zero, %one {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %flags, ^loop(%zero : i32), ^bypass(%one : i32) {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^loop(%i: i32):
    ssavc4.br ^done

  ^bypass(%j: i32):
    ssavc4.br ^body(%j : i32)

  ^body(%k: i32):
    %next = ssavc4.alu.add %k, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.br ^loop(%next : i32)

  ^done:
    ssavc4.thread_end
  }
}
