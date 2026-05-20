// RUN: not vc4-opt %s --convert-ssavc4-to-vc4 2>&1 | FileCheck %s

// CHECK: P3 block-argument lowering does not yet support cyclic parallel copies on loop backedges
ssavc4.module @block_args_loop_cycle_unsupported {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    ssavc4.br ^loop(%zero, %one : i32, i32)

  ^loop(%a: i32, %b: i32):
    ssavc4.br ^loop(%b, %a : i32, i32)
  }
}
