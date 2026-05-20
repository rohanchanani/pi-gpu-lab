// RUN: not vc4-opt %s --convert-ssavc4-to-vc4 -o /dev/null 2>&1 | FileCheck %s

// CHECK: P2 block-argument lowering supports only acyclic merges; loop-carried block arguments are deferred to P3
ssavc4.module @block_args_backedge_unsupported {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    ssavc4.br ^loop(%zero : i32)
  ^loop(%i: i32):
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %next = ssavc4.alu.add %i, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.br ^loop(%next : i32)
  }
}
