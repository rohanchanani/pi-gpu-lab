// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck %s

// CHECK-LABEL: ssavc4.func @block_args_tail_loop
// CHECK: ssavc4.br
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_br
// CHECK: ssavc4.thread_end
// CHECK-NOT: vc4tile.
vc4tile.kernel @block_args_tail_loop attributes {public_name = "block_args_tail_loop"} {
  %zero = arith.constant 0 : i32
  %limit = arith.constant 16 : i32
  cf.br ^loop(%zero : i32)
^loop(%base: i32):
  %mask = vc4tile.core_tail_mask %base, %limit : i32, i32 -> vector<16xi1>
  %step = arith.constant 16 : i32
  %next = arith.addi %base, %step : i32
  %done = arith.cmpi ult, %base, %limit : i32
  cf.cond_br %done, ^loop(%next : i32), ^exit
^exit:
  vc4tile.return
}
