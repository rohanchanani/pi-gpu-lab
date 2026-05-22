// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | vc4-opt --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses | FileCheck %s

// CHECK-LABEL: vc4.func @block_args_loop_vc4tile
// CHECK: vc4.qpu.
vc4tile.kernel @block_args_loop_vc4tile attributes {
  public_name = "block_args_loop_vc4tile",
  launch_abi = {
    public_name = "block_args_loop_vc4tile",
    code_symbol = "block_args_loop_vc4tile_shader",
    tail_policy = "exact_multiple",
    uniform_words_per_qpu = 1 : i32,
    args = [{direction = "by_value", kind = "scalar", name = "dummy", type = "u32", uniform_index = 0 : i32}],
    builtins = []
  }
} {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  cf.br ^loop(%zero : i32)
^loop(%i: i32):
  %next = arith.addi %i, %one : i32
  %limit = arith.constant 1 : i32
  %pred = arith.cmpi eq, %i, %limit : i32
  cf.cond_br %pred, ^done, ^loop(%next : i32)
^done:
  vc4tile.return
}
