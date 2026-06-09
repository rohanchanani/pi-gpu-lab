// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @cf_memref_block_arg_invalid(
    %in: memref<64xf32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  cf.br ^use(%in : memref<64xf32, #vc4value.global>)

^use(%arg: memref<64xf32, #vc4value.global>):
  return
}

// CHECK: memref successor operands are not in the Phase 8 VC4 value control-flow subset
// CHECK: memref block arguments are not in the Phase 8 VC4 value control-flow subset
