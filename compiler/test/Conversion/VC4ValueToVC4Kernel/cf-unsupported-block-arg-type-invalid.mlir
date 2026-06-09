// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @cf_unsupported_block_arg_type_invalid()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %h = arith.constant 0.000000e+00 : f16
  cf.br ^use(%h : f16)

^use(%arg: f16):
  return
}

// CHECK: block argument type 'f16' is not Phase 8 value control-flow lowerable
// CHECK: READY_FOR_TRITON remains NO
