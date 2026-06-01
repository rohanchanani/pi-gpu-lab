// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s
module {
  vc4kernel.kernel @bad attributes {
    public_name = "bad",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    %t = arith.constant true
    %f = arith.constant false
    // CHECK: integer arith operations in vc4kernel require scalar i32 operands and results
    %r = arith.addi %t, %f : i1
    vc4kernel.return
  }
}
