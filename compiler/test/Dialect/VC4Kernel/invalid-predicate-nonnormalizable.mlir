// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s
module {
  vc4kernel.kernel @bad(%base : i32, %limit : i32) attributes {
    public_name = "bad",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "base", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "limit", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    resource = {
      uses_vpm = false,
      uses_barrier = false,
      require_full_block_residency = false,
      warps_per_block_max = 1 : i32,
      vpm_rows_per_block = 0 : i32,
      vpm_bytes_per_block = 0 : i32,
      semaphores_per_block = 0 : i32
    }
  } {
    %c1 = arith.constant 1 : i32
    %tail0 = vc4kernel.pred.tail %base, %limit : i32, i32 -> !vc4kernel.pred<16>
    %tail1_base = arith.addi %base, %c1 : i32
    %tail1 = vc4kernel.pred.tail %tail1_base, %limit : i32, i32 -> !vc4kernel.pred<16>
    // CHECK: predicate expression is not normalizable
    %bad = vc4kernel.pred.or %tail0, %tail1 : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    vc4kernel.return
  }
}
