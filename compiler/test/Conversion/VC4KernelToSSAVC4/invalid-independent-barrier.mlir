// RUN: not vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 2>&1 | FileCheck %s

// CHECK: vc4kernel.barrier requires schedule_mode = cooperative_block
module {
  vc4kernel.kernel @invalid_independent_barrier attributes {
    public_name = "invalid_independent_barrier",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    vc4kernel.barrier
    vc4kernel.return
  }
}
