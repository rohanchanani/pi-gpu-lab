// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s
// CHECK-LABEL: ssavc4.func @bar
// CHECK-SAME: schedule_mode = "cooperative_block"
// CHECK: ssavc4.barrier
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @bar attributes {
    public_name = "bar",
    schedule_mode = #vc4kernel.schedule_mode<cooperative_block>,
    arg_attrs = [],
    resource = {
      uses_vpm = false, uses_barrier = true,
      require_full_block_residency = true,
      warps_per_block_max = 4 : i32, vpm_rows_per_block = 0 : i32,
      vpm_bytes_per_block = 0 : i32, semaphores_per_block = 4 : i32
    }
  } {
    vc4kernel.barrier
    vc4kernel.return
  }
}
