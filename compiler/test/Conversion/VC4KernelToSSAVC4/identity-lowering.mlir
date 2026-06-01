// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s
// CHECK-LABEL: ssavc4.func @ids
// CHECK-SAME: builtins = [
// CHECK-SAME: logical_request
// CHECK-SAME: logical_warp_id
// CHECK: ssavc4.uniform.read
// CHECK: ssavc4.uniform.read
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @ids attributes {
    public_name = "ids",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    resource = {
      uses_vpm = false, uses_barrier = false,
      require_full_block_residency = false,
      warps_per_block_max = 1 : i32, vpm_rows_per_block = 0 : i32,
      vpm_bytes_per_block = 0 : i32, semaphores_per_block = 0 : i32
    }
  } {
    %p = vc4kernel.program_id : i32
    %w = vc4kernel.warp_id : i32
    vc4kernel.return
  }
}
