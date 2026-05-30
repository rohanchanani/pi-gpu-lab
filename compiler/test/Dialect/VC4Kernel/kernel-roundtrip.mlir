// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s
// CHECK-LABEL: vc4kernel.kernel @minimal
// CHECK-NOT: <invalid-symbol>
// CHECK: public_name = "minimal"
// CHECK: vc4kernel.return
module {
  vc4kernel.kernel @minimal attributes {
    public_name = "minimal",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
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
    vc4kernel.return
  }
}
