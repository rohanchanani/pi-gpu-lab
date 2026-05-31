// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s

// CHECK-LABEL: vc4kernel.kernel @with_args(%arg0 : i32, %arg1 : f32) attributes {
// CHECK-SAME: arg_attrs = [{direction = "in", elem_type = "f32", kind = "buffer", name = "ptr"}, {direction = "by_value", kind = "scalar", name = "scale", type = "f32"}]
// CHECK-SAME: public_name = "with_args"
// CHECK-SAME: resource = {
// CHECK-SAME: uses_vpm = false
// CHECK-SAME: schedule_mode = #vc4kernel.schedule_mode<independent_vector>
// CHECK: vc4kernel.return
module {
  vc4kernel.kernel @with_args(%ptr : i32, %scale : f32) attributes {
    public_name = "with_args",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "ptr", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "scale", kind = "scalar", direction = "by_value", type = "f32"}
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
    vc4kernel.return
  }
}
