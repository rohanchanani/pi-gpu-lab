// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s
module {
  // CHECK-LABEL: vc4kernel.kernel @independent_source_shape
  // CHECK-SAME: schedule_mode = #vc4kernel.schedule_mode<independent_vector>
  // CHECK-SAME: warps_per_block = 1 : i32
  vc4kernel.kernel @independent_source_shape attributes {
    public_name = "independent_source_shape",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    vc4kernel.return
  }

  // CHECK-LABEL: vc4kernel.kernel @cooperative_source_shape
  // CHECK-SAME: schedule_mode = #vc4kernel.schedule_mode<cooperative_block>
  // CHECK-SAME: warps_per_block = 4 : i32
  vc4kernel.kernel @cooperative_source_shape attributes {
    public_name = "cooperative_source_shape",
    schedule_mode = #vc4kernel.schedule_mode<cooperative_block>,
    arg_attrs = [],
    warps_per_block = 4 : i32
  } {
    vc4kernel.return
  }
}
