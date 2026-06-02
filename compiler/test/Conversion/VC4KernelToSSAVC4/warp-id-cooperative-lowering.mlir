// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @warp_id_cooperative
// CHECK-SAME: logical_warp_id
// CHECK: %[[WARP:.*]] = ssavc4.uniform.read 1 : i32
// CHECK: ssavc4.splat %[[WARP]] : i32 -> vector<16xi32>
// CHECK-NOT: QPU_NUMBER
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @warp_id_cooperative(%out : i32) attributes {
    public_name = "warp_id_cooperative",
    schedule_mode = #vc4kernel.schedule_mode<cooperative_block>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "u32"}
    ],
    warps_per_block = 4 : i32
  } {
    %warp = vc4kernel.warp_id : i32
    vc4kernel.splat %warp : i32 -> vector<16xi32>
    vc4kernel.return
  }
}
