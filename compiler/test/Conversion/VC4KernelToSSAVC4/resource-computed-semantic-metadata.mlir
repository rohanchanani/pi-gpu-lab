// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @resource_vdw
// CHECK-SAME: builtins = [
// CHECK-SAME: vpm_base_row
// CHECK-SAME: vc4.resource = {
// CHECK-SAME: compiler_vpm_staging_rows_per_block = 0 : i32
// CHECK-SAME: compiler_vpm_staging_rows_per_warp = 1 : i32
// CHECK-SAME: requires_vpm_base_row_builtin = true
// CHECK-SAME: schedule_mode = "independent_vector"
// CHECK-SAME: total_vpm_rows_per_block = 1 : i32
// CHECK-SAME: uses_vdw = true
// CHECK-SAME: uses_vpm = true
// CHECK-NOT: shared{{_}}vpm{{_}}bytes
// CHECK-NOT: user{{_}}shared{{_}}vpm{{_}}rows{{_}}per{{_}}block
// CHECK-NOT: vpm{{_}}bytes{{_}}per{{_}}block
// CHECK-NOT: warps{{_}}per{{_}}block{{_}}max
// CHECK-NOT: uses{{_}}shared{{_}}vpm
// CHECK-NOT: require{{_}}full{{_}}block{{_}}residency
// CHECK-NOT: semaphores{{_}}per{{_}}block
// CHECK: ssavc4.vdw.store {{.*}} : i32, vector<16xi32>, i32
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @resource_vdw(%out : i32) attributes {
    public_name = "resource_vdw",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "out", kind = "buffer", direction = "out", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %offs = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %value = vc4kernel.fragment_const {value = dense<1> : vector<16xi32>} : vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %offs, %value, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
