// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.module @vc4kernel_lowered
// CHECK: vc4.resource
// CHECK-NOT: shared{{_}}vpm{{_}}bytes
// CHECK-NOT: user{{_}}shared{{_}}vpm{{_}}rows{{_}}per{{_}}block
// CHECK-NOT: vpm{{_}}bytes{{_}}per{{_}}block
// CHECK-NOT: warps{{_}}per{{_}}block{{_}}max
// CHECK-NOT: uses{{_}}shared{{_}}vpm
// CHECK-NOT: require{{_}}full{{_}}block{{_}}residency
// CHECK-NOT: semaphores{{_}}per{{_}}block
// CHECK-NOT: vc4.qpu
// CHECK-NOT: vc4kernel.
// CHECK: ssavc4.thread_end
module {
  vc4kernel.kernel @no_old_fields(%out : i32) attributes {
    public_name = "no_old_fields",
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
