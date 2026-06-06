// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @fragment_reduce_empty_zero
// CHECK: ssavc4.splat
// CHECK-NOT: ssavc4.make_flags
// CHECK: ssavc4.rotate {{.*}} {amount = 8 : i32}
// CHECK: ssavc4.rotate {{.*}} {amount = 4 : i32}
// CHECK: ssavc4.rotate {{.*}} {amount = 2 : i32}
// CHECK: ssavc4.rotate {{.*}} {amount = 1 : i32}
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @fragment_reduce_empty_zero(%out : i32, %value : f32) attributes {
    public_name = "fragment_reduce_empty_zero",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "f32"},
      {name = "value", kind = "scalar", direction = "by_value", type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %empty = vc4kernel.pred.empty : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %byte_offsets = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %values = vc4kernel.splat %value : f32 -> vector<16xf32>
    %sum = vc4kernel.fragment_reduce %values, %empty {kind = #vc4kernel.reduce<add>, fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %sum, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
