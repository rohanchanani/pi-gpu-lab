// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @tmu_tail
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_br
// CHECK: ssavc4.cond_select
// CHECK: ssavc4.tmu.request
// CHECK: ssavc4.tmu.read
// CHECK: ssavc4.cond_select
// CHECK: ssavc4.br
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @tmu_tail(%ptr : i32, %base_index : i32, %limit : i32) attributes {
    public_name = "tmu_tail",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "ptr", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "base_index", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "limit", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %base_bytes = arith.shli %base_index, %c2 : i32
    %tail = vc4kernel.pred.tail %base_index, %limit : i32, i32 -> !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_shl %lanes, %c2 : vector<16xi32>, i32 -> vector<16xi32>
    %base_bytes_v = vc4kernel.splat %base_bytes : i32 -> vector<16xi32>
    %offs = vc4kernel.fragment_add %base_bytes_v, %lane_bytes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %v = vc4kernel.tmu_load_fragment %ptr, %offs, %tail : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.return
  }
}
