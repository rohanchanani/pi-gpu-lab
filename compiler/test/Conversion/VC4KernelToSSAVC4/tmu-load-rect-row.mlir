// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @tmu_rect
// CHECK: ssavc4.cond_br
// CHECK: ssavc4.cond_br
// CHECK: ssavc4.cond_br
// CHECK: ssavc4.cond_select
// CHECK: ssavc4.cond_select
// CHECK: ssavc4.tmu.request
// CHECK: ssavc4.tmu.read
// CHECK: ssavc4.cond_select
// CHECK: ssavc4.cond_select
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @tmu_rect(%ptr : i32, %row : i32, %rows : i32, %col_base : i32, %cols : i32) attributes {
    public_name = "tmu_rect",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "ptr", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "row", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "rows", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "col_base", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "cols", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %rect = vc4kernel.pred.rect %row, %rows, %col_base, %cols : i32, i32, i32, i32 -> !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %offs_shift = vc4kernel.splat %c2 : i32 -> vector<16xi32>
    %offs = vc4kernel.fragment_alu.add %lanes, %offs_shift {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v = vc4kernel.tmu_load_fragment %ptr, %offs, %rect : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.return
  }
}
