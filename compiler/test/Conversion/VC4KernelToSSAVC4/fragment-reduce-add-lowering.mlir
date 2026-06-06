// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @fragment_reduce_add
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_select
// CHECK: ssavc4.rotate {{.*}} {amount = 8 : i32}
// CHECK: ssavc4.alu.add
// CHECK-SAME: #vc4.add_opcode<fadd>
// CHECK: ssavc4.rotate {{.*}} {amount = 4 : i32}
// CHECK: ssavc4.rotate {{.*}} {amount = 2 : i32}
// CHECK: ssavc4.rotate {{.*}} {amount = 1 : i32}
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @fragment_reduce_add(%out : i32, %input : i32, %threshold : i32) attributes {
    public_name = "fragment_reduce_add",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "f32"},
      {name = "input", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "threshold", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %byte_offsets_shift = vc4kernel.splat %c2 : i32 -> vector<16xi32>
    %byte_offsets = vc4kernel.fragment_alu.add %lanes, %byte_offsets_shift {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %threshold_v = vc4kernel.splat %threshold : i32 -> vector<16xi32>
    %mask = vc4kernel.fragment_cmp %lanes, %threshold_v {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %values = vc4kernel.tmu_load_fragment %input, %byte_offsets, %full : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xf32>
    %sum = vc4kernel.fragment_reduce %values, %mask {kind = #vc4kernel.reduce<add>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %sum, %full : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
