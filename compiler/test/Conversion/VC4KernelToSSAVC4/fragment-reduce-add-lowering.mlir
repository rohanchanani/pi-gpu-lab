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
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %byte_offsets = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %threshold_v = vc4kernel.splat %threshold : i32 -> vector<16xi32>
    %mask = vc4kernel.fragment_cmp %lanes, %threshold_v {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %p7_safe0 = arith.constant 0 : i32
    %values = vc4kernel.tmu_load_fragment %input, %byte_offsets, %full, %p7_safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %sum = vc4kernel.fragment_reduce %values, %mask {kind = #vc4kernel.reduce<add>, fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %sum, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
