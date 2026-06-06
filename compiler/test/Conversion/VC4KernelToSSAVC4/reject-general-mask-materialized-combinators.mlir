// RUN: not vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 2>&1 | FileCheck %s

// CHECK: sparse VDW store masks are not supported in P8
module {
  vc4kernel.kernel @general_mask_materialized_combinators(%out : i32, %lo : i32, %hi : i32) attributes {
    public_name = "general_mask_materialized_combinators",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "inout", elem_type = "i32"},
      {name = "lo", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "hi", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %byte_offsets = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %lo_v = vc4kernel.splat %lo : i32 -> vector<16xi32>
    %hi_v = vc4kernel.splat %hi : i32 -> vector<16xi32>
    %tag_v = vc4kernel.fragment_const {value = dense<1515870810> : vector<16xi32>} : vector<16xi32>
    %below_hi = vc4kernel.fragment_cmp %lanes, %hi_v {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %below_lo = vc4kernel.fragment_cmp %lanes, %lo_v {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %at_or_above_lo = vc4kernel.pred.not %below_lo : !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %mask = vc4kernel.pred.and %below_hi, %at_or_above_lo : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %value = vc4kernel.fragment_alu.add %tag_v, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %value, %mask {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
