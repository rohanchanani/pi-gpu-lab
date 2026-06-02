// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @general_mask_materialized_combinators
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<cs>}
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<zs>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<and>}
// CHECK: ssavc4.tmu.request
// CHECK: ssavc4.tmu.read
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<zc>}
// CHECK: ssavc4.vdw.store
// CHECK-NOT: vc4kernel.
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
    %c2 = arith.constant 2 : i32
    %tag = arith.constant 1515870810 : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %byte_offsets = vc4kernel.fragment_shl %lanes, %c2 : vector<16xi32>, i32 -> vector<16xi32>
    %lo_v = vc4kernel.splat %lo : i32 -> vector<16xi32>
    %hi_v = vc4kernel.splat %hi : i32 -> vector<16xi32>
    %tag_v = vc4kernel.splat %tag : i32 -> vector<16xi32>
    %below_hi = vc4kernel.fragment_cmp %lanes, %hi_v {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %below_lo = vc4kernel.fragment_cmp %lanes, %lo_v {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %at_or_above_lo = vc4kernel.pred.not %below_lo : !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %mask = vc4kernel.pred.and %below_hi, %at_or_above_lo : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %value = vc4kernel.fragment_add %tag_v, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %value, %mask : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
