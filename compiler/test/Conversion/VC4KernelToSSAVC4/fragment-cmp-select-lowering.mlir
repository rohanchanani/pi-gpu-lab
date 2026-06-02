// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @fragment_cmp_select
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<cs>}
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<zc>}
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<cc>}
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @fragment_cmp_select(%out : i32, %threshold : i32) attributes {
    public_name = "fragment_cmp_select",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "threshold", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %c7 = arith.constant 7 : i32
    %c11 = arith.constant 11 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %byte_offsets = vc4kernel.fragment_shl %lanes, %c2 : vector<16xi32>, i32 -> vector<16xi32>
    %threshold_v = vc4kernel.splat %threshold : i32 -> vector<16xi32>
    %true_v = vc4kernel.splat %c7 : i32 -> vector<16xi32>
    %false_v = vc4kernel.splat %c11 : i32 -> vector<16xi32>
    %ult = vc4kernel.fragment_cmp %lanes, %threshold_v {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %sel_ult = vc4kernel.fragment_select %ult, %true_v, %false_v : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %ne = vc4kernel.fragment_cmp %lanes, %threshold_v {predicate = #vc4kernel.cmp<ne>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %sel_ne = vc4kernel.fragment_select %ne, %sel_ult, %false_v : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %ule = vc4kernel.fragment_cmp %lanes, %threshold_v {predicate = #vc4kernel.cmp<ule>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %sel_ule = vc4kernel.fragment_select %ule, %sel_ne, %true_v : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %sel_ule, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
