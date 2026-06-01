// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s
module {
  vc4kernel.kernel @preds(%base : i32, %limit : i32) attributes {
    public_name = "preds",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "base", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "limit", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %xv = vc4kernel.splat %base : i32 -> vector<16xi32>
    // CHECK: vc4kernel.pred.full
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    // CHECK: vc4kernel.pred.empty
    %empty = vc4kernel.pred.empty : !vc4kernel.pred<16>
    // CHECK: vc4kernel.pred.tail
    %tail = vc4kernel.pred.tail %base, %limit : i32, i32 -> !vc4kernel.pred<16>
    // CHECK: vc4kernel.pred.rect
    %rect = vc4kernel.pred.rect %base, %limit, %base, %limit : i32, i32, i32, i32 -> !vc4kernel.pred<16>
    // CHECK: vc4kernel.pred.and
    %and = vc4kernel.pred.and %full, %tail : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    // CHECK: vc4kernel.pred.or
    %or = vc4kernel.pred.or %empty, %tail : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    // CHECK: vc4kernel.pred.not
    %not_full = vc4kernel.pred.not %full : !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    // CHECK: vc4kernel.pred.not
    %not_empty = vc4kernel.pred.not %empty : !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    // CHECK: vc4kernel.pred.or
    %general_from_or = vc4kernel.pred.or %tail, %rect : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    // CHECK: vc4kernel.pred.not
    %general_from_not = vc4kernel.pred.not %tail : !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    // CHECK: vc4kernel.fragment_cmp
    %cmp = vc4kernel.fragment_cmp %lanes, %xv {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    // CHECK: vc4kernel.pred.any
    %any = vc4kernel.pred.any %cmp : !vc4kernel.pred<16> -> i1
    // CHECK: vc4kernel.pred.all
    %all = vc4kernel.pred.all %general_from_or : !vc4kernel.pred<16> -> i1
    vc4kernel.return
  }
}
