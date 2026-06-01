// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s
module {
  vc4kernel.kernel @fragment_cmp_predicate_consumer(%x : i32) attributes {
    public_name = "fragment_cmp_predicate_consumer",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %xv = vc4kernel.splat %x : i32 -> vector<16xi32>
    // CHECK: vc4kernel.fragment_cmp
    %cmp = vc4kernel.fragment_cmp %lanes, %xv {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    // CHECK: vc4kernel.pred.any
    %any = vc4kernel.pred.any %cmp : !vc4kernel.pred<16> -> i1
    // CHECK: vc4kernel.pred.all
    %all = vc4kernel.pred.all %cmp : !vc4kernel.pred<16> -> i1
    // CHECK: vc4kernel.fragment_select
    %sel = vc4kernel.fragment_select %cmp, %lanes, %xv : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    vc4kernel.return
  }
}
