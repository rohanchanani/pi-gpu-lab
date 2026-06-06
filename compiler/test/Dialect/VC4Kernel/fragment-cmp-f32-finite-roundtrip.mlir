// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s

module {
  vc4kernel.kernel @fragment_cmp_f32_finite_roundtrip(%x : f32, %y : f32) attributes {
    public_name = "fragment_cmp_f32_finite_roundtrip",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "f32"},
      {name = "y", kind = "scalar", direction = "by_value", type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    %xv = vc4kernel.splat %x : f32 -> vector<16xf32>
    %yv = vc4kernel.splat %y : f32 -> vector<16xf32>
    // CHECK: #vc4kernel.fp_cmp_policy<finite_only>
    // CHECK-SAME: #vc4kernel.cmp<oeq>
    %oeq = vc4kernel.fragment_cmp %xv, %yv {predicate = #vc4kernel.cmp<oeq>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    // CHECK: #vc4kernel.fp_cmp_policy<finite_only>
    // CHECK-SAME: #vc4kernel.cmp<one>
    %one = vc4kernel.fragment_cmp %xv, %yv {predicate = #vc4kernel.cmp<one>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    // CHECK: #vc4kernel.fp_cmp_policy<finite_only>
    // CHECK-SAME: #vc4kernel.cmp<olt>
    %olt = vc4kernel.fragment_cmp %xv, %yv {predicate = #vc4kernel.cmp<olt>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    // CHECK: #vc4kernel.fp_cmp_policy<finite_only>
    // CHECK-SAME: #vc4kernel.cmp<ole>
    %ole = vc4kernel.fragment_cmp %xv, %yv {predicate = #vc4kernel.cmp<ole>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    // CHECK: #vc4kernel.fp_cmp_policy<finite_only>
    // CHECK-SAME: #vc4kernel.cmp<ogt>
    %ogt = vc4kernel.fragment_cmp %xv, %yv {predicate = #vc4kernel.cmp<ogt>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    // CHECK: #vc4kernel.fp_cmp_policy<finite_only>
    // CHECK-SAME: #vc4kernel.cmp<oge>
    %oge = vc4kernel.fragment_cmp %xv, %yv {predicate = #vc4kernel.cmp<oge>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    %any0 = vc4kernel.pred.any %oeq : !vc4kernel.pred<16> -> i1
    cf.cond_br %any0, ^next, ^done
  ^next:
    %combined0 = vc4kernel.pred.or %one, %olt : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %combined1 = vc4kernel.pred.and %ole, %ogt : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %combined2 = vc4kernel.pred.or %combined0, %combined1 : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %combined3 = vc4kernel.pred.and %combined2, %oge : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %all = vc4kernel.pred.all %combined3 : !vc4kernel.pred<16> -> i1
    cf.cond_br %all, ^done, ^done
  ^done:
    vc4kernel.return
  }
}
