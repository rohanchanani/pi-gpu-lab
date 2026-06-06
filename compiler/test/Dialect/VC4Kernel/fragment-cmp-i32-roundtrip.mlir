// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s

module {
  vc4kernel.kernel @fragment_cmp_i32_roundtrip(%x : i32, %y : i32) attributes {
    public_name = "fragment_cmp_i32_roundtrip",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "y", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %xv = vc4kernel.splat %x : i32 -> vector<16xi32>
    %yv = vc4kernel.splat %y : i32 -> vector<16xi32>
    // CHECK: #vc4kernel.cmp<eq>
    %eq = vc4kernel.fragment_cmp %xv, %yv {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    // CHECK: #vc4kernel.cmp<ne>
    %ne = vc4kernel.fragment_cmp %xv, %yv {predicate = #vc4kernel.cmp<ne>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    // CHECK: #vc4kernel.cmp<ult>
    %ult = vc4kernel.fragment_cmp %xv, %yv {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    // CHECK: #vc4kernel.cmp<ule>
    %ule = vc4kernel.fragment_cmp %xv, %yv {predicate = #vc4kernel.cmp<ule>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    // CHECK: #vc4kernel.cmp<ugt>
    %ugt = vc4kernel.fragment_cmp %xv, %yv {predicate = #vc4kernel.cmp<ugt>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    // CHECK: #vc4kernel.cmp<uge>
    %uge = vc4kernel.fragment_cmp %xv, %yv {predicate = #vc4kernel.cmp<uge>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    // CHECK: #vc4kernel.cmp<slt>
    %slt = vc4kernel.fragment_cmp %xv, %yv {predicate = #vc4kernel.cmp<slt>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    // CHECK: #vc4kernel.cmp<sle>
    %sle = vc4kernel.fragment_cmp %xv, %yv {predicate = #vc4kernel.cmp<sle>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    // CHECK: #vc4kernel.cmp<sgt>
    %sgt = vc4kernel.fragment_cmp %xv, %yv {predicate = #vc4kernel.cmp<sgt>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    // CHECK: #vc4kernel.cmp<sge>
    %sge = vc4kernel.fragment_cmp %xv, %yv {predicate = #vc4kernel.cmp<sge>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %combined0 = vc4kernel.pred.and %eq, %ne : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %combined1 = vc4kernel.pred.or %ult, %ule : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %combined2 = vc4kernel.pred.and %ugt, %uge : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %combined3 = vc4kernel.pred.or %slt, %sle : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %combined4 = vc4kernel.pred.and %sgt, %sge : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %combined5 = vc4kernel.pred.or %combined0, %combined1 : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %combined6 = vc4kernel.pred.and %combined2, %combined3 : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %combined7 = vc4kernel.pred.or %combined4, %combined5 : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %any = vc4kernel.pred.any %combined6 : !vc4kernel.pred<16> -> i1
    cf.cond_br %any, ^then, ^else
  ^then:
    %all = vc4kernel.pred.all %combined7 : !vc4kernel.pred<16> -> i1
    cf.cond_br %all, ^done, ^done
  ^else:
    vc4kernel.return
  ^done:
    vc4kernel.return
  }
}
