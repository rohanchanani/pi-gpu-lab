// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @fragment_cmp_f32_finite_lowering
// CHECK: ssavc4.make_flags {{.*}} {kind = #ssavc4.flag_kind<fsub>}
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<zs>}
// CHECK: ssavc4.make_flags {{.*}} {kind = #ssavc4.flag_kind<fsub>}
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<zc>}
// CHECK: ssavc4.make_flags {{.*}} {kind = #ssavc4.flag_kind<fsub>}
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<ns>}
// CHECK: ssavc4.make_flags {{.*}} {kind = #ssavc4.flag_kind<fsub>}
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<nc>}
// CHECK: ssavc4.make_flags {{.*}} {kind = #ssavc4.flag_kind<fsub>}
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<ns>}
// CHECK: ssavc4.make_flags {{.*}} {kind = #ssavc4.flag_kind<fsub>}
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<nc>}
// CHECK-NOT: vc4kernel.
// CHECK-NOT: vc4.qpu
module {
  vc4kernel.kernel @fragment_cmp_f32_finite_lowering(%out : i32, %lhs : f32, %rhs : f32) attributes {
    public_name = "fragment_cmp_f32_finite_lowering",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "lhs", kind = "scalar", direction = "by_value", type = "f32"},
      {name = "rhs", kind = "scalar", direction = "by_value", type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %offsets = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %lhs_v = vc4kernel.splat %lhs : f32 -> vector<16xf32>
    %rhs_v = vc4kernel.splat %rhs : f32 -> vector<16xf32>
    %one = vc4kernel.fragment_const {value = dense<1> : vector<16xi32>} : vector<16xi32>
    %two = vc4kernel.fragment_const {value = dense<2> : vector<16xi32>} : vector<16xi32>
    %three = vc4kernel.fragment_const {value = dense<3> : vector<16xi32>} : vector<16xi32>
    %four = vc4kernel.fragment_const {value = dense<4> : vector<16xi32>} : vector<16xi32>
    %five = vc4kernel.fragment_const {value = dense<5> : vector<16xi32>} : vector<16xi32>
    %six = vc4kernel.fragment_const {value = dense<6> : vector<16xi32>} : vector<16xi32>
    %zero = vc4kernel.fragment_const {value = dense<0> : vector<16xi32>} : vector<16xi32>
    %oeq = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<oeq>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    %sel_oeq = vc4kernel.fragment_select %oeq, %one, %zero : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %one_pred = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<one>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    %sel_one = vc4kernel.fragment_select %one_pred, %two, %sel_oeq : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %olt = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<olt>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    %sel_olt = vc4kernel.fragment_select %olt, %three, %sel_one : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %oge = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<oge>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    %sel_oge = vc4kernel.fragment_select %oge, %four, %sel_olt : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %ogt = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<ogt>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    %sel_ogt = vc4kernel.fragment_select %ogt, %five, %sel_oge : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %ole = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<ole>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    %sel_ole = vc4kernel.fragment_select %ole, %six, %sel_ogt : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %offsets, %sel_ole, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
