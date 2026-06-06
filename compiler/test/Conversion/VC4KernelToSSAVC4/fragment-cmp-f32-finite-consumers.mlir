// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @fragment_cmp_f32_finite_consumers
// CHECK: ssavc4.make_flags {{.*}} {kind = #ssavc4.flag_kind<fsub>}
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<ns>}
// CHECK: ssavc4.make_flags {{.*}} {kind = #ssavc4.flag_kind<zero_test>}
// CHECK: ssavc4.cond_br {{.*}} {cond = #vc4.branch_cond<any_z_clear>}
// CHECK: ssavc4.make_flags {{.*}} {kind = #ssavc4.flag_kind<zero_test>}
// CHECK: ssavc4.cond_br {{.*}} {cond = #vc4.branch_cond<all_z_clear>}
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @fragment_cmp_f32_finite_consumers(%lhs : f32, %rhs : f32) attributes {
    public_name = "fragment_cmp_f32_finite_consumers",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "lhs", kind = "scalar", direction = "by_value", type = "f32"},
      {name = "rhs", kind = "scalar", direction = "by_value", type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    %lhs_v = vc4kernel.splat %lhs : f32 -> vector<16xf32>
    %rhs_v = vc4kernel.splat %rhs : f32 -> vector<16xf32>
    %cmp = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<olt>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    %any = vc4kernel.pred.any %cmp : !vc4kernel.pred<16> -> i1
    cf.cond_br %any, ^check_all, ^done
  ^check_all:
    %all = vc4kernel.pred.all %cmp : !vc4kernel.pred<16> -> i1
    cf.cond_br %all, ^done, ^done
  ^done:
    vc4kernel.return
  }
}
