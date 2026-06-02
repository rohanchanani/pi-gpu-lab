// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @pred_any_all
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<cs>}
// CHECK: ssavc4.make_flags {{.*}} {kind = #ssavc4.flag_kind<zero_test>}
// CHECK: ssavc4.cond_br {{.*}} {cond = #vc4.branch_cond<any_z_clear>}
// CHECK: ssavc4.make_flags {{.*}} {kind = #ssavc4.flag_kind<zero_test>}
// CHECK: ssavc4.cond_br {{.*}} {cond = #vc4.branch_cond<all_z_clear>}
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_br {{.*}} {cond = #vc4.branch_cond<any_c_set>}
// CHECK: ssavc4.cond_br
// CHECK: ssavc4.cond_br
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @pred_any_all(%threshold : i32) attributes {
    public_name = "pred_any_all",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "threshold", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %threshold_v = vc4kernel.splat %threshold : i32 -> vector<16xi32>
    %cmp = vc4kernel.fragment_cmp %lanes, %threshold_v {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %any_cmp = vc4kernel.pred.any %cmp : !vc4kernel.pred<16> -> i1
    cf.cond_br %any_cmp, ^check_all, ^done
  ^check_all:
    %all_cmp = vc4kernel.pred.all %cmp : !vc4kernel.pred<16> -> i1
    cf.cond_br %all_cmp, ^rect, ^done
  ^rect:
    %rect = vc4kernel.pred.rect %c0, %c1, %c0, %c16 : i32, i32, i32, i32 -> !vc4kernel.pred<16>
    %rect_all = vc4kernel.pred.all %rect : !vc4kernel.pred<16> -> i1
    cf.cond_br %rect_all, ^tail, ^done
  ^tail:
    %tail = vc4kernel.pred.tail %c0, %threshold : i32, i32 -> !vc4kernel.pred<16>
    %tail_any = vc4kernel.pred.any %tail : !vc4kernel.pred<16> -> i1
    cf.cond_br %tail_any, ^done, ^done
  ^done:
    vc4kernel.return
  }
}
