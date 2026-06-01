// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @predicate_condition_plan_producers
// CHECK: ssavc4.uniform.read 0 : i32
// CHECK: ssavc4.load_imm
// CHECK: ssavc4.element_number
// CHECK: ssavc4.splat
// CHECK: ssavc4.thread_end
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @predicate_condition_plan_producers(%limit : i32) attributes {
    public_name = "predicate_condition_plan_producers",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "limit", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %empty = vc4kernel.pred.empty : !vc4kernel.pred<16>
    %tail = vc4kernel.pred.tail %c0, %limit : i32, i32 -> !vc4kernel.pred<16>
    %rect = vc4kernel.pred.rect %c0, %c1, %c0, %limit : i32, i32, i32, i32 -> !vc4kernel.pred<16>
    %and = vc4kernel.pred.and %full, %tail : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %or = vc4kernel.pred.or %empty, %rect : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %not = vc4kernel.pred.not %empty : !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %limit_v = vc4kernel.splat %limit : i32 -> vector<16xi32>
    %cmp = vc4kernel.fragment_cmp %lanes, %limit_v {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %any = vc4kernel.pred.any %cmp : !vc4kernel.pred<16> -> i1
    %all = vc4kernel.pred.all %and : !vc4kernel.pred<16> -> i1
    %scalar_cmp = arith.cmpi ult, %c0, %limit : i32
    vc4kernel.return
  }
}
