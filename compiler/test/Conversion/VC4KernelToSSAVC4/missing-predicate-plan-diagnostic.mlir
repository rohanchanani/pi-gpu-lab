// RUN: not vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @missing_condition_plan attributes {
    public_name = "missing_condition_plan",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    vc4kernel.return
  ^dead(%cond : i1):
    // CHECK: condition operand has no lowering plan
    cf.cond_br %cond, ^then, ^else
  ^then:
    vc4kernel.return
  ^else:
    vc4kernel.return
  }
}
