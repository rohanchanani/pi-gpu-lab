// RUN: not vc4-opt %s --allow-unregistered-dialect --verify-vc4kernel 2>&1 | FileCheck %s
module {
  vc4kernel.kernel @bad attributes {
    public_name = "bad",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    // General masks are legal; unknown predicate producers are rejected.
    // CHECK: predicate value must be produced by a known vc4kernel predicate op
    %unknown = "vc4kernel.pred.unknown"() : () -> !vc4kernel.pred<16>
    %any = vc4kernel.pred.any %unknown : !vc4kernel.pred<16> -> i1
    vc4kernel.return
  }
}
