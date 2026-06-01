// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s
module {
  vc4kernel.kernel @ok attributes {
    public_name = "ok",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    vc4kernel.return
  }

  // CHECK: only vc4kernel.kernel operations may appear at module top level
  vc4.func private @host() attributes {
    domain = #vc4.execution_domain<qpu>,
    form = #vc4.function_form<scheduled>,
    threading = #vc4.threading_mode<single>
  }
}
