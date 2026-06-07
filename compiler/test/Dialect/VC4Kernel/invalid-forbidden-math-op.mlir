// RUN: not vc4-opt %s --verify-vc4kernel --allow-unregistered-dialect 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @bad_math(%x : f32) attributes {
    public_name = "bad_math",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "f32"}],
    warps_per_block = 1 : i32
  } {
    // CHECK: dialect 'math' is forbidden inside vc4kernel
    %y = "math.sqrt"(%x) : (f32) -> f32
    vc4kernel.return
  }
}
