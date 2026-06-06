// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @bad_f32_add_reduce_policy(%x : f32) attributes {
    public_name = "bad_f32_add_reduce_policy",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "f32"}],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %v = vc4kernel.splat %x : f32 -> vector<16xf32>
    // CHECK: f32 fragment_reduce requires finite_tree policy in P4
    %bad = vc4kernel.fragment_reduce %v, %full {kind = #vc4kernel.reduce<add>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    vc4kernel.return
  }

  vc4kernel.kernel @bad_f32_min_reduce_policy(%x : f32) attributes {
    public_name = "bad_f32_min_reduce_policy",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "f32"}],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %v = vc4kernel.splat %x : f32 -> vector<16xf32>
    // CHECK: f32 fragment_reduce requires finite_tree policy in P4
    %bad = vc4kernel.fragment_reduce %v, %full {kind = #vc4kernel.reduce<fmin>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    vc4kernel.return
  }
}
