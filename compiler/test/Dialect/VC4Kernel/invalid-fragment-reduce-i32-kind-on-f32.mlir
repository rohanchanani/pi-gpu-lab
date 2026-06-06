// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @bad_i32_kind_on_f32(%x : f32) attributes {
    public_name = "bad_i32_kind_on_f32",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "f32"}],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %v = vc4kernel.splat %x : f32 -> vector<16xf32>
    // CHECK: integer reduce kind requires vector<16xi32>
    %bad = vc4kernel.fragment_reduce %v, %full {kind = #vc4kernel.reduce<min_s>, fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    vc4kernel.return
  }
}
