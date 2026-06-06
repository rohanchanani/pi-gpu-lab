// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @bad_f32_kind_on_i32(%x : i32) attributes {
    public_name = "bad_f32_kind_on_i32",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %v = vc4kernel.splat %x : i32 -> vector<16xi32>
    // CHECK: f32-only reduce kind requires vector<16xf32>
    %bad = vc4kernel.fragment_reduce %v, %full {kind = #vc4kernel.reduce<fmin>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.return
  }
}
