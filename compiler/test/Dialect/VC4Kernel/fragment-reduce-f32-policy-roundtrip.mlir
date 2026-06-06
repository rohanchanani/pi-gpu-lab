// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s

module {
  vc4kernel.kernel @fragment_reduce_f32_policy(%x : f32) attributes {
    public_name = "fragment_reduce_f32_policy",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "f32"}],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %v = vc4kernel.splat %x : f32 -> vector<16xf32>
    // CHECK: #vc4kernel.reduce<add>
    %add = vc4kernel.fragment_reduce %v, %full {kind = #vc4kernel.reduce<add>, fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    // CHECK: #vc4kernel.reduce<fmin>
    %min = vc4kernel.fragment_reduce %add, %full {kind = #vc4kernel.reduce<fmin>, fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    // CHECK: #vc4kernel.reduce<fmax>
    %max = vc4kernel.fragment_reduce %min, %full {kind = #vc4kernel.reduce<fmax>, fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    vc4kernel.return
  }
}
