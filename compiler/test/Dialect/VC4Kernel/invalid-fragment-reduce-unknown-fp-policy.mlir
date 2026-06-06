// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @bad_unknown_fp_reduce_policy(%x : f32) attributes {
    public_name = "bad_unknown_fp_reduce_policy",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "f32"}],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %v = vc4kernel.splat %x : f32 -> vector<16xf32>
    // CHECK: expected ::mlir::vc4kernel::FPReducePolicy to be one of: finite_tree
    %bad = vc4kernel.fragment_reduce %v, %full {kind = #vc4kernel.reduce<add>, fp_policy = #vc4kernel.fp_reduce_policy<ieee>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    vc4kernel.return
  }
}
