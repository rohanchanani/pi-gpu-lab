// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @bad(%x : i32) attributes {
    public_name = "bad",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : i32 -> vector<16xi32>
    // CHECK: fp comparison policy is only valid for f32 fragment_cmp
    %cmp = vc4kernel.fragment_cmp %v, %v {predicate = #vc4kernel.cmp<slt>, fp_policy = "finite_only"} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    vc4kernel.return
  }
}
