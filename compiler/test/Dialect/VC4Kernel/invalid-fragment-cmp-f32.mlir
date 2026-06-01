// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s
module {
  vc4kernel.kernel @bad(%x : f32) attributes {
    public_name = "bad",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    %a = vc4kernel.splat %x : f32 -> vector<16xf32>
    // CHECK: fragment_cmp supports only vector<16xi32> operands in v1
    %cmp = vc4kernel.fragment_cmp %a, %a {predicate = #vc4kernel.cmp<ult>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    vc4kernel.return
  }
}
