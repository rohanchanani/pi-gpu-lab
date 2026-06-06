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
    // CHECK: 'vc4kernel.fragment_cmp' op operand #0 must be vector<16xi32> or vector<16xf32>, but got 'i32'
    %cmp = "vc4kernel.fragment_cmp"(%x, %x) {predicate = #vc4kernel.cmp<slt>} : (i32, i32) -> !vc4kernel.pred<16>
    vc4kernel.return
  }
}
