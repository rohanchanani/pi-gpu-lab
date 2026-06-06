// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @bad_reduce_type(%x : i32) attributes {
    public_name = "bad_reduce_type",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    // CHECK: operand #0 must be vector<16xi32> or vector<16xf32>
    %bad = "vc4kernel.fragment_reduce"(%x, %full) {kind = #vc4kernel.reduce<add>} : (i32, !vc4kernel.pred<16>) -> vector<16xi32>
    vc4kernel.return
  }
}
