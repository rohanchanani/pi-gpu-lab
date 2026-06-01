// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s
module {
  vc4kernel.kernel @bad(%x : i32) attributes {
    public_name = "bad", schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    // CHECK: vector dialect operations are forbidden
    %v = vector.broadcast %x : i32 to vector<16xi32>
    vc4kernel.return
  }
}
