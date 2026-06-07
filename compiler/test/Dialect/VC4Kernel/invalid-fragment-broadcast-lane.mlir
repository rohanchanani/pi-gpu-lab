// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @bad_fragment_broadcast_lane(%x : i32) attributes {
    public_name = "bad_fragment_broadcast_lane",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : i32 -> vector<16xi32>
    // CHECK: 'vc4kernel.fragment_broadcast_lane' op unknown or forbidden vc4kernel operation
    %bad = "vc4kernel.fragment_broadcast_lane"(%v, %x) : (vector<16xi32>, i32) -> vector<16xi32>
    vc4kernel.return
  }
}
