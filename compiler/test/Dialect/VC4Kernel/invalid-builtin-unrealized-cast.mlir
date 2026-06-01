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
    // CHECK: builtin dialect operations are forbidden inside vc4kernel.kernel
    %cast = builtin.unrealized_conversion_cast %x : i32 to i32
    vc4kernel.return
  }
}
