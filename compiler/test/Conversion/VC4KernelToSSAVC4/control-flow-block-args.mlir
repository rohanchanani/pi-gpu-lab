// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @control_flow_block_args
// CHECK: ssavc4.cond_br
// CHECK-SAME: ^bb{{[0-9]+}}({{.*}} : i32), ^bb{{[0-9]+}}({{.*}} : i32)
// CHECK: ^{{.*}}(%{{.*}}: i32):
// CHECK: ssavc4.splat
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @control_flow_block_args(%n : i32) attributes {
    public_name = "control_flow_block_args",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "n", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c4 = arith.constant 4 : i32
    %c7 = arith.constant 7 : i32
    %c11 = arith.constant 11 : i32
    %cond = arith.cmpi ult, %n, %c4 : i32
    cf.cond_br %cond, ^merge(%c7 : i32), ^merge(%c11 : i32)
  ^merge(%selected : i32):
    %v = vc4kernel.splat %selected : i32 -> vector<16xi32>
    vc4kernel.return
  }
}
