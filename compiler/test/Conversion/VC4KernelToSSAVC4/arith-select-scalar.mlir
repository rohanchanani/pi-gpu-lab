// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @arith_select_scalar
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_select
// CHECK-SAME: cond = #vc4.cond<cc>
// CHECK: ssavc4.cond_select
// CHECK-SAME: cond = #vc4.cond<cc>
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @arith_select_scalar(%n : i32, %alpha : f32, %beta : f32) attributes {
    public_name = "arith_select_scalar",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "n", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "alpha", kind = "scalar", direction = "by_value", type = "f32"},
      {name = "beta", kind = "scalar", direction = "by_value", type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c16 = arith.constant 16 : i32
    %small = arith.constant 3 : i32
    %large = arith.constant 9 : i32
    %cond = arith.cmpi uge, %n, %c16 : i32
    %selected_i32 = arith.select %cond, %large, %small : i32
    %selected_f32 = arith.select %cond, %alpha, %beta : f32
    %v0 = vc4kernel.splat %selected_i32 : i32 -> vector<16xi32>
    %v1 = vc4kernel.splat %selected_f32 : f32 -> vector<16xf32>
    vc4kernel.return
  }
}
