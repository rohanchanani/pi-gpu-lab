// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @scalar_cmpi_cond_br
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_br
// CHECK-SAME: cond = #vc4.branch_cond<any_c_set>
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @scalar_cmpi_cond_br(%limit : i32) attributes {
    public_name = "scalar_cmpi_cond_br",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "limit", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c16 = arith.constant 16 : i32
    %cond = arith.cmpi ult, %limit, %c16 : i32
    cf.cond_br %cond, ^small, ^large
  ^small:
    vc4kernel.return
  ^large:
    vc4kernel.return
  }
}
