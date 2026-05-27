// RUN: vc4-opt --legalize-vc4tile-core-cfg %s | FileCheck %s

// CHECK-NOT: scf.
// CHECK-NOT: index
// CHECK: cf.cond_br
// CHECK: cf.br
// CHECK: vc4tile.return
// CHECK-NOT: scf.
// CHECK-NOT: index
vc4tile.kernel @legalize_scf_if_no_result_core attributes {public_name = "legalize_scf_if_no_result_core"} {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %go = arith.cmpi ult, %zero, %one : i32
  scf.if %go {
    %x = arith.addi %zero, %one : i32
  }
  vc4tile.return
}
