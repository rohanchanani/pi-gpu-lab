// RUN: vc4-opt --verify-vc4tile-core %s | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @verify_core_valid_cf
// CHECK: cf.cond_br
// CHECK: ^bb{{[0-9]+}}(%{{.*}}: i32)
vc4tile.kernel @verify_core_valid_cf attributes {public_name = "verify_core_valid_cf"} {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %go = arith.cmpi ult, %zero, %one : i32
  cf.cond_br %go, ^then(%one : i32), ^merge(%zero : i32)
^then(%x: i32):
  cf.br ^merge(%x : i32)
^merge(%y: i32):
  vc4tile.return
}
