// RUN: not vc4-opt --verify-vc4tile-core %s 2>&1 | FileCheck %s

// CHECK: vc4tile core
// CHECK: scf
vc4tile.kernel @verify_core_invalid_scf attributes {public_name = "verify_core_invalid_scf"} {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %go = arith.cmpi ult, %zero, %one : i32
  scf.if %go {
    %x = arith.addi %zero, %one : i32
  }
  vc4tile.return
}
