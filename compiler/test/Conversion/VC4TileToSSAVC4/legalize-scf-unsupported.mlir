// RUN: not vc4-opt --legalize-vc4tile-core-cfg %s 2>&1 | FileCheck %s

// CHECK: unsupported scf
vc4tile.kernel @legalize_scf_unsupported attributes {public_name = "legalize_scf_unsupported"} {
  %zero = arith.constant 0 : i32
  %result = scf.execute_region -> i32 {
    scf.yield %zero : i32
  }
  %use = arith.addi %result, %zero : i32
  vc4tile.return
}
