// RUN: not vc4-opt --legalize-vc4tile-core-cfg %s 2>&1 | FileCheck %s

// CHECK: negative

vc4tile.kernel @legalize_scf_for_negative_constant_invalid attributes {public_name = "legalize_scf_for_negative_constant_invalid"} {
  %lb = arith.constant -1 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  scf.for %i = %lb to %ub step %step {
  }
  vc4tile.return
}
