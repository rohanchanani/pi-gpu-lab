// RUN: not vc4-opt --legalize-vc4tile-core-cfg %s 2>&1 | FileCheck %s

// CHECK: positive static

vc4tile.kernel @legalize_scf_for_zero_step_invalid attributes {public_name = "legalize_scf_for_zero_step_invalid"} {
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 0 : index
  scf.for %i = %lb to %ub step %step {
  }
  vc4tile.return
}
