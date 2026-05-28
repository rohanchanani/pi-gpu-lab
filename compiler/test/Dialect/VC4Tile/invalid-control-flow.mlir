// RUN: not vc4-opt %s 2>&1 | FileCheck %s

vc4tile.kernel @invalid_control_flow attributes {public_name = "invalid_control_flow"} {
  %mask = vc4tile.mask_all
  cf.cond_br %mask, ^then, ^else
^then:
  vc4tile.return
^else:
  vc4tile.return
}

// CHECK: error:
// CHECK: cf.cond_br
