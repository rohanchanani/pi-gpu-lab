// RUN: not vc4-opt --verify-vc4tile-core %s 2>&1 | FileCheck %s

// CHECK: i1
// CHECK: cf.cond_br
vc4tile.kernel @verify_core_invalid_cf_cond_type attributes {public_name = "verify_core_invalid_cf_cond_type"} {
  %bad = arith.constant 0 : i32
  cf.cond_br %bad, ^then, ^else
^then:
  vc4tile.return
^else:
  vc4tile.return
}
