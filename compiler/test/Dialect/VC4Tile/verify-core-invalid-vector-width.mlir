// RUN: not vc4-opt --verify-vc4tile-core %s 2>&1 | FileCheck %s

// CHECK: vector
// CHECK: 16
vc4tile.kernel @verify_core_invalid_vector_width attributes {public_name = "verify_core_invalid_vector_width"} {
  %bad = arith.constant dense<0> : vector<8xi32>
  vc4tile.return
}
