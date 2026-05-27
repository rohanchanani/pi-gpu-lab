// RUN: not vc4-opt --verify-vc4tile-core %s 2>&1 | FileCheck %s

// CHECK: index
vc4tile.kernel @verify_core_invalid_index attributes {public_name = "verify_core_invalid_index"} {
  %idx = arith.constant 0 : index
  vc4tile.return
}
