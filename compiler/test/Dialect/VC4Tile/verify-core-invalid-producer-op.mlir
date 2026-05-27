// RUN: not vc4-opt --allow-unregistered-dialect --verify-vc4tile-core %s 2>&1 | FileCheck %s

// CHECK: producer
vc4tile.kernel @verify_core_invalid_producer_op attributes {public_name = "verify_core_invalid_producer_op"} {
  "gpu.fake_op"() : () -> ()
  vc4tile.return
}
