// REQUIRES: vc4-has-triton-cpp-frontend

// RUN: not %vc4_triton_opt %s --allow-unregistered-dialect --convert-triton-to-vc4-value='test-value-output-boundary=1' -o - 2>&1 | FileCheck %s

// CHECK: forbidden operation in TTIR-to-VC4Value output
// CHECK: vc4kernel.fake_lower
// CHECK: dialect 'vc4kernel'

module {
  func.func @bad_lower() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    "vc4kernel.fake_lower"() : () -> ()
    return
  }
}
