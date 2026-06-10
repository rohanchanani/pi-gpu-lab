// REQUIRES: vc4-has-triton-cpp-frontend

// RUN: not %vc4_triton_opt %s --allow-unregistered-dialect --convert-triton-to-vc4-value='test-value-output-boundary=1' -o - 2>&1 | FileCheck %s

// CHECK: forbidden operation in TTIR-to-VC4Value output
// CHECK: stablehlo.fake
// CHECK: dialect 'stablehlo'

module {
  func.func @bad_tensor() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %0 = "stablehlo.fake"() : () -> tensor<16xf32>
    return
  }
}
