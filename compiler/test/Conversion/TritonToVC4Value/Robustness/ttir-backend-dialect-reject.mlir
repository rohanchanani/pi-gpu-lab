// REQUIRES: vc4-has-triton-cpp-frontend

// RUN: not %vc4_triton_opt %s --allow-unregistered-dialect --convert-triton-to-vc4-value='test-value-output-boundary=1' -o - 2>&1 | FileCheck %s

// CHECK: forbidden operation in TTIR-to-VC4Value output
// CHECK: ttg.fake_backend
// CHECK: dialect 'ttg'

module {
  func.func @bad_backend() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    "ttg.fake_backend"() : () -> ()
    return
  }
}
