// REQUIRES: vc4-has-triton-cpp-frontend

// RUN: not %vc4_triton_opt %s --convert-triton-to-vc4-value='test-value-output-boundary=1' -o - 2>&1 | FileCheck %s

// CHECK: forbidden operation in TTIR-to-VC4Value output
// CHECK: builtin.unrealized_conversion_cast
// CHECK: dialect 'builtin'

module {
  func.func @bad_unrealized_cast(%x: i32) attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %0 = builtin.unrealized_conversion_cast %x : i32 to i32
    return
  }
}
