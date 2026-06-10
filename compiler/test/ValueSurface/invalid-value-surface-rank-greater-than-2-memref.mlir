// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

builtin.module {
  // expected-error @+2 {{memref rank greater than 2 is not legal in the VC4 value surface}}
  // expected-error @+1 {{public value kernel @rank3_memref argument #0 'x': public memref argument rank must be 1 or 2}}
  func.func @rank3_memref(%x: memref<1x1x1xf32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    func.return
  }
}
