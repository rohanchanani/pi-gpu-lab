// RUN: rm -f %t.mlir
// RUN: echo 'module {' >> %t.mlir
// RUN: echo '  vc4kernel.kernel @bad attributes {' >> %t.mlir
// RUN: echo '    public_name = "bad",' >> %t.mlir
// RUN: echo '    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,' >> %t.mlir
// RUN: echo '    arg_attrs = [],' >> %t.mlir
// RUN: echo '    warps_per_block = 1 : i32' >> %t.mlir
// RUN: echo '  } {' >> %t.mlir
// RUN: echo '    %%c1 = arith.constant 1 : i32' >> %t.mlir
// RUN: echo '    %%v = vc4kernel.splat %%c1 : i32 -> vector<16xi32>' >> %t.mlir
// RUN: echo '    %%bad = "vc4kernel.fragment_''mul"(%%v, %%v) : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>' >> %t.mlir
// RUN: echo '    vc4kernel.return' >> %t.mlir
// RUN: echo '  }' >> %t.mlir
// RUN: echo '}' >> %t.mlir
// RUN: not vc4-opt %t.mlir --allow-unregistered-dialect --verify-vc4kernel 2>&1 | FileCheck %s

// CHECK: unknown or forbidden vc4kernel operation
