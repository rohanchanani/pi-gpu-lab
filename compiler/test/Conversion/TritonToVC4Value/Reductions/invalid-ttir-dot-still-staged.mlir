// REQUIRES: vc4-has-triton-cpp-frontend
// RUN: not %vc4_triton_opt "%vc4_repo_root/compiler/test/Conversion/TritonToVC4Value/Inputs/supported-plus-dot.ttir.mlir" --convert-triton-to-vc4-value -o %t.mlir 2>&1 | FileCheck %s

// CHECK: tt.dot / contract
// CHECK: staged
// CHECK: READY_FOR_TRITON remains NO
