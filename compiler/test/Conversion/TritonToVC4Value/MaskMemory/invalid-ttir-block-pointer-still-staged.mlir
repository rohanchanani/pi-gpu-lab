// RUN: not %vc4_triton_opt "%vc4_repo_root/compiler/test/Conversion/TritonToVC4Value/Inputs/block-pointer-staged.ttir.mlir" --convert-triton-to-vc4-value -o %t 2>&1 | FileCheck %s

// CHECK: block pointer or tensor pointer argument
// CHECK: READY_FOR_TRITON remains NO
