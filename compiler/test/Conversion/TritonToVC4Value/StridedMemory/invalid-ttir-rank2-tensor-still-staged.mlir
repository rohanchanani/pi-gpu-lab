// RUN: not %vc4_triton_opt "%vc4_repo_root/compiler/test/Conversion/TritonToVC4Value/Inputs/rank2-pointer-tensor-staged.ttir.mlir" --convert-triton-to-vc4-value -o %t 2>&1 | FileCheck %s

// CHECK: rank-2 or higher TTIR tensor result
// CHECK: READY_FOR_TRITON remains NO
