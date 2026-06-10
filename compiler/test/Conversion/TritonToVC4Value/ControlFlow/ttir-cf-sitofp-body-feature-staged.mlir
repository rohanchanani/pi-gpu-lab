# REQUIRES: vc4-has-triton-cpp-frontend

# RUN: not %vc4_triton_opt "%vc4_repo_root/compiler/test/Conversion/TritonToVC4Value/Inputs/control-flow-sitofp-body-feature.ttir.mlir" --convert-triton-to-vc4-value -o %t.value.mlir 2>&1 | FileCheck %s

# CHECK: arith.sitofp staged by body feature, not unsupported control flow
# CHECK: READY_FOR_TRITON remains NO
