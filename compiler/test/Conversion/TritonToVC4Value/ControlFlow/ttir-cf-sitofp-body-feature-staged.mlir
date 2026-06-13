# REQUIRES: vc4-has-triton-cpp-frontend

# RUN: not %vc4_triton_opt "%vc4_repo_root/compiler/test/Conversion/TritonToVC4Value/Inputs/control-flow-sitofp-body-feature.ttir.mlir" --convert-triton-to-vc4-value -o %t.value.mlir 2>&1 | FileCheck %s

# CHECK: i32 to f32 numeric cast staged by lower-half gap
# CHECK: READY_FOR_TRITON remains NO
