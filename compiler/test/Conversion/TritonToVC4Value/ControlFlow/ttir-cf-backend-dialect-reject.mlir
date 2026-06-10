# REQUIRES: vc4-has-triton-cpp-frontend

# RUN: not %vc4_triton_opt "%vc4_repo_root/compiler/test/TritonFrontend/Inputs/cpp-ttg-backend-reject.mlir" --convert-triton-to-vc4-value -o %t.value.mlir 2>&1 | FileCheck %s

# CHECK: unregistered dialect
