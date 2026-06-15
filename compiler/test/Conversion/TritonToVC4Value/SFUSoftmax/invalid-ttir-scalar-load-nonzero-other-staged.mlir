// REQUIRES: vc4-has-triton-cpp-frontend
// RUN: not %vc4_triton_opt "%vc4_repo_root/examples/triton/phase15_sfu_softmax/generated/ttir_scalar_load_nonzero_other_reject_b16.ttir.mlir" --convert-triton-to-vc4-value -o - 2>&1 | FileCheck %s

// CHECK: tt.load nonzero other value is not currently lowerable by the VC4 TTIR target profile
// CHECK: staged TTIR target-profile feature
// CHECK: READY_FOR_TRITON remains NO
