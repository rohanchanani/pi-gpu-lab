// RUN: not %vc4_triton_opt "%vc4_repo_root/examples/triton/phase10_mask_memory/generated/ttir_mask_nonzero_other_reject_b16.ttir.mlir" --convert-triton-to-vc4-value -o %t 2>&1 | FileCheck %s

// CHECK: tt.load nonzero other value
// CHECK: READY_FOR_TRITON remains NO
