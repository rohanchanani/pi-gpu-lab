// RUN: not %vc4_triton_opt "%vc4_repo_root/examples/triton/phase10_mask_memory/generated/ttir_mask_sparse_store_reject_b16.ttir.mlir" --convert-triton-to-vc4-value -o %t 2>&1 | FileCheck %s

// CHECK: sparse or unknown tt.store memory mask
// CHECK: READY_FOR_TRITON remains NO
