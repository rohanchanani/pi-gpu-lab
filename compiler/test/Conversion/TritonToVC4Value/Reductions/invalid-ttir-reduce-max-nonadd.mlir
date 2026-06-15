// REQUIRES: vc4-has-triton-cpp-frontend
// RUN: not %vc4_triton_opt "%vc4_repo_root/examples/triton/phase12_reductions/generated/ttir_reduce_max_reject_b16.ttir.mlir" --convert-triton-to-vc4-value -o %t.mlir 2>&1 | FileCheck %s

// Phase 15 accepts finite f32 max reductions through the controlled
// SFUSoftmax snapshots. This older Phase 12 reject snapshot remains invalid
// because its load uses a nonzero inactive value outside the accepted memory
// profile.
// CHECK: tt.load nonzero other value
// CHECK: READY_FOR_TRITON remains NO
