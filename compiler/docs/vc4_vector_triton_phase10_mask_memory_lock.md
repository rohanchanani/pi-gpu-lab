# VC4 Vector/Triton Phase 10 Mask-Memory Final Lock

Phase 10 locks the value and TTIR mask classifier plus memory legality slice.
The accepted executable subset remains rank-1 flattened memory with i32/f32
`vector<16>` transfers, full/empty/clamped-tail memory masks, compute-mask
selects, zero-other loads with inactive-zero behavior, and inactive-preserve
stores. Sparse transfer masks, nonzero load `other`, rank-2/strided/gather/
block-pointer memory, and unrelated compute features remain staged.

The final TTIR mixed fixture is
`mixed_ttir_mask_memory_cf_axes_b16_vc4triton`. It uses a source-controlled
real Triton source and real emitted TTIR snapshot, enters through the C++
TTIR importer, lowers through the value mask classifier, and runs on hardware
with `active_qpus=12`, strict CPU oracle checks, sentinels, and nonzero output
hash checking.

PHASE10_RESULT=LOCKED
FEATURE=VALUE_AND_TTIR_MASK_CLASSIFIER_MEMORY_LEGALITY
VALUE_MASK_MEMORY_CONTRACT=LOCKED
VALUE_MASK_CLASSIFIER_IMPLEMENTED=YES
VALUE_MEMORY_LEGALITY_CLASSIFIER_IMPLEMENTED=YES
VALUE_MASK_MEMORY_STATIC=PASS
VALUE_MASK_MEMORY_HARDWARE_ISOLATION=PASS
VALUE_MASK_MEMORY_MIXED_ACCEPTANCE=PASS
REAL_TRITON_MASK_MEMORY_SOURCES=YES
REAL_TTIR_MASK_MEMORY_SNAPSHOTS=YES
TTIR_MASK_MEMORY_IMPORTER_STATIC=PASS
TTIR_MASK_MEMORY_HARDWARE_ISOLATION=PASS
TTIR_MASK_MEMORY_MIXED_ACCEPTANCE=PASS
SPARSE_TRANSFER_MASKS_STAGED=YES
NONZERO_LOAD_OTHER_STAGED=YES
RANK2_STRIDED_GATHER_BLOCK_POINTER_STAGED=YES
VALUE_MIXED_REGRESSION=PASS
TTIR_MIXED_REGRESSION=PASS
LAYERED_REGRESSION_POLICY_APPLIED=YES
NO_TEMPORARY_MASK_MEMORY_WORKAROUNDS=YES
READY_FOR_PHASE11_STRIDED_RANKED_MEMORY_SKELETONS=YES
READY_FOR_TRITON=NO
