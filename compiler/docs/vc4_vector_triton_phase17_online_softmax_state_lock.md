PHASE17_RESULT=LOCKED
FEATURE=VALUE_AND_TTIR_ONLINE_MULTIBLOCK_SOFTMAX_STATE
VALUE_ONLINE_SOFTMAX_STATE_CONTRACT=LOCKED
VALUE_ONLINE_SOFTMAX_STATE_STATIC=PASS
VALUE_ONLINE_ATTENTION_APPLY_STATIC=PASS
VALUE_ONLINE_SOFTMAX_STATE_HARDWARE_ISOLATION=PASS
VALUE_ONLINE_ATTENTION_APPLY_HARDWARE_ISOLATION=PASS
VALUE_ONLINE_SOFTMAX_MIXED_ACCEPTANCE=PASS
REAL_TRITON_ONLINE_SOFTMAX_SOURCES=YES
REAL_TTIR_ONLINE_SOFTMAX_SNAPSHOTS=YES
TTIR_ONLINE_SOFTMAX_IMPORTER_STATIC=PASS
TTIR_ONLINE_ATTENTION_APPLY_IMPORTER_STATIC=PASS
TTIR_ONLINE_SOFTMAX_HARDWARE_ISOLATION=PASS
TTIR_ONLINE_ATTENTION_APPLY_HARDWARE_ISOLATION=PASS
TTIR_ONLINE_SOFTMAX_MIXED_ACCEPTANCE=PASS
PRECOMPUTED_SCORES_ONLY=YES
TRANSPOSED_V_LAYOUT_REQUIRED=YES
K_RANGE_ACCEPTED=1_TO_64
K_ZERO_STAGED=YES
NONTRANSPOSED_V_GATHER_STAGED=YES
SCALAR_GLOBAL_LOAD_STAGED=YES
QK_SCORE_GENERATION_STAGED=YES
TL_DOT_TT_DOT_STAGED=YES
BLOCK_POINTERS_STAGED=YES
COOPERATIVE_VPM_TILING_STAGED=YES
FULL_ATTENTION_STAGED=YES
FLASHATTENTION_STAGED=YES
VALUE_MIXED_REGRESSION=PASS
TTIR_MIXED_REGRESSION=PASS
LAYERED_REGRESSION_POLICY_APPLIED=YES
NO_TEMPORARY_ONLINE_SOFTMAX_WORKAROUNDS=YES
READY_FOR_PHASE18_VECTOR_CONTRACT_TL_DOT_TT_DOT=YES
READY_FOR_TRITON=NO

# VC4 Vector/Triton Phase 17 Online Softmax State Lock

Phase 17 locks online/multiblock softmax state for the value and TTIR paths.
The accepted scope is the precomputed-score online recurrence over K blocks of
16 with loop-carried scalar f32 `m/l/acc` state, transposed V layout, natural
`exp` under explicit approximate-SFU policy, weighted reductions, and scalar
result stores.

The value path is locked from the standard value composite through
`vc4kernel`, `ssavc4`, scheduled `vc4`, and real hardware. The TTIR path is
locked from source-controlled real Triton sources and emitted TTIR snapshots
through the C++ importer, the same value lowering path, and real hardware.
Both value and TTIR mixed acceptance suites pass with checked claims,
sentinels, CPU oracles, nonzero output hashes, and `active_qpus=12` where
applicable.

The Phase 17 lock does not implement QK score generation, `tl.dot`, `tt.dot`,
`vector.contract`, scalar global loads, non-transposed V gather, block
pointers, tensor descriptors, cooperative/VPM tiling, full attention, or
FlashAttention. Those remain staged for later phases. K=0 remains staged unless
a future phase introduces and hardware-proves an explicit no-op guard.

`READY_FOR_TRITON` remains `NO`. Phase 18 is now allowed to start the separate
vector.contract / `tl.dot` / `tt.dot` feature increment.
