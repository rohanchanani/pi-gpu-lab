PHASE16_RESULT=LOCKED
FEATURE=VALUE_AND_TTIR_ATTENTION_APPLY_V0
VALUE_ATTENTION_APPLY_V0_CONTRACT=LOCKED
VALUE_ATTENTION_APPLY_V0_STATIC=PASS
VALUE_ATTENTION_APPLY_V0_HARDWARE_ISOLATION=PASS
VALUE_ATTENTION_APPLY_V0_MIXED_ACCEPTANCE=PASS
REAL_TRITON_ATTENTION_APPLY_V0_SOURCES=YES
REAL_TTIR_ATTENTION_APPLY_V0_SNAPSHOTS=YES
TTIR_ATTENTION_APPLY_V0_IMPORTER_STATIC=PASS
TTIR_ATTENTION_APPLY_V0_HARDWARE_ISOLATION=PASS
TTIR_ATTENTION_APPLY_V0_MIXED_ACCEPTANCE=PASS
PRECOMPUTED_SCORES_ONLY=YES
TRANSPOSED_V_LAYOUT_REQUIRED=YES
NONTRANSPOSED_V_GATHER_STAGED=YES
SCALAR_GLOBAL_LOAD_STAGED=YES
K_ZERO_ATTENTION_APPLY_STAGED_OR_GUARD_REQUIRED=YES
MULTIBLOCK_SOFTMAX_STAGED=YES
QK_SCORE_GENERATION_STAGED=YES
TL_DOT_TT_DOT_STAGED=YES
FULL_ATTENTION_STAGED=YES
VALUE_MIXED_REGRESSION=PASS
TTIR_MIXED_REGRESSION=PASS
LAYERED_REGRESSION_POLICY_APPLIED=YES
NO_TEMPORARY_ATTENTION_APPLY_WORKAROUNDS=YES
READY_FOR_PHASE17_ONLINE_MULTIBLOCK_SOFTMAX_STATE=YES
READY_FOR_TRITON=NO

# VC4 Vector/Triton Phase 16 Attention-Apply v0 Lock

Phase 16 locks precomputed-score attention-apply v0 for the value layer and
controlled real TTIR snapshots. The accepted form is intentionally narrow:
row-contiguous score loads, transposed-V row-contiguous loads, stable
natural-exp softmax, weighted add reduction, and scalar result store.

Phase 16 does not implement QK score generation, `tl.dot`, `tt.dot`,
`vector.contract`, non-transposed V gather, scalar global scale loads,
multiblock or online softmax, full attention, or FlashAttention. Those forms
remain staged with explicit reject coverage.

The proof chain includes controlled real Triton sources and TTIR snapshots,
value contract/static lowering, value hardware isolation, value mixed
acceptance, structural C++ TTIR importer lowering, TTIR hardware isolation,
and TTIR mixed acceptance. Both value and TTIR mixed regressions passed under
the layered regression policy.

The natural-exp softmax behavior remains governed by the Phase15B base-2 SFU
semantic repair: public `math.exp`/`tl.exp` are natural exponentiation and lower
through the target base-2 SFU with explicit approximate-SFU policy and finite
tolerance caveats.

`READY_FOR_TRITON` remains `NO`.
