# VC4 Vector/Triton Phase 6 TTIR Frontend Lock

This document is a source-controlled readiness mirror for Phase 6. It exists so later C++ TTIR frontend setup and implementation prompts can depend on source-controlled phase readiness lines rather than ignored .vc4_auto artifacts.

## Readiness Lines

```text
VC4_TRITON_PHASE6_FRONTEND_LOCKED=YES
VC4_TRITON_PHASE6_PINNED_TRITON_VERSION=3.7.0
VC4_TRITON_PHASE6_REAL_TTIR_CORPUS=YES
VC4_TRITON_PHASE6_TTIR_GENERATION_TOOL=YES
VC4_TRITON_PHASE6_TTIR_PARSE_INVENTORY=YES
VC4_TRITON_PHASE6_IMPORTER_SKELETON=YES
VC4_TRITON_PHASE6_NO_TTIR_TO_VALUE_SEMANTIC_LOWERING=YES
VC4_TRITON_PHASE6_NO_CORE_TRITON_DEPENDENCY=YES
READY_FOR_PHASE7_REAL_TTIR_ELEMENTWISE_SMOKE=YES
READY_FOR_TRITON=NO
```

## Evidence

- `compiler/docs/vc4_vector_triton_phase6_frontend_lock.md` records the Phase 6 TTIR frontend scope, pinned Triton 3.7.0 source of truth, corpus, importer skeleton, and readiness lines.
- `compiler/docs/vc4_vector_triton_phase7_ttir_elementwise_hardware_lock.md` records that Phase 7 consumed the checked-in Phase 6 Triton 3.7.0 TTIR corpus and hardware-proved the narrow TTIR elementwise path after Phase 6.
- `compiler/docs/vc4_ttir_target_profile.md` records the Phase 6 real TTIR frontend lock notes and preserves `READY_FOR_TRITON=NO`.
- `compiler/docs/vc4_vector_triton_phase5_value_elementwise_hardware_lock.md` records the Phase 5 handoff into Phase 6.
- `.vc4_auto/vector_triton_phase6g_final_acceptance/REPORT.md` records `PHASE6_RESULT=LOCKED` with the same readiness lines and no blockers.

## Scope

Phase 6 established the real TTIR frontend boundary and snapshots/importer skeleton. It did not claim broad Triton support; the readiness line remains `READY_FOR_TRITON=NO`.
