ROADMAP_PHASE9_MULTI_AXIS_ALIGNMENT=YES
PHASE9_FEATURE=VALUE_AND_TTIR_MULTI_AXIS_LAUNCH_IDENTITY
PHASE10_FEATURE=MASK_CLASSIFIER_AND_MEMORY_LEGALITY
PREVIOUS_PHASE9_MASK_DOCS_SUPERSEDED_BY_FEATURE_LADDER=YES
READY_FOR_TRITON=NO

# VC4 Vector/Triton Feature Ladder Plan

This document is the superseding roadmap alignment for the post-Phase 8.5
feature ladder. Historical docs and reports may still describe Phase 9 as mask
classifier and richer memory legality. That text remains historical context,
but the active ladder now inserts multi-axis logical launch identity first.

Phase 8.5 locked TTIR scalar/coherent control flow using controlled real
Triton-emitted fixtures. The top-half robustness lock then hardened the
frontend boundary: unsupported TTIR operations must stage exactly, source names
are metadata only, silent TTIR op drops are rejected, and value output is
verified before entering the lower stack.

The next feature for GEMV, GEMM, and attention-style kernels is multi-axis
logical launch identity. These kernels need 2D or 3D logical grid coordinates
before richer row/column masks and memory legality can be classified in a
natural source-facing way.

## Active Ladder

```text
Phase 8.5  TTIR scalar/coherent control-flow bridge lock
Top-half   Frontend robustness lock
Phase 9.0  Rebaseline and roadmap alignment
Phase 9    VALUE_AND_TTIR_MULTI_AXIS_LAUNCH_IDENTITY
Phase 10   MASK_CLASSIFIER_AND_MEMORY_LEGALITY
```

Phase 9 must not implement the mask classifier, rank-2 memory planning,
gather/scatter legality, strided memory expansion, block pointers, tensor
descriptors, reductions, dot, or native GEMV/GEMM lowering. Those remain
separate feature phases after multi-axis launch identity is proven.

Controlled fixtures, not exploratory probes, define acceptance. Exploratory
inventory may identify real Triton forms and staging requirements, but locked
support must come from checked-in fixtures that lower through:

```text
real Triton source
  -> source-controlled emitted TTIR
  -> VC4Value
  -> vc4kernel
  -> ssavc4
  -> scheduled vc4
  -> hardware when the phase requires hardware proof
```

`READY_FOR_TRITON=NO` remains true. The ladder supports incremental Triton
bridges, not global Triton readiness.
