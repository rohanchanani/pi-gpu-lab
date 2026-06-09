# VC4 Vector/Triton Phase 6 TTIR Frontend Lock

## Scope

Phase 6 locks the frontend setup boundary for real Triton TTIR. It covers
pinned Triton setup, real source corpus, generated TTIR snapshots, parse
inventory, and an importer skeleton. It does not claim executable Triton
support.

## Pinned Triton Version

The Phase 6 source-of-truth release is Triton 3.7.0:

```text
triton==3.7.0
source tag v3.7.0
```

The pinned setup and source fallback are documented in
`compiler/docs/vc4_triton_frontend_setup.md`.

## Source Corpus

The source corpus lives under `examples/triton/phase6/kernels/`. Required
Phase 7 elementwise candidates are:

- `vector_add_b16.py`;
- `saxpy_select_b16.py`;
- `i32_add_select_b16.py`.

Future headline source-only/staged families include softmax, matmul/dot, and
layer norm. These keep the roadmap honest but are not Phase 7 lowerable
commitments.

## Generated TTIR Corpus

Generated TTIR snapshots live under `examples/triton/phase6/generated/`.
Metadata lives under `examples/triton/phase6/metadata/`, and the corpus manifest
is `examples/triton/phase6/manifest.json`.

The TTIR snapshots are source-controlled for reproducibility. They are emitted
from real Triton 3.7.0 source by `tools/vc4_emit_ttir.py`. They are TTIR, not
TTGIR, `ttnvgpu`, `nvgpu`, `nvvm`, `rocdl`, LLVM IR, PTX, cubin, or hsaco.

## Importer Skeleton

`tools/vc4-triton-import` is an inventory-only importer skeleton. In Phase 6,
it parses real TTIR through Triton's MLIR parser and can write inventory JSON.
Its `lower-elementwise-v1` mode intentionally fails with the Phase 6 diagnostic:

```text
TTIR-to-value semantic lowering is not implemented in Phase 6; run Phase 7 after Phase 6 final lock.
```

## What Phase 6 Does Not Implement

Phase 6 does not implement semantic TTIR-to-value lowering. It does not lower
TTIR to VC4Kernel, SSAVC4, scheduled VC4, artifacts, runtime, or hardware. It
does not ingest TTGIR or vendor GPU IR. It does not make normal `check-vc4`
depend on Triton, PyTorch, CUDA, HIP, a GPU, or TTIR regeneration.

## Phase 7 Handoff

Phase 7 may extend the importer skeleton for the required elementwise TTIR
forms:

- `tt.func`, `tt.return`;
- axis-0 `tt.get_program_id`;
- `tt.make_range` / arange-derived lane ranges;
- `tt.splat`;
- tensor `arith` operations;
- masked `tt.load` with zero `other`;
- masked `tt.store`.

Future staged forms remain reductions, math/SFU policy, dot/contract, block
pointers and tensor descriptors, atomics, cache/eviction modifiers, and
volatile memory.

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
