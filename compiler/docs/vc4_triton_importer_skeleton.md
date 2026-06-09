# VC4 Triton Importer Skeleton

Phase 6 adds `tools/vc4-triton-import` as an importer boundary skeleton only. It is inventory-only because the project has not yet implemented or hardware-proved semantic TTIR-to-value lowering.

READY_FOR_TRITON remains NO.

## Phase 6 Scope

The importer skeleton accepts real `.ttir.mlir` input and parses it through Triton's MLIR parser and dialect registration. After parse succeeds, it can write an inventory JSON summary containing the entry function, operation counts, Phase 7 candidate classification, and unsupported initial feature notes.

It does not:

- lower TTIR to the standard value layer;
- lower TTIR to `vc4kernel`;
- emit `vc4kernel`, `ssavc4`, scheduled `vc4`, TTGIR, PTX, cubin, or hsaco;
- parse TTIR with a regex importer;
- make `check-vc4` depend on Triton, PyTorch, CUDA, HIP, or a GPU.

Text inspection is used only after real Triton parse succeeds, and only for reporting.

## Phase 7 Boundary

Phase 7 will extend this boundary with TTIR elementwise-to-value IR lowering for the locked Phase 7 candidate corpus. The source of truth for that work is emitted TTIR, not Python-level Triton metaprogramming.

The expected future demo path is:

```text
real Triton Python source
  -> real TTIR snapshot
  -> standard value-layer MLIR
  -> vc4kernel
  -> ssavc4
  -> scheduled vc4
  -> artifacts/runtime/hardware
```

Phase 6 intentionally stops at the TTIR inventory boundary.

## CLI

Inventory mode:

```bash
tools/vc4-triton-import examples/triton/phase6/generated/vector_add_b16.ttir.mlir \
  --target cuda:80:32 \
  --mode inventory \
  --summary-json .vc4_auto/vector_add_import_inventory.json
```

Semantic lowering mode is reserved for Phase 7 and fails in Phase 6 with:

```text
TTIR-to-value semantic lowering is not implemented in Phase 6; run Phase 7 after Phase 6 final lock.
```

## Producer IR Policy

TTIR is the first producer IR because it is Triton's frontend MLIR before target-specific GPU lowering. Phase 6 does not ingest TTGIR, `ttnvgpu`, `nvgpu`, `nvvm`, `rocdl`, LLVM IR, PTX, cubin, or hsaco.
