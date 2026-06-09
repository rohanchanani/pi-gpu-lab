# VC4 Triton Importer Boundary

Phase 6 added `tools/vc4-triton-import` as an importer boundary skeleton.
Phase 7 extended that Python tool with a narrow static TTIR-to-value lowering
mode for the real Phase 6 elementwise corpus. Phase 7.5 retires that Python
semantic lowering mode from accepted tests and hardware proof.

The accepted semantic importer is now the optional C++ MLIR tool:

```text
compiler/build-triton-llvm/bin/vc4-triton-opt \
  --convert-triton-to-vc4-value
```

Python remains allowed for Triton source to TTIR snapshot generation and
inventory. The C++ importer is still not full Triton support.

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

Phase 7 extended this boundary with TTIR elementwise-to-value IR lowering for
the locked Phase 7 candidate corpus. Phase 7.5 moves the accepted semantic
path to `vc4-triton-opt --convert-triton-to-vc4-value`. The source of truth is
emitted TTIR, not Python-level Triton metaprogramming.

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

## Phase 7c Static Lowering

The retired Python `--mode lower-elementwise-v1` accepted exactly these real
TTIR snapshots:

```text
examples/triton/phase6/generated/vector_add_b16.ttir.mlir
examples/triton/phase6/generated/saxpy_select_b16.ttir.mlir
examples/triton/phase6/generated/i32_add_select_b16.ttir.mlir
```

The mode first parses TTIR through Triton's MLIR parser and dialect machinery.
Only after parse succeeds does it build a small SSA operation model and check
the Phase 7 V1 elementwise form:

- one public kernel function;
- axis-0 program id;
- `tt.make_range` exactly `0..16`;
- contiguous `base + pid*16 + lane` pointer expressions;
- masked zero-fill loads and canonical tail-mask stores;
- the f32/i32 arithmetic, compare, scalar splat, and select forms used by the
  three Phase 6 elementwise snapshots.

It emits standard value IR only: `func`, tiny `vc4value`, `vector`, `memref`,
and `arith`. It must not emit `vc4kernel`, `ssavc4`, scheduled `vc4`, TTGIR,
vendor GPU dialects, LLVM IR, PTX, cubin, or hsaco.

Unsupported V1-adjacent forms reject with stable diagnostics beginning with:

```text
unsupported TTIR op/form for Phase 7 elementwise V1
```

Staged rejects include reductions, dot/contract, rank-2 tensor/block-pointer
forms, nonzero load `other`, unmasked or noncanonical memory operations, axis
1/2 program ids, `BLOCK_SIZE != 16`, and unsupported element types.

## CLI

Inventory mode:

```bash
tools/vc4-triton-import examples/triton/phase6/generated/vector_add_b16.ttir.mlir \
  --target cuda:80:32 \
  --mode inventory \
  --summary-json .vc4_auto/vector_add_import_inventory.json
```

Semantic lowering mode was reserved in Phase 6 and failed with:

```text
TTIR-to-value semantic lowering is not implemented in Phase 6; run Phase 7 after Phase 6 final lock.
```

In Phase 7c, that mode succeeded for the three elementwise snapshots listed
above and rejected staged forms. In Phase 7.5 and later, accepted tests must
use the C++ `vc4-triton-opt --convert-triton-to-vc4-value` path instead.

## Producer IR Policy

TTIR is the first producer IR because it is Triton's frontend MLIR before target-specific GPU lowering. Phase 6 does not ingest TTGIR, `ttnvgpu`, `nvgpu`, `nvvm`, `rocdl`, LLVM IR, PTX, cubin, or hsaco.
