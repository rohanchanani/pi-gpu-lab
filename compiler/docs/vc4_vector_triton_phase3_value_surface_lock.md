# VC4 Vector/Triton Phase 3 Value Surface Verifier Lock

## 1. Final readiness lines

VC4_VALUE_SURFACE_VERIFIER_LOCKED=YES
VC4_VALUE_SURFACE_SUPPORT_MATRIX_LOCKED=YES
VC4_VALUE_SURFACE_ALLOWED_DIALECTS_AUDITED=YES
VC4_VALUE_SURFACE_FORBIDDEN_DIALECTS_REJECTED=YES
VC4_VALUE_SURFACE_NO_HARDWARE_OPS=YES
READY_FOR_PHASE4_VALUE_ABI=YES
READY_FOR_VALUE_TO_VC4KERNEL_LOWERING=NO
READY_FOR_TRITON=NO

## 2. What Phase 3 locked

Phase 3 locks the standard value-surface verifier contract for:

```text
func + tiny vc4value + vector + memref + arith + math + scf/cf
```

The locked contract verifies value-surface admissibility, not current
lowerability. It establishes the static source boundary that later phases must
pass before ABI work and value-to-VC4Kernel planning.

The allowed dialect family is audited as `builtin`, `func`, `vc4value`,
`vector`, `memref`, `arith`, `math`, `scf`, and `cf`. Producer dialects such as
`tt`, `ttg`, `gpu`, `linalg`, `nvgpu`, `nvvm`, `rocdl`, `spirv`, `iree`,
`stablehlo`, and `mhlo` are rejected at this boundary. Target/lower-half
dialects `vc4kernel`, `ssavc4`, and `vc4` are also rejected in value input.

## 3. What Phase 3 did not implement

Phase 3 is static-only. It added no executable value/Triton semantics and did
not run hardware.

Phase 3 did not implement:

- value-to-VC4Kernel lowering;
- memref ABI lowering;
- kernel launch ABI lowering;
- `vector.transfer_read` or `vector.transfer_write` lowering;
- TTIR or Triton import;
- hardware fixtures or candidate generation;
- new `vc4value` operations.

Valid value-surface IR still cannot be assumed hardware-executable.

## 4. Verifier pass

The locked verifier pass is:

```text
--vc4-verify-value-surface
```

The pass is a `ModuleOp` verifier. It rejects invalid dialect namespaces,
invalid `vc4value.kernel` metadata, `vc4value.program_id` and
`vc4value.num_programs` outside a kernel wrapper, launch axes outside the
declared grid rank, direct memref side-effect operations, sparse-store-shaped
vector operations, unsupported control-flow subset operations, and unsupported
surface types.

The pass does not rewrite IR and does not emit VC4Kernel.

## 5. Support matrix summary

The locked Phase 3 support matrix is:

```text
compiler/docs/vc4_value_surface_support_matrix.json
```

The locked matrix contains 26 rows:

- accepted_surface_contract=9
- staged_future_surface_contract=7
- deterministic_reject_surface_policy=8
- internal_only=2

The matrix `generated_by` value is `Phase3f_value_surface_audit_lock`.

## 6. Diagnostics and deterministic rejects

The diagnostic corpus under `compiler/test/ValueSurface` covers valid value
surface inputs and deterministic rejects for:

- missing or invalid kernel metadata;
- launch identity ops outside `vc4value.kernel`;
- launch axes outside `vc4value.grid_rank`;
- forbidden producer dialects;
- forbidden target/lower-half dialects;
- `vc4value` scope creep;
- direct memref side-effect operations;
- sparse-store-shaped vector operations;
- unsupported value-surface type shapes;
- unsupported Phase 3 control subset operations.

These are source-boundary diagnostics. They are not permanent hardware
impossibility proofs.

## 7. Audit contract

The locked Phase 3 audits are:

```text
compiler/test/ValueSurface/Support/check_vc4_value_surface_matrix.py
compiler/test/ValueSurface/Support/audit_vc4_value_surface.py
```

The matrix checker validates row identity, counts, required fields, staged
phase targets, and deterministic-reject diagnostic policy.

The audit validates matrix consistency, verifier documentation, verifier source
presence, diagnostic coverage, valid-test purity, `vc4value` two-op scope, and
absence of Phase 3 readiness contradictions.

## 8. Phase 4 handoff

Phase 4 is responsible for kernel wrapper, launch ABI, and memref ABI. It must
build on this verifier contract and keep value-to-VC4Kernel lowering disabled.

Phase 5 is the first value-to-VC4Kernel lowering phase. TTIR/Triton ingestion
remains blocked until after handwritten value lowering is proven.

## 9. Still forbidden after Phase 3

The value surface still must not expose TMU, VDR, VDW, or VPM operations
directly. It must not expose `vc4value` memory, tile, fragment, barrier, warp,
thread, lane, or physical QPU identity operations.

Direct Triton/TTIR ingestion remains unavailable. Value-to-VC4Kernel lowering
remains unavailable. VC4Kernel still must not lower directly to scheduled VC4,
and VC4Tile remains retired.
