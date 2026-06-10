PHASE85_STATIC_TTIR_CONTROL_FLOW_LOCK=YES
TTIR_CONTROL_FLOW_FEATURE_DRIVEN_FIXTURES=YES
REAL_TRITON_CF_SOURCES=YES
REAL_TTIR_CF_SNAPSHOTS=YES
TTIR_REGION_LOWERING_FRAMEWORK=YES
TTIR_CF_STATIC_PIPELINE=PASS
ARITH_SITOFP_STAGED_BY_BODY_FEATURE=YES
VECTOR_INDEX_CAST_IMPORTER_BOILERPLATE_REMOVED=YES
FRONTEND_ROBUSTNESS_AUDIT=PASS
NO_TTIR_TOOLCHAIN_REBUILD=YES
READY_FOR_PHASE85R5_HARDWARE_ISOLATION=YES
TTIR_CF_ISOLATION_HARDWARE=PASS
TTIR_CF_SCALAR_IF_HARDWARE=PASS
TTIR_CF_TL_RANGE_HARDWARE=PASS
TTIR_CF_WHILE_HARDWARE=PASS
TTIR_CF_PERSISTENT_LOOP_HARDWARE=PASS
READY_FOR_PHASE85R6_MIXED_ACCEPTANCE=YES
TTIR_CF_MIXED_ACCEPTANCE=PASS
TTIR_MIXED_REGRESSION=PASS
LAYERED_REGRESSION_POLICY_APPLIED=YES
PHASE85_NO_TTIR_REGEN_IN_HARDWARE_PHASES=YES
READY_FOR_PHASE85R7_FINAL_LOCK=YES
READY_FOR_TRITON=NO

# VC4 Vector/Triton Phase 8.5 TTIR Control-Flow Static Lock

Phase 8.5R4 locks the static TTIR control-flow contract before hardware. The
accepted feature is real Triton-emitted scalar/coherent TTIR control flow using
only already-supported body features.

## Fixture Contract

The acceptance-driving fixtures are source-controlled real Triton sources and
real emitted TTIR snapshots:

```text
examples/triton/phase8_5_control_flow/sources/
examples/triton/phase8_5_control_flow/generated/
examples/triton/phase8_5_control_flow/manifest.json
```

The manifest records the source file, generated TTIR file, kernel name,
signature, generator command, and Triton version. Implementation and hardware
phases must consume these checked-in snapshots and must not regenerate TTIR as
part of ordinary acceptance.

Generic Phase85A1 probes remain inventory-only evidence. They are not the
acceptance fixture set, especially where they contain staged body features such
as `arith.sitofp`.

## Accepted Static Forms

The controlled fixtures cover:

- scalar runtime `if` emitted as `scf.if`;
- runtime `tl.range` emitted as `scf.for`;
- scalar coherent `while` emitted as `scf.while`;
- persistent pid-strided loop skeleton using `tt.get_program_id` and
  axis-0 `tt.get_num_programs`;
- a mixed tail/load/store/control-flow fixture combining accepted elementwise
  body features with scalar coherent control flow.

The optional static-range fixture is an import regression for specialized or
unrolled TTIR and is not a runtime control-flow hardware fixture.

Accepted fixture bodies exclude `arith.sitofp`, fptosi, dot, reduce, gather,
scatter, block pointers, tensor descriptors, atomics, subword features, SFU or
math-expansion features, TTGIR, TritonGPU, NVGPU, NVVM, and GPU backend IR.

## Importer Contract

The Phase 8.5 importer lowers structurally:

```text
tt.func
  -> func.func + vc4value metadata
scf.for / scf.if / scf.while
  -> value-layer scf regions with scoped SSA mappings
```

The importer emits value-layer IR only. It does not emit or preserve raw TTIR,
backend dialects, `vc4kernel`, `ssavc4`, or scheduled `vc4` operations.

Canonical `tt.make_range 0..16` pointer and tail-mask expressions are consumed
structurally, so the importer does not emit dead common-prefix
`vector<16xindex>` to `vector<16xi32>` boilerplate. True data uses of
`tt.make_range` remain separately classified by the importer contract.

`arith.sitofp` remains staged by body feature. Numeric cast support is not part
of the TTIR control-flow feature.

## Static Pipeline Lock

The accepted controlled snapshots statically pass:

```text
vc4-triton-opt --convert-triton-to-vc4-value
vc4-opt --vc4-verify-value-surface
vc4-opt --convert-scf-to-cf --convert-vc4-value-to-vc4kernel --verify-vc4kernel
vc4-opt --convert-vc4kernel-to-ssavc4
vc4-opt --convert-ssavc4-to-vc4
```

Boundary audits confirm:

- value output contains no TTIR/backend/lower-half operations;
- VC4Kernel output contains no producer operations;
- SSAVC4 output contains no producer or VC4Kernel operations;
- scheduled VC4 output contains no producer, VC4Kernel, or SSAVC4 operations.

## Hardware Isolation Lock

Phase 8.5R5 adds targeted TTIR hardware isolation fixtures under:

```text
compiler/test/CodeGen/Triton/Hardware/Run/
```

The accepted hardware fixtures consume the source-controlled R2 TTIR snapshots
as `input.ttir.mlir` and do not regenerate TTIR:

- `ttir_cf_scalar_if_b16_vc4triton`;
- `ttir_cf_tl_range_loop_b16_vc4triton`;
- `ttir_cf_while_loop_b16_vc4triton`;
- `ttir_cf_persistent_loop_b16_vc4triton`.

Each fixture runs with `active_qpus=12`, checks a CPU oracle, verifies guard
sentinels, requires an exact nonzero output hash, and records result fields for
the real TTIR snapshot, C++ TTIR importer path, value SCF/CF path,
value-to-VC4Kernel lowering, and the fixture-specific control-flow form.

The R5 claim audit checks exact snapshot provenance, expected TTIR control-flow
operations, no accepted-path `arith.sitofp`, and required hardware result
fields.

## Mixed Acceptance Lock

Phase 8.5R6 adds `mixed_ttir_cf_loop_if_tail_vc4triton` to the TTIR mixed
acceptance suite. The fixture consumes the source-controlled
`mixed_ttir_cf_loop_if_tail_b16.ttir.mlir` snapshot and combines the supported
TTIR feature set so far:

- C++ TTIR importer path;
- axis-0 `tt.get_program_id`;
- `tt.make_range` 0..16 / `BLOCK_SIZE=16`;
- tail mask with masked `tt.load` / `tt.store`;
- f32 add/sub/mul/cmp/select;
- scalar `scf.if`;
- scalar coherent `scf.while`;
- strict CPU oracle, guard sentinels, and locked nonzero output hash;
- `active_qpus=12`.

The full TTIR mixed suite passed on real VC4 hardware. Layered regression policy
classified the lowest touched semantic layer as `TTIR_IMPORTER`, so value and
VC4Kernel mixed suites were not required for R6.
