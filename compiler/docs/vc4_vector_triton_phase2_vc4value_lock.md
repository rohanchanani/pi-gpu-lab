# VC4 Vector/Triton Phase 2 VC4Value Lock

## 1. Final result

Phase 2 is locked as a syntax and static-verification phase for the tiny
`vc4value` launch/policy dialect. It added no value-to-VC4Kernel lowering, no
TTIR/Triton import, no hardware fixtures, and no executable hardware semantics.

The locked end state is:

- `vc4value` is registered in `vc4-opt`;
- `vc4value` has exactly two operations, `program_id` and `num_programs`;
- both operations return `index`;
- both operations accept only `axis = 0 : i32`, `axis = 1 : i32`, or
  `axis = 2 : i32`;
- unknown `vc4value.*` operations are rejected;
- the dialect has a source-controlled support matrix and tiny-surface audit.

## 2. Layer boundary

The Phase 2 value stack remains:

```text
func + tiny vc4value + vector + memref + arith + math + scf/cf
  -> vc4kernel
  -> ssavc4
  -> scheduled vc4
  -> artifacts/runtime/hardware
```

`vc4value` sits above VC4Kernel. It is not VC4Kernel, not a tile dialect, not a
memory dialect, and not a lower-half dialect. It must not depend on VC4Kernel,
SSAVC4, or scheduled VC4 headers.

The value surface must continue to keep TMU, VDR, VDW, and VPM out of
producer-facing syntax. Later lowering/planning phases choose those target
paths.

## 3. Accepted vc4value operations

The accepted operation set is exactly:

```mlir
%pid = vc4value.program_id {axis = 0 : i32} : index
%n = vc4value.num_programs {axis = 0 : i32} : index
```

`vc4value.program_id` and `vc4value.num_programs` model logical launch-grid
identity. They are not physical QPU, warp, thread, or lane identity operations.

The result type is `index` because value-layer memref and shape/address
arithmetic use `index`. Conversion to the locked VC4Kernel i32 launch identity
surface is later value-to-VC4Kernel lowering work and is not implemented in
Phase 2.

## 4. Metadata conventions

Phase 2 documents these function metadata conventions as ordinary named attrs:

- `vc4value.kernel`
- `vc4value.grid_rank`
- `vc4value.target_profile`
- `vc4value.math_policy`

Phase 2 does not semantically verify these attrs. Phase 3 should verify attrs
such as `vc4value.kernel` and `vc4value.grid_rank` as part of the value-surface
verifier/audit.

## 5. Explicitly forbidden scope

The following remain outside `vc4value`:

- memory ops such as `vc4value.load` or `vc4value.store`;
- tile ops such as `vc4value.tile`;
- hardware-path ops such as `vc4value.tmu`, `vc4value.vdr`, `vc4value.vdw`, or
  `vc4value.vpm`;
- fragment ops such as `vc4value.fragment`;
- lane/warp/thread/block/QPU identity ops such as `vc4value.lane_id`,
  `vc4value.warp_id`, `vc4value.thread_id`, `vc4value.block_id`, or any
  physical QPU identity op;
- synchronization ops such as `vc4value.barrier` or `vc4value.semaphore`;
- conversion or lowering passes;
- TTIR/Triton ingestion.

Lane identity stays in the standard value surface through `vector.step`, not a
`vc4value.lane_id` operation.

## 6. Tests and audits

Phase 2 added source-path VC4Value lit coverage for:

- dialect registration;
- `program_id` and `num_programs` roundtrip;
- invalid negative and too-large axis values;
- invalid non-index result types;
- ordinary metadata attr roundtrip on `func.func`;
- forbidden `vc4value.*` operation rejection;
- the tiny-surface audit.

The source-controlled audit is:

```text
compiler/test/Dialect/VC4Value/Support/audit_vc4value_tiny_surface.py
```

The source-controlled support matrix is:

```text
compiler/docs/vc4value_support_matrix.json
```

The audit requires:

- `generated_by = Phase2e_vc4value_tiny_surface_lock`;
- allowed ops exactly `program_id,num_programs`;
- `unknown_ops_allowed = false`;
- no `allowUnknownOperations()` in the dialect;
- no forbidden op tokens in parsed VC4Value op mnemonics or op definition
  names;
- no VC4Value dependency on VC4Kernel/SSAVC4/VC4 lower-half headers;
- no VC4Value conversion mention under `compiler/lib/Conversion`.

## 7. Phase 3 handoff

Phase 3 should build the value-surface verifier/audit around:

```text
func + vc4value + vector + memref + arith + math + scf/cf
```

Phase 3 must not implement lowering. It should use `vc4value.program_id` and
`vc4value.num_programs` as the only launch identity ops, verify metadata attrs
such as `vc4value.kernel` and `vc4value.grid_rank`, and continue to keep TMU,
VDR, VDW, and VPM out of the value surface.

Value-to-VC4Kernel lowering remains Phase 5+ work. TTIR/Triton import remains
disabled until the handwritten value path is proven.

## 8. Readiness lines

VC4VALUE_TINY_DIALECT_LOCKED=YES
VC4VALUE_OPS=program_id,num_programs
VC4VALUE_UNKNOWN_OPS_REJECTED=YES
VC4VALUE_NO_MEMORY_TILE_FRAGMENT_OPS=YES
VC4VALUE_NO_LOWERING_IN_PHASE2=YES
VC4VALUE_NO_TRITON_IN_PHASE2=YES
READY_FOR_PHASE3_VALUE_SURFACE_VERIFIER=YES
READY_FOR_VALUE_TO_VC4KERNEL_LOWERING=NO
READY_FOR_TRITON=NO
