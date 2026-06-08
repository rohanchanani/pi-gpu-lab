# VC4Value Dialect Minimal Specification

## 1. Purpose

`vc4value` is the tiny launch/policy plumbing dialect for the standard MLIR
value surface above VC4Kernel. It fills only the gap that `func`, `vector`,
`memref`, `arith`, `math`, `scf`, and `cf` do not fill: source-level logical
launch-grid identity.

`vc4value` is not a full producer dialect. It has no hardware executable
semantics by itself, does not import Triton, and does not perform lowering.

## 2. Layer boundary

The value stack is:

```text
func + tiny vc4value + vector + memref + arith + math + scf/cf
  -> vc4kernel
  -> ssavc4
  -> scheduled vc4
  -> artifacts/runtime/hardware
```

The value layer must not expose TMU, VDR, VDW, or VPM directly. Those are
VC4Kernel/lower-half planning choices made by later lowering phases.

No vector dialect ops are legal inside verified VC4Kernel, though VC4Kernel may
use `vector<16xT>` carrier types. Producer dialect ops, including `vc4value`
ops, must not appear in verified VC4Kernel hardware fixtures.

## 3. Accepted operations

The accepted `vc4value` operation set is exactly:

```mlir
%pid = vc4value.program_id {axis = 0 : i32} : index
%n = vc4value.num_programs {axis = 0 : i32} : index
```

`vc4value.program_id` returns the logical program id for launch-grid axis 0, 1,
or 2. `vc4value.num_programs` returns the logical launch-grid extent for launch
axis 0, 1, or 2.

These axes are logical launch-grid axes, not physical QPU IDs.

## 4. Operation verifier contract

Both accepted operations:

- are pure;
- take no operands;
- require exactly one `axis` attribute;
- require `axis` to be an i32 integer attribute;
- require `axis` to be exactly 0, 1, or 2;
- return exactly `index`.

The return type is `index` because value-layer IR uses `index` for memref
addressing and shape arithmetic. Lowering to the i32 VC4Kernel launch identity
surface happens later; Phase 2 does not implement that lowering.

## 5. Metadata conventions

These named attributes are Phase 2 conventions only. They are ordinary named
attributes until Phase 3 adds the value-surface verifier.

`vc4value.kernel`
: UnitAttr on `func.func` marking a value-layer kernel candidate.

`vc4value.grid_rank`
: Integer attr, intended valid values 1, 2, or 3 once Phase 3 verifies it.

`vc4value.target_profile`
: StringAttr identifying VC4 target-profile intent, if needed by later phases.

`vc4value.math_policy`
: StringAttr or future policy attr used by later phases; Phase 2 does not
define exact math lowering.

Phase 2 does not verify these attrs. Phase 3 will add the value-surface
verifier.

## 6. Explicitly forbidden operations

The following operations are not part of `vc4value`:

- `vc4value.load`
- `vc4value.store`
- `vc4value.tile`
- `vc4value.vpm`
- `vc4value.tmu`
- `vc4value.vdr`
- `vc4value.vdw`
- `vc4value.fragment`
- `vc4value.lane_id`
- `vc4value.warp_id`
- `vc4value.thread_id`
- `vc4value.block_id`
- `vc4value.barrier`
- `vc4value.semaphore`

The dialect must not allow unknown operations. If any listed operation is ever
introduced, it must be through an explicit future contract change, not by
silently accepting unknown `vc4value.*` syntax.

## 7. Relationship to vector.step and lanes

Lane identity in the value surface is represented by `vector.step` and standard
vector operations, not by `vc4value.lane_id`. `vc4value.program_id` and
`vc4value.num_programs` describe logical launch-grid identity only.

## 8. Relationship to vc4kernel

`vc4value` sits above VC4Kernel. It is not VC4Kernel, it is not a tile DSL, it
does not expose VC4Kernel memory-path operations, and it does not lower directly
to VC4.

Later value-to-VC4Kernel lowering will map logical `vc4value.program_id` and
`vc4value.num_programs` to the locked VC4Kernel launch identity surface. That
lowering does not exist in Phase 2.

## 9. Non-goals before Phase 3/4/5

Before Phase 3/4/5, `vc4value` does not provide:

- value-surface semantic verification;
- kernel ABI lowering;
- memref ABI lowering;
- memory-space attributes such as `#vc4value.global`;
- value-to-VC4Kernel lowering;
- vector transfer lowering;
- TTIR/Triton ingestion;
- hardware fixtures;
- memory, tile, fragment, barrier, lane, warp, thread, or physical QPU ID ops.

## 10. Readiness lines

VC4VALUE_MINIMAL_SPEC_LOCKED=YES
READY_FOR_PHASE2E_VC4VALUE_AUDIT=YES
READY_FOR_VALUE_TO_VC4KERNEL_LOWERING=NO
READY_FOR_TRITON=NO
