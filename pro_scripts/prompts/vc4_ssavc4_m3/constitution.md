# VC4 SSAVC4 M3 Constitution

You are working on Milestone 3 of the Raspberry Pi VideoCore IV MLIR backend. M3 defines the new `ssavc4` dialect and lowers `ssavc4` to the existing scheduled `vc4` dialect. M3 does not lower MLIR `gpu`, Triton, IREE, or any frontend producer IR. M4 is the VC4 Tile dialect (`vc4tile`) and `vc4tile -> ssavc4` lowering; producer lowering into `vc4tile` is later work.

The scheduled `vc4` dialect is the locked M2 backend sink. It contains `vc4.module`, `vc4.func`, `vc4.qpu.ldi`, `vc4.qpu.sema`, `vc4.qpu.bundle`, and `vc4.qpu.branch`, plus launch/resource metadata and live scheduled-QPU attrs/enums. Do not resurrect removed structured `vc4` operations such as `vc4.uniform.*`, `vc4.tmu.*`, `vc4.vpm.*`, `vc4.dma.*`, `vc4.sfu.*`, `vc4.return`, `vc4.program_end`, `vc4.async.wait`, `vc4.enqueue_qpu`, `vc4.reserve_qpu`, `vc4.v3d.*`, or `function_form<structured>`.

`ssavc4` is target-specific machine SSA for VC4/QPU semantics. It is pre-register-allocation and pre-scheduling. It should expose SSA values, explicit effects/tokens, logical launch/resource metadata, and low-level VC4 hardware concepts without exposing final physical register addresses, final instruction order, branch immediates, or branch delay-slot layout to users.

The conversion `ssavc4 -> scheduled vc4` must perform instruction selection, out-of-SSA, register allocation with spill-frame support, scheduling, bundling, hazard insertion, branch layout, and metadata preservation. Correctness beats optimality.

M3 originally planned a conservative no-spill allocator, but the current SSAVC4 lower half now includes spilling support and block-argument/edge-copy lowering. Future milestones should treat spilling and SSAVC4 block arguments as existing lower-half capabilities, while still generating low-pressure IR where practical. Keep internal seams for instruction selection/templates, virtual values, liveness, allocation, scheduling, hazard insertion, and branch layout. Do not expose public `ssavc4.push`/`ssavc4.pop` or stack operations. Spilling is a lowering-private allocator feature using private per-logical-request spill frames in global GPU memory allocated from the existing VC4 program heap, with shared/VPM spilling deferred as a later optimization.

M3 implementation must be vertical after the early scaffold slices. When a slice adds an executable feature, it must add dialect ops/types/attrs, verifiers/effects, lowering, scheduled-output checks, artifact checks, and fixture verification for that feature in the same slice or the immediately adjacent slice.

M3 regression is cumulative. Each committed slice must keep `ninja -C compiler/build check-vc4` green. Future-slice SSAVC4 tests must not be checked into active lit paths before their owning implementation slice, and expected-red tests must not live under global `check-vc4`.

Non-negotiable bans:

- Do not special-case fixture names or public names.
- Do not fake `VC4_TEST_RESULT` or edit expected JSON to match broken output.
- Do not bypass `ssavc4 -> scheduled vc4` lowering by copying reference QASM into generated outputs.
- Do not mutate M2 reference bundles, `generated_examples`, existing VC4 fixture reference trees, or existing VC4 expected JSON.
- Do not weaken M2 scheduled sink verifiers, artifact contracts, runtime support checks, or generic milestone drivers.
- Do not require `vc4-codegen` to consume SSAVC4 directly unless a later implementation explicitly adds a wrapper; the clean path is `vc4-opt --convert-ssavc4-to-vc4` followed by existing `vc4-codegen`.
- Do not use physical QPU number as normal logical identity. Use launch ABI uniforms/runtime logical request/block/warp identity.

When a gate fails, fix the narrow contract violation. Do not broaden scope, hide the failure, remove the check, or route around the scheduled sink.
