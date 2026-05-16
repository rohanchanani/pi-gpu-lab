# VC4 SSAVC4 Milestone 3 Plan

**Document status:** M3 automation and implementation plan for the post-M2, post-structured-`vc4` cleanup repository state.  
**Scope:** define the target-specific `ssavc4` dialect and implement incremental lowering from `ssavc4` to the existing scheduled `vc4` sink.  
**Non-scope:** MLIR `gpu -> ssavc4` lowering, direct `gpu -> vc4` lowering, resurrecting structured `vc4`, replacing the M2 artifact emitter, or changing the M2 runtime/artifact shape.

---

## 1. What M3 is

M3 is the bridge between future frontend lowering and the completed scheduled VC4 backend. It introduces `ssavc4`, a target-specific SSA machine IR for VideoCore IV QPU programs, and lowers supported SSAVC4 kernels into scheduled `vc4.qpu.*` programs that the M2 backend already emits and runs.

The required path is:

```text
ssavc4.module / ssavc4.func
  ↓ vc4-opt --convert-ssavc4-to-vc4
vc4.module / scheduled vc4.func / vc4.qpu.*
  ↓ vc4-codegen --emit-bundle
manifest.json / layout.json / QASM / kernel_launch.c/h / shader arrays
  ↓ libpi-backed runtime
VideoCore IV hardware
```

M3 is accepted only when SSAVC4-input fixtures lower into scheduled VC4, emit M2-compatible artifacts, and run through the existing M2 backend path. Dialect scaffolding alone is not sufficient.

---

## 2. What M3 is not

M3 does not lower MLIR `gpu` dialect. That is M4. M3 must not lower directly from `gpu` to scheduled `vc4`, even as a convenience shortcut.

M3 does not revive the old structured `vc4` operation surface. The active `vc4` dialect remains the scheduled sink with `vc4.module`, `vc4.func`, `vc4.qpu.ldi`, `vc4.qpu.sema`, `vc4.qpu.bundle`, and `vc4.qpu.branch`, plus metadata and live scheduled-QPU attrs/enums.

M3 does not replace `VC4ArtifactEmitter.cpp` or the M2 runtime path. Normal M3 work should not need emitter changes. If an implementation discovers an unavoidable generic emitter support change, it must preserve every M2 verifier check and document why the change is unavoidable.

---

## 3. Why SSAVC4 is separate from scheduled `vc4`

Scheduled `vc4` is QASM-near. It exposes physical read/write register addresses, muxes, ADD/MUL opcodes, pack/unpack encodings, QPU signals, load-immediate modes, branch immediates, delay slots, semaphores, peripheral setup words, and final scheduling hazards. That is the right boundary for artifact emission, but it is too physical for upstream lowering.

SSAVC4 is target-specific but pre-register-allocation and pre-scheduling. It models VC4/QPU semantics with SSA values, explicit effects, tokens, logical launch/resource metadata, and virtual dataflow. The conversion pass owns instruction selection, out-of-SSA, register allocation, scheduling, bundling, hazard insertion, delay-slot handling, and branch layout.

Keeping the dialects separate prevents ambiguity:

```text
ssavc4 = pre-RA/pre-scheduling SSA target machine IR
vc4    = post-RA/post-scheduling artifact sink IR
```

---

## 4. Why old structured `vc4` is not reused

The structured `vc4` surface was intentionally removed after M2. Reusing it would blur the locked scheduled sink boundary and invite old assumptions about `vc4.uniform.*`, `vc4.tmu.*`, `vc4.vpm.*`, `vc4.dma.*`, `vc4.sfu.*`, `vc4.return`, `vc4.program_end`, and `function_form<structured>`.

SSAVC4 is new, explicitly separate, and can define only the minimal descriptor/effect attrs it needs. It may reuse live scheduled-VC4 attrs/enums such as ADD/MUL opcodes and branch conditions, but descriptor concepts removed from `vc4` must be defined in `ssavc4` if M3 needs them.

---

## 5. Core implementation decisions

- Dialect mnemonic: `ssavc4`.
- C++ namespace: `::mlir::ssavc4`.
- Ordinary data types: builtin `i32`, `f32`, `vector<16xi32>`, and `vector<16xf32>`.
- Minimal custom types: `!ssavc4.async.token`, `!ssavc4.tmu.desc`, `!ssavc4.vpm.desc`, and `!ssavc4.flags` as needed.
- Launch/resource metadata: preserve existing `vc4.launch_abi` and `vc4.resource` dictionaries unchanged.
- Pure ops: immediate/value-shape/ALU/pack/unpack/rotate operations only.
- Effectful ops: uniform, TMU, SFU, VPM, VDW, sema, mutex, barrier, and thread termination.
- Flags: restricted SSA pseudo-values, single-use in M3 v1.
- Scheduling v1: conservative one-active-pipe bundles, nops for hazards/delay slots, no spilling, no useful delay-slot filling.

---

## 6. Slice plan

### m3-00-milestone-package

Install the milestone descriptor, worklist, verification spec, context profiles, prompt handoffs, and this plan. This slice must pass immediately after placing the package and must not require compiler source changes.

### m3-01-ssavc4-dialect-scaffold

Create the separate `ssavc4` dialect skeleton, CMake integration, `vc4-opt` registration, and only scaffold smoke tests such as dialect visibility or minimal roundtrip. This slice must keep full `check-vc4` green and must not add active tests for `load_imm`, pure ops, flags, the type model, TMU/VPM/VDW/DMA, lowering, or fixtures.

### m3-02-type-model-and-pure-ops

Add the SSAVC4 type model, minimal custom types/attrs, pure value operations, parser/printer/verifier tests, and invalid tests. These pure-op/type-model tests are owned by this slice and must pass under global `check-vc4`.

### m3-03-lowering-skeleton

Create `--convert-ssavc4-to-vc4`, lower minimal SSAVC4 functions to scheduled `vc4`, copy metadata, and prove lowered output passes scheduled checks.

### m3-04-global-store

Implement `ssavc4.vdw.store` or equivalent minimal VPM→VDW vector store. Verify `vector_store_smoke_ssavc4` through artifact generation and hardware.

### m3-05-branch-tail-lowering

Implement restricted flags, branches, tail control, final-layout branch immediate computation, and explicit three-op delay-slot regions.

### m3-06-tmu-direct-load-saxpy

Implement direct TMU request/read for `tmu0`/`raw32`, preserve `r4` lifetime, and verify `saxpy_full_ssavc4`.

### m3-07-reductions-and-rotate

Implement rotate/pack/unpack lowering and harden liveness/register allocation for `warp_reduce_sum_ssavc4` or an equivalent independent-vector reduction fixture.

### m3-08-cooperative-barrier-basics

Implement sema acquire/release and barrier lowering using cooperative resource metadata and verify a barrier fixture.

### m3-09-shared-vpm-cooperative

Implement the minimal shared VPM read/write subset needed for a cooperative shared-memory fixture such as `shared_transpose_16x16_ssavc4`.

### m3-10-final-acceptance

Run cumulative M3 verification, source-contract audits, full `check-vc4`, and the full generic M2 verifier regression.

---

## 7. Verification plan

M3 uses existing generic verifier mechanisms: `source_products`, `build`, `command`, `manifest_schema`, `program_artifact_bundle`, and `all_qasm_assemble`. New SSAVC4-specific checks are command-based, usually through lit subsets or an SSAVC4 wrapper script that first converts SSAVC4 to scheduled VC4 and then reuses the M2 artifact runner.

After implementation slices, every executable feature must have:

1. Dialect roundtrip tests.
2. Invalid verifier tests.
3. Conversion tests with scheduled-output checks.
4. M2-compatible artifact checks.
5. Fixture/hardware verification when the slice requires hardware.

Final M3 acceptance must run:

```bash
python3 pro_scripts/vc4_milestone_verifier.py verify \
  --repo "$PWD" \
  --milestone-config pro_scripts/milestones/vc4-ssavc4-m3.json \
  --slice m3-00-milestone-package \
  --slice m3-01-ssavc4-dialect-scaffold \
  --slice m3-02-type-model-and-pure-ops \
  --slice m3-03-lowering-skeleton \
  --slice m3-04-global-store \
  --slice m3-05-branch-tail-lowering \
  --slice m3-06-tmu-direct-load-saxpy \
  --slice m3-07-reductions-and-rotate \
  --slice m3-08-cooperative-barrier-basics \
  --slice m3-09-shared-vpm-cooperative \
  --slice m3-10-final-acceptance \
  --out .vc4_auto/ssavc4_m3/manual/m3-final.json \
  --timeout-sec 7200 \
  --keep-going
```

and the generic M2 regression specified in `m3-10-final-acceptance`.

---

## 8. Acceptance criteria

M3 is accepted only when:

- `ssavc4` parses, prints, verifies, and round-trips.
- Pure ops are pure.
- Hardware-state ops are effectful and/or token-ordered.
- `--convert-ssavc4-to-vc4` lowers supported kernels to scheduled `vc4`.
- Scheduled output passes existing scheduled VC4 verifiers.
- SSAVC4 fixtures emit M2-compatible artifacts.
- Global store and TMU/ALU/VDW fixtures run on hardware and pass expected JSON.
- At least one reduction fixture and at least one cooperative fixture are implemented or explicitly scoped by the verification spec.
- No compiler source branches on fixture names.
- No normal lowering path reads physical QPU number for logical identity.
- The full generic M2 verifier passes.

---

## 9. Relationship to M2 and M4

M2 is cumulative and preserved. M3 final acceptance proves this by running the M2 milestone verifier from `m2-00-scaffold` through `m2-10-final-acceptance`.

M4 will lower MLIR `gpu` to SSAVC4. M3 should leave M4 with a stable target-specific SSA dialect, documented launch/resource metadata expectations, and tested SSAVC4-to-scheduled-VC4 lowering.

---

## 10. Generic automation entry point

```bash
python3 pro_scripts/vc4_milestone_resume.py   --repo "$PWD"   --milestone-config pro_scripts/milestones/vc4-ssavc4-m3.json   --gpt-mode current_tab   --timeout-sec 7200
```

The initial package smoke is:

```bash
python3 pro_scripts/vc4_milestone_verifier.py verify   --repo "$PWD"   --milestone-config pro_scripts/milestones/vc4-ssavc4-m3.json   --slice m3-00-milestone-package   --out /tmp/m3-00-package-smoke.json   --timeout-sec 7200   --keep-going
```


The detailed design document for implementation is `compiler/docs/codegen/ssavc4-ir-design-m3-post-cleanup.md`.
