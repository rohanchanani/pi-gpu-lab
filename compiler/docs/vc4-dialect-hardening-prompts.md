# VC4 dialect hardening prompt sequence

This sequence assumes the repository is at the current post-dialect state reflected in these files:

- `include/vc4/Dialect/VC4/IR/VC4Attrs.h`
- `include/vc4/Dialect/VC4/IR/VC4Attrs.td`
- `include/vc4/Dialect/VC4/IR/VC4Base.td`
- `include/vc4/Dialect/VC4/IR/VC4Dialect.h`
- `include/vc4/Dialect/VC4/IR/VC4Dialect.td`
- `include/vc4/Dialect/VC4/IR/VC4Enums.h`
- `include/vc4/Dialect/VC4/IR/VC4Enums.td`
- `include/vc4/Dialect/VC4/IR/VC4OpEnums.td`
- `include/vc4/Dialect/VC4/IR/VC4Ops.h`
- `include/vc4/Dialect/VC4/IR/VC4SideEffects.h`
- `include/vc4/Dialect/VC4/IR/VC4StructuredOps.td`
- `include/vc4/Dialect/VC4/IR/VC4Types.h`
- `include/vc4/Dialect/VC4/IR/VC4Types.td`
- `lib/Dialect/VC4/IR/VC4Dialect.cpp`
- `lib/Dialect/VC4/IR/VC4Enums.cpp`
- `lib/Dialect/VC4/IR/VC4Ops.cpp`
- `lib/Dialect/VC4/IR/VC4Types.cpp`
- `tools/vc4-opt/vc4-opt.cpp`
- `test/Dialect/VC4/*.mlir`
- `test/lit.cfg.py`

This hardening plan is driven by the Broadcom VideoCore IV QPU encoding tables, branch conditions, signal values, pack/unpack tables, register-address map, TMU/VPM/DMA interface rules, thread/program-end restrictions, and instruction-sequence restrictions in the hardware guide. fileciteturn4file0

## Operating rules for this sequence

1. **Commit after every successful verification prompt.**
2. **Do not rewrite already-working infrastructure unless the prompt explicitly asks for it.** In particular, do not casually rewrite `VC4FuncOp::parse`, `VC4FuncOp::print`, `vc4.module`, `vc4.func`, `vc4.return`, or `vc4.builtin` handling unless a prompt explicitly requires it.
3. **Do not implement lowering, codegen, qasm emission, launcher generation, or runtime integration in this sequence.** This is a dialect-hardening milestone only.
4. **Keep the fragment/TLB/stencil/scoreboard world out of scope.** Do not reintroduce fragment-specific structured ops or tile-buffer-specific semantics.
5. **For risky parser/textual changes, prefer incremental changes and targeted tests.**
6. **When a prompt says not to touch specific files or areas, obey it literally.**
7. **All verification prompts are written to run in a cold, no-context session.** They should inspect the files on disk and repair mismatches directly.

---

## Prompt H1 — hardware-backed enum/encoding audit and cleanup

```text
You are working in the root of the `compiler` repository.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- include/vc4/Dialect/VC4/IR/VC4Enums.td
- include/vc4/Dialect/VC4/IR/VC4Attrs.td
- include/vc4/Dialect/VC4/IR/VC4OpEnums.td
- include/vc4/Dialect/VC4/IR/VC4StructuredOps.td
- lib/Dialect/VC4/IR/VC4Ops.cpp
- compiler/test/Dialect/VC4/enum-attr-roundtrip.mlir
- compiler/test/Dialect/VC4/qpu-bundle-roundtrip.mlir
- compiler/test/Dialect/VC4/qpu-bundle-invalid.mlir
- compiler/test/Dialect/VC4/qpu-branch-roundtrip.mlir
- compiler/test/Dialect/VC4/qpu-ldi-sema-roundtrip.mlir
- compiler/test/Dialect/VC4/sync-thread-roundtrip.mlir

Task:
Audit and fix the hardware-backed enum/attribute surface so that the VC4 sink-facing enums and enum attrs match the actual VideoCore IV hardware encodings and naming intent.

This prompt is the first prompt in a cold session. Be methodical. Do not make speculative parser rewrites.

Required outcomes:
1. In `include/vc4/Dialect/VC4/IR/VC4Enums.td`, audit and correct the hardware-backed enums that are used by the sink-facing and hardware-facing IR.
2. At minimum, the following enums must be checked and fixed if necessary:
   - `VC4_AddOpcode`
   - `VC4_MulOpcode`
   - `VC4_Cond`
   - `VC4_LoadImmMode`
   - `VC4_RegfileAUnpackMode`
   - `VC4_RegfileAPackMode`
   - `VC4_R4UnpackMode`
   - `VC4_MulPackMode`
   - `VC4_QPUSignal`
   - `VC4_QPUMux`
   - `VC4_BranchCond`
3. Use the real hardware numeric values from the Broadcom tables. Do not use dense placeholder numbering for hardware encodings.
4. Be especially careful about the following concrete points:
   - `AddOpcode` values such as `add = 12`, `sub = 13`, `clz = 24`, `v8adds = 30`, `v8subs = 31` must remain hardware-accurate.
   - `LoadImmMode` must reflect the actual load-immediate encodings used by the hardware instruction forms.
   - `QPUSignal` must preserve actual signal numbers. If fragment-only/tile-buffer-only signals remain intentionally omitted from the dialect scope, do **not** renumber the remaining members.
   - The hardware signal value for “last thread switch” must be represented explicitly in `VC4_QPUSignal` with its correct hardware value.
   - `BranchCond` must use the real branch condition values, including unconditional/always.
5. Keep the attr spellings stable and readable. The preferred textual spellings continue to be forms like:
   - `#vc4.add_opcode<add>`
   - `#vc4.mul_opcode<fmul>`
   - `#vc4.cond<always>`
   - `#vc4.load_imm_mode<splat32>`
6. Keep `include/vc4/Dialect/VC4/IR/VC4Attrs.td` and `include/vc4/Dialect/VC4/IR/VC4OpEnums.td` consistent with the enum definitions.
7. Update tests only where enum textual spelling or enum value correctness requires it. Relevant tests are primarily:
   - `test/Dialect/VC4/enum-attr-roundtrip.mlir`
   - `test/Dialect/VC4/qpu-bundle-roundtrip.mlir`
   - `test/Dialect/VC4/qpu-bundle-invalid.mlir`
   - `test/Dialect/VC4/qpu-branch-roundtrip.mlir`
   - `test/Dialect/VC4/qpu-ldi-sema-roundtrip.mlir`
   - `test/Dialect/VC4/sync-thread-roundtrip.mlir`
8. Do **not** rewrite `VC4FuncOp::parse` or `VC4FuncOp::print` in this prompt.
9. Do **not** change the parentage or semantics of ops in this prompt.
10. Do **not** implement passes in this prompt.

Implementation notes:
- If an enum is conceptual rather than a direct hardware encoding, leave it alone unless it is clearly wrong.
- Add short comments in `VC4Enums.td` where helpful to explain that omitted dialect members do not imply renumbering.
- Preserve source compatibility of generated enum attrs as much as possible.

At the end:
- configure/build/tests if possible
- fix any failures before finishing
- summarize exactly which enum definitions changed
```

### Verification Prompt H1

```text
You are verifying the repository state after Hardening Prompt H1 of the VC4 dialect hardening plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- include/vc4/Dialect/VC4/IR/VC4Enums.td
- include/vc4/Dialect/VC4/IR/VC4Attrs.td
- include/vc4/Dialect/VC4/IR/VC4OpEnums.td
- include/vc4/Dialect/VC4/IR/VC4StructuredOps.td
- lib/Dialect/VC4/IR/VC4Ops.cpp
- test/Dialect/VC4/enum-attr-roundtrip.mlir
- test/Dialect/VC4/qpu-bundle-roundtrip.mlir
- test/Dialect/VC4/qpu-bundle-invalid.mlir
- test/Dialect/VC4/qpu-branch-roundtrip.mlir
- test/Dialect/VC4/qpu-ldi-sema-roundtrip.mlir
- test/Dialect/VC4/sync-thread-roundtrip.mlir

Expected state after H1:
1. `include/vc4/Dialect/VC4/IR/VC4Enums.td` uses real hardware values for the sink-facing/hardware-facing enums, not dense placeholder numbering.
2. The following are specifically correct in the source file:
   - `VC4_AddOpcode` reflects the hardware opcode table.
   - `VC4_MulOpcode` reflects the hardware opcode table.
   - `VC4_Cond` reflects the hardware condition-code table.
   - `VC4_LoadImmMode` reflects the hardware load-immediate form encoding choices.
   - `VC4_RegfileAUnpackMode`, `VC4_RegfileAPackMode`, `VC4_R4UnpackMode`, and `VC4_MulPackMode` reflect the hardware pack/unpack tables.
   - `VC4_QPUSignal` preserves hardware numbers and includes the explicit last-thread-switch signal value.
   - `VC4_QPUMux` reflects the ALU input mux table.
   - `VC4_BranchCond` reflects the hardware branch condition values and unconditional branch condition.
3. `include/vc4/Dialect/VC4/IR/VC4Attrs.td` and `include/vc4/Dialect/VC4/IR/VC4OpEnums.td` are consistent with the updated enums.
4. Tests involving textual enum attrs and scheduled sink ops are updated if needed and still coherent.
5. `VC4FuncOp::parse` / `print` and the core Prompt-2 function/container infrastructure were not rewritten in this prompt.

Your job:
- inspect the repository
- compare the implementation against the expected H1 state above
- fix any mismatches directly
- run build/tests if possible
- do not invent new design beyond H1
- do not start function-domain or codegen-contract work yet

Required checks:
- Inspect exact enum cases and numeric values in `include/vc4/Dialect/VC4/IR/VC4Enums.td`.
- Confirm the enum-attr spellings still parse and print in the roundtrip tests.
- Confirm no accidental renumbering of remaining `VC4_QPUSignal` members happened when some fragment-only signals were omitted.
- Confirm the last-thread-switch signal exists explicitly.

Build/test instructions:
- Configure a build directory if needed.
- If lit is not auto-discovered, retry configure with:
  `-DLLVM_EXTERNAL_LIT="$(command -v llvm-lit || command -v lit)"`
- Build and run:
  - `vc4-opt`
  - `check-vc4`
- Fix any failures before finishing.

At the end, report:
- whether the source matches H1 exactly
- any fixes you applied
- the exact commands you ran
- whether `check-vc4` passed
```

---

## Prompt H2 — add explicit function execution domain and normalize function attrs

```text
You are continuing the VC4 dialect hardening milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- include/vc4/Dialect/VC4/IR/VC4Enums.td
- include/vc4/Dialect/VC4/IR/VC4OpEnums.td
- include/vc4/Dialect/VC4/IR/VC4StructuredOps.td
- lib/Dialect/VC4/IR/VC4Ops.cpp
- all existing tests under `test/Dialect/VC4/`

Task:
Add an explicit function execution-domain distinction so the dialect no longer mixes QPU/device functions and host/system functions too loosely, and normalize function attrs in tests to dialect enum syntax.

Required outcomes:
1. Add a new function-level enum in `include/vc4/Dialect/VC4/IR/VC4Enums.td` and matching op enum attr in `include/vc4/Dialect/VC4/IR/VC4OpEnums.td`.
2. The new enum must distinguish at least:
   - `qpu`
   - `host`
3. Add the corresponding required attr to `vc4.func` in `include/vc4/Dialect/VC4/IR/VC4StructuredOps.td`.
   - The attr name should be `domain`.
   - It should be required in practice, not optional-by-accident.
4. Update `lib/Dialect/VC4/IR/VC4Ops.cpp` so `vc4.func` verification now requires all three function attrs:
   - `threading`
   - `form`
   - `domain`
5. Update `lib/Dialect/VC4/IR/VC4Ops.cpp` function verification so the domain meaning is enforced conservatively:
   - `domain = #vc4.execution_domain<qpu>` allows the existing QPU/device-side structured ops and scheduled sink ops according to `form`.
   - `domain = #vc4.execution_domain<qpu>` must reject the host/system launcher-facing ops:
     - `vc4.enqueue_qpu`
     - `vc4.reserve_qpu`
     - `vc4.v3d.query`
     - `vc4.v3d.configure`
   - `domain = #vc4.execution_domain<host>` must reject all device-side QPU ops, including:
     - `vc4.builtin`
     - `vc4.uniform.*`
     - `vc4.mov`
     - `vc4.read`
     - `vc4.alu.*`
     - `vc4.load_imm`
     - `vc4.pack`
     - `vc4.unpack`
     - `vc4.rotate`
     - `vc4.tmu.*`
     - `vc4.sfu.*`
     - `vc4.vpm.*`
     - `vc4.dma.*`
     - `vc4.mutex`
     - `vc4.semaphore`
     - `vc4.host_interrupt`
     - `vc4.thread_switch`
     - `vc4.program_end`
     - all `vc4.qpu.*`
   - `domain = #vc4.execution_domain<host>` may continue to allow:
     - `vc4.enqueue_qpu`
     - `vc4.reserve_qpu`
     - `vc4.v3d.query`
     - `vc4.v3d.configure`
     - `vc4.async.wait`
     - `vc4.cf.branch`
     - `vc4.return`
6. Keep the existing `kernel` unit attr, but make verification enforce:
   - `kernel` is only legal on `domain = #vc4.execution_domain<qpu>` functions.
7. Normalize all function attrs in tests away from raw integer spellings like `threading = 0 : i32` and `form = 1 : i32`.
   - Use dialect enum attrs instead.
   - Example target style:
     - `threading = #vc4.threading_mode<single>`
     - `form = #vc4.function_form<structured>`
     - `domain = #vc4.execution_domain<qpu>`
8. Update all affected tests under `test/Dialect/VC4/`.
   - Be concrete and systematic.
   - At minimum inspect and update:
     - `module-func-roundtrip.mlir`
     - `invalid.mlir`
     - `function-form-segregation-invalid.mlir`
     - `uniform-and-read-*.mlir`
     - `alu-and-load-imm-*.mlir`
     - `value-shape-*.mlir`
     - `tmu-*.mlir`
     - `sfu-*.mlir`
     - `vpm-*.mlir`
     - `dma-*.mlir`
     - `structured-system-*.mlir`
     - `sync-thread-*.mlir`
     - `qpu-*.mlir`
     - `side-effects.mlir`
     - `types-roundtrip.mlir`
9. Do **not** rewrite `VC4FuncOp::parse` / `print` in this prompt unless it is absolutely required for the new attr syntax, and if it is required keep the change minimal.
10. Do **not** implement new passes in this prompt.

Implementation notes:
- Keep the domain verifier conservative and whitelist-based rather than clever.
- If a function-form/domain split forces a test-file split, that is acceptable, but do not rename files unnecessarily.
- Preserve the current dialect structure; this is a hardening prompt, not a redesign prompt.

At the end:
- build and run tests
- fix failures before finishing
- summarize the exact op families assigned to qpu vs host domain
```

### Verification Prompt H2

```text
You are verifying the repository state after Hardening Prompt H2 of the VC4 dialect hardening plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- include/vc4/Dialect/VC4/IR/VC4Enums.td
- include/vc4/Dialect/VC4/IR/VC4OpEnums.td
- include/vc4/Dialect/VC4/IR/VC4StructuredOps.td
- lib/Dialect/VC4/IR/VC4Ops.cpp
- test/Dialect/VC4/*.mlir

Expected state after H2:
1. `vc4.func` has a required `domain` attr with a dialect enum attr type.
2. The new execution-domain enum and attr exist in the IR definitions.
3. `vc4.func` verification requires `threading`, `form`, and `domain`.
4. QPU-domain functions reject host/system launcher-facing ops.
5. Host-domain functions reject device/QPU ops.
6. `kernel` is only legal on QPU-domain functions.
7. Function attrs in tests use dialect enum attr syntax rather than raw integer literals.
8. Existing test coverage remains coherent after the migration.

Your job:
- inspect the repository
- compare implementation against the exact expected state above
- fix any mismatches directly
- run build/tests if possible
- do not add emit-contract passes yet
- do not add lowering/codegen/runtime work

Concrete inspection requirements:
- Check that `include/vc4/Dialect/VC4/IR/VC4StructuredOps.td` now includes a `domain` attr on `VC4_FuncOp`.
- Check that `lib/Dialect/VC4/IR/VC4Ops.cpp` function verification enforces domain separation with explicit op-family checks.
- Check that test files no longer use `threading = 0 : i32` / `form = 0 : i32` style for `vc4.func` attrs.
- Check that representative tests now spell attrs like:
  - `#vc4.threading_mode<single>`
  - `#vc4.function_form<structured>`
  - `#vc4.execution_domain<qpu>`
- Check that host/system tests actually use host-domain functions where appropriate.
- Check that scheduled sink tests remain QPU-domain.

Build/test instructions:
- Configure if needed, with `LLVM_EXTERNAL_LIT` fallback if needed.
- Build and run `vc4-opt` and `check-vc4`.
- Fix all failures before finishing.

At the end, report:
- whether H2 is satisfied exactly
- any fixes applied
- the commands run
- whether `check-vc4` passed
```

---

## Prompt H3 — add explicit emit-contract verification pass and document the boundary

```text
You are continuing the VC4 dialect hardening milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- include/vc4/Dialect/VC4/IR/VC4StructuredOps.td
- include/vc4/Dialect/VC4/IR/VC4Ops.h
- lib/Dialect/VC4/IR/VC4Ops.cpp
- tools/vc4-opt/vc4-opt.cpp
- test/Dialect/VC4/*.mlir

Task:
Codify the current “what is legal input to code generation later” boundary without implementing code generation.

Required outcomes:
1. Add a verifier-style pass named exactly:
   - `--vc4-verify-emit-contract`
2. Implement it in the smallest clean place.
   - It is acceptable to implement it in `tools/vc4-opt/vc4-opt.cpp` like the existing `VC4TestPrintEffectsPass` if that is the least intrusive option.
   - Do not add codegen.
3. The pass must verify the contract conservatively at module/function level.
4. The pass must distinguish two codegen-targetable categories:
   - **QASM-input candidates**:
     - `vc4.func`
     - `domain = #vc4.execution_domain<qpu>`
     - `form = #vc4.function_form<scheduled>`
     - only scheduled sink ops are allowed in the function body
   - **Launcher/system-input candidates**:
     - `vc4.func`
     - `domain = #vc4.execution_domain<host>`
     - `form = #vc4.function_form<structured>`
     - only host/system ops allowed by H2 are allowed in the function body
5. The pass must explicitly reject the following mismatches with diagnostics:
   - scheduled qpu-domain function containing structured device ops
   - scheduled qpu-domain function containing host/system ops
   - host structured function containing qpu device ops
   - host structured function containing scheduled sink ops
   - qpu structured functions being treated as qasm-emittable
6. Keep this as a verifier boundary only. Do not mutate IR.
7. Add focused tests:
   - one positive roundtrip-ish file or pass test that contains:
     - one qpu scheduled function valid for later qasm emission
     - one host structured function valid for later launcher emission
   - one negative file exercising each rejected mixed case above
8. Update `docs/vc4-implementation-guide.md` with a short, concrete section that says:
   - qasm emission later will consume scheduled qpu-domain functions only
   - launcher generation later will consume host structured functions only
   - qpu structured functions are not directly emittable
   - structured ops such as uniforms/TMU/VPM/DMA/value-shape ops must be lowered/normalized before qasm emission
9. Do **not** implement lowering or code generation.
10. Do **not** change existing dialect semantics outside what is needed for this verifier boundary.

Implementation notes:
- Make diagnostics specific and actionable.
- Reuse existing domain/form information rather than inventing a second classification system.
- It is acceptable for the pass to require a single-block linear scheduled function if that meaningfully simplifies the contract, but if you do this you must document it and test it.

At the end:
- build and run tests
- fix all failures before finishing
- summarize the exact contract the pass enforces
```

### Verification Prompt H3

```text
You are verifying the repository state after Hardening Prompt H3 of the VC4 dialect hardening plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- tools/vc4-opt/vc4-opt.cpp
- include/vc4/Dialect/VC4/IR/VC4StructuredOps.td
- lib/Dialect/VC4/IR/VC4Ops.cpp
- test/Dialect/VC4/*.mlir

Expected state after H3:
1. A pass named `--vc4-verify-emit-contract` exists and is registered in `vc4-opt`.
2. The pass is verifier-only and does not mutate IR.
3. It enforces a concrete distinction between:
   - qpu scheduled functions that are candidates for future qasm emission
   - host structured functions that are candidates for future launcher/system emission
4. It rejects mixed or illegal cases with useful diagnostics.
5. `docs/vc4-implementation-guide.md` now contains a short explicit codegen-boundary section.
6. There are dedicated tests for the pass, including positive and negative coverage.

Your job:
- inspect the repository
- verify the exact pass name, registration, and behavior
- fix any mismatches directly
- run the relevant tests
- do not add codegen

Concrete checks:
- Inspect `tools/vc4-opt/vc4-opt.cpp` and confirm the pass argument string is exactly `vc4-verify-emit-contract`.
- Inspect test files and confirm there is at least one positive and one negative test specifically exercising this pass.
- Inspect `docs/vc4-implementation-guide.md` and confirm it now spells out the legal-to-emit boundary.
- Confirm the pass distinguishes host-vs-qpu and structured-vs-scheduled as described above.

Build/test instructions:
- Configure if needed, with `LLVM_EXTERNAL_LIT` fallback if needed.
- Build and run `vc4-opt` and `check-vc4`.
- Fix all failures before finishing.

At the end, report:
- whether H3 is satisfied exactly
- any fixes applied
- commands run
- whether `check-vc4` passed
```

---

## Prompt H4A — complete scheduled sink local encoding/verifier coverage

```text
You are continuing the VC4 dialect hardening milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- include/vc4/Dialect/VC4/IR/VC4Enums.td
- include/vc4/Dialect/VC4/IR/VC4StructuredOps.td
- lib/Dialect/VC4/IR/VC4Ops.cpp
- test/Dialect/VC4/qpu-bundle-roundtrip.mlir
- test/Dialect/VC4/qpu-bundle-invalid.mlir
- test/Dialect/VC4/qpu-ldi-sema-roundtrip.mlir
- test/Dialect/VC4/qpu-ldi-sema-invalid.mlir
- test/Dialect/VC4/qpu-branch-roundtrip.mlir

Task:
Tighten the scheduled sink surface so the local encoding/verifier rules better match the actual QPU instruction formats and register-space rules.

Required outcomes:
1. In `lib/Dialect/VC4/IR/VC4Ops.cpp`, update the scheduled sink verifiers so `vc4.qpu.bundle` read-address checking reflects the ALU instruction encoding, not an artificially truncated subset.
2. Specifically:
   - `vc4.qpu.bundle` `raddr_a` and `raddr_b` must allow the full 6-bit read-address range `[0, 63]`.
   - `vc4.qpu.branch` `raddr_a` must remain checked against the branch instruction field width, i.e. `[0, 31]`.
3. Update the small-immediate verifier for `vc4.qpu.bundle` so it accepts the full hardware encoding range `[0, 63]`, not just `[0, 47]`.
4. Add comments/helper naming that distinguish:
   - immediate small-immediate values / float literals
   - rotate-by-r5 encoding (`48`)
   - immediate rotate encodings (`49`–`63`)
5. Add a local verifier rule for `vc4.qpu.sema` derived from the hardware note on the semaphore instruction:
   - reject stall-capable closely-coupled peripheral write addresses in `waddr_add` / `waddr_mul`
   - at minimum reject TLB write addresses `43`–`47`, SFU `52`–`55`, and TMU `56`–`63`
   - do not overreach into unrelated restrictions in this prompt
6. Add or update tests to prove the new local legality:
   - add a positive scheduled sink roundtrip case using a `small_imm` rotate encoding (`48` or `49`–`63`)
   - add a positive roundtrip case with `raddr_a` or `raddr_b` in the I/O range, for example `32` or `35`
   - add negative tests for:
     - bundle `small_imm = 64`
     - bundle `raddr_b = 64`
     - qpu.sema targeting a forbidden stall-capable write address
7. Keep this prompt local to sink encodings/verifiers.
8. Do **not** implement cross-instruction hazard analysis yet.
9. Do **not** change domain separation or emit-contract behavior here.

At the end:
- build and run tests
- fix failures before finishing
- summarize the exact local sink-verifier rules added or changed
```

### Verification Prompt H4A

```text
You are verifying the repository state after Hardening Prompt H4A of the VC4 dialect hardening plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- include/vc4/Dialect/VC4/IR/VC4Enums.td
- include/vc4/Dialect/VC4/IR/VC4StructuredOps.td
- lib/Dialect/VC4/IR/VC4Ops.cpp
- test/Dialect/VC4/qpu-bundle-roundtrip.mlir
- test/Dialect/VC4/qpu-bundle-invalid.mlir
- test/Dialect/VC4/qpu-ldi-sema-roundtrip.mlir
- test/Dialect/VC4/qpu-ldi-sema-invalid.mlir
- test/Dialect/VC4/qpu-branch-roundtrip.mlir

Expected state after H4A:
1. `vc4.qpu.bundle` read addresses allow the full hardware 6-bit read space `[0, 63]`.
2. `vc4.qpu.branch` still checks `raddr_a` against the branch-field range `[0, 31]`.
3. `vc4.qpu.bundle` allows `small_imm` encodings in `[0, 63]`.
4. The code clearly distinguishes literal vs rotate small-immediate encodings.
5. `vc4.qpu.sema` rejects stall-capable TLB/SFU/TMU write addresses locally.
6. Tests exist and cover:
   - positive rotate/small_imm scheduled bundle case
   - positive scheduled bundle case with read address in I/O range
   - negative out-of-range `small_imm`
   - negative out-of-range `raddr_b`
   - negative forbidden `qpu.sema` write address

Your job:
- inspect the repository
- compare implementation to the exact expected state above
- fix mismatches directly
- run build/tests
- do not add cross-instruction hazard work yet

Concrete inspection requirements:
- Check the helper/verifier functions in `lib/Dialect/VC4/IR/VC4Ops.cpp` rather than assuming tests imply correctness.
- Confirm the qpu bundle verifier no longer rejects valid register-space I/O read addresses.
- Confirm the qpu sema verifier contains an explicit forbidden-range check.
- Confirm tests exercise the new accepted and rejected cases.

Build/test instructions:
- Configure if needed, with `LLVM_EXTERNAL_LIT` fallback if needed.
- Build and run `vc4-opt` and `check-vc4`.
- Fix all failures before finishing.

At the end, report:
- whether H4A is satisfied exactly
- any fixes applied
- commands run
- whether `check-vc4` passed
```

---

## Prompt H4B — add scheduled end-of-program and thread-signal legality verifier pass

```text
You are continuing the VC4 dialect hardening milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- include/vc4/Dialect/VC4/IR/VC4Enums.td
- include/vc4/Dialect/VC4/IR/VC4StructuredOps.td
- lib/Dialect/VC4/IR/VC4Ops.cpp
- tools/vc4-opt/vc4-opt.cpp
- all existing `test/Dialect/VC4/qpu-*.mlir` tests

Task:
Add a verifier-style pass for scheduled hardware legality around thread-end and thread-signal restrictions that are inherently cross-instruction and do not belong in individual op verifiers.

Required outcomes:
1. Add a pass named exactly:
   - `--vc4-verify-scheduled-hardware-rules`
2. This pass must operate only as a verifier. Do not mutate IR.
3. The pass must inspect scheduled QPU-domain functions only.
4. It must check the following concrete hardware-rule subset, derived from the hardware guide and representable from the current sink IR:
   - if a scheduled instruction signals program end / thread end, that instruction must not write to physical regfile A or B
   - the thread-end instruction and its following two instructions must not read or write regfile address `14`
   - the thread-end instruction and its following two instructions must not perform uniform or VPM/VDR/VDW accesses
   - a `last_thread_switch` signal is only legal in functions with `threading = #vc4.threading_mode<threadable>`
5. Define the window over the instruction stream concretely and document it in code comments.
6. If you need a simple assumption to make this verifier well-defined, you may require scheduled functions checked by this pass to be single-block top-level linear functions, while still correctly accounting for `vc4.qpu.branch` delay-slot regions as explicit nested linear instruction sequences.
7. Add targeted tests:
   - at least one positive scheduled function that passes the verifier
   - negative tests for each bullet above
8. Do **not** attempt to implement all hardware sequence restrictions in this prompt.
9. Do **not** add codegen.

Implementation notes:
- Keep the pass scope narrow and explain the exact checked subset in a comment near registration or implementation.
- Reuse existing read/write-address helpers where practical.
- Be explicit about which register-space addresses correspond to uniforms and VPM/VDR/VDW.

At the end:
- build and run tests
- fix failures before finishing
- summarize the exact scheduled-hardware subset now covered by the pass
```

### Verification Prompt H4B

```text
You are verifying the repository state after Hardening Prompt H4B of the VC4 dialect hardening plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- tools/vc4-opt/vc4-opt.cpp
- lib/Dialect/VC4/IR/VC4Ops.cpp
- test/Dialect/VC4/qpu-*.mlir

Expected state after H4B:
1. A verifier-style pass named `--vc4-verify-scheduled-hardware-rules` exists and is registered.
2. The pass checks a concrete, documented subset of cross-instruction hardware rules for scheduled QPU-domain functions.
3. That subset includes:
   - thread-end instruction must not write physical regfile A/B
   - thread-end plus following two instructions must not touch regfile address 14
   - thread-end plus following two instructions must not do uniform or VPM/VDR/VDW accesses
   - last-thread-switch only legal in threadable functions
4. There are dedicated tests covering both accepted and rejected cases.
5. The pass is verifier-only.

Your job:
- inspect the repository
- compare implementation against the exact expected state above
- fix mismatches directly
- run build/tests
- do not expand the pass to unrelated hazards in this verifier session unless needed to repair a mismatch

Concrete checks:
- Confirm the pass argument string is exactly `vc4-verify-scheduled-hardware-rules`.
- Confirm the implementation actually inspects scheduled QPU-domain functions, not all functions indiscriminately.
- Confirm the negative tests map directly to the documented rule subset.
- Confirm the pass is not silently acting as a transform.

Build/test instructions:
- Configure if needed, with `LLVM_EXTERNAL_LIT` fallback if needed.
- Build and run `vc4-opt` and `check-vc4`.
- Fix all failures before finishing.

At the end, report:
- whether H4B is satisfied exactly
- any fixes applied
- commands run
- whether `check-vc4` passed
```

---

## Prompt H4C — add adjacent-instruction hazard verifier pass for the subset representable today

```text
You are continuing the VC4 dialect hardening milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- include/vc4/Dialect/VC4/IR/VC4Enums.td
- include/vc4/Dialect/VC4/IR/VC4StructuredOps.td
- lib/Dialect/VC4/IR/VC4Ops.cpp
- tools/vc4-opt/vc4-opt.cpp
- test/Dialect/VC4/qpu-*.mlir

Task:
Add a second verifier-style scheduled-hardware pass for the adjacent-instruction hazard subset that is directly representable from the current sink IR.

Required outcomes:
1. Add a pass named exactly:
   - `--vc4-verify-scheduled-adjacent-hazards`
2. The pass must be verifier-only.
3. It must check, at minimum, the following representable subset:
   - if instruction N writes a physical regfile-A location `0..31`, instruction N+1 must not read that same physical regfile-A location
   - if instruction N writes a physical regfile-B location `0..31`, instruction N+1 must not read that same physical regfile-B location
   - if instruction N writes `r5`, instruction N+1 must not use the `small_imm = 48` rotate-by-r5 encoding
   - if instruction N writes one of the SFU write addresses (`52`–`55`), the next two instructions must not read `r4` and must not trigger another write-to-r4 event from the subset currently representable in sink IR
4. You may define the “write-to-r4 event subset currently representable in sink IR” conservatively as:
   - TMU load/read signals represented in `vc4.qpu.bundle` signal values
   - another SFU write
5. Document the exact subset in code comments. Do not claim to verify more than the current sink IR can actually express.
6. Reuse or share linearization helpers with H4B if sensible.
7. Add focused tests:
   - positive case(s) that pass
   - negative case for regfile A hazard
   - negative case for regfile B hazard
   - negative case for rotate-by-r5 after r5 write
   - negative case for SFU/r4 hazard window
8. Do **not** try to implement every instruction-sequence rule from the hardware guide in this prompt.
9. Do **not** add codegen.

At the end:
- build and run tests
- fix failures before finishing
- summarize the exact adjacent-hazard subset now covered
```

### Verification Prompt H4C

```text
You are verifying the repository state after Hardening Prompt H4C of the VC4 dialect hardening plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- tools/vc4-opt/vc4-opt.cpp
- lib/Dialect/VC4/IR/VC4Ops.cpp
- test/Dialect/VC4/qpu-*.mlir

Expected state after H4C:
1. A verifier-style pass named `--vc4-verify-scheduled-adjacent-hazards` exists and is registered.
2. It checks a narrow, documented subset of adjacent-instruction hazards representable from the current sink IR.
3. That subset includes:
   - regfile A write-then-next-instruction-read same physical address
   - regfile B write-then-next-instruction-read same physical address
   - rotate-by-r5 immediately after writing r5
   - SFU/r4 two-instruction hazard window for the representable subset
4. Tests exist for each of those cases.
5. The pass is verifier-only.

Your job:
- inspect the repository
- compare implementation against the exact expected state above
- fix mismatches directly
- run build/tests
- do not expand the pass beyond the documented subset unless required to repair a mismatch

Concrete checks:
- Confirm the exact pass argument string.
- Confirm the pass operates on scheduled QPU-domain functions.
- Confirm the tests directly exercise the documented hazards rather than unrelated failures.
- Confirm the implementation comments accurately describe the checked subset.

Build/test instructions:
- Configure if needed, with `LLVM_EXTERNAL_LIT` fallback if needed.
- Build and run `vc4-opt` and `check-vc4`.
- Fix all failures before finishing.

At the end, report:
- whether H4C is satisfied exactly
- any fixes applied
- commands run
- whether `check-vc4` passed
```

---

## Prompt H5 — settle and document the structured value-shape contract (`pack` / `unpack` / `rotate`)

```text
You are continuing the VC4 dialect hardening milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- include/vc4/Dialect/VC4/IR/VC4StructuredOps.td
- lib/Dialect/VC4/IR/VC4Ops.cpp
- test/Dialect/VC4/value-shape-roundtrip.mlir
- test/Dialect/VC4/value-shape-invalid.mlir
- test/Dialect/VC4/alu-and-load-imm-roundtrip.mlir

Task:
Settle the structured SSA contract for the value-shape family so it is stable enough to lower from and normalize away later.

Required outcomes:
1. Explicitly document in `docs/vc4-implementation-guide.md` the intended contract for:
   - `vc4.pack`
   - `vc4.unpack`
   - `vc4.rotate`
   - `vc4.load_imm` structured form, where relevant
2. The documentation must make the following concrete:
   - these are **structured** ops, not scheduled sink ops
   - they are expected to normalize away before qasm emission
   - `pack` / `unpack` are modeled as 32-bit carrier-word transformations in structured SSA, not as true MLIR `i8`/`i16` storage types
   - `rotate` is a structured 16-lane operation and is not itself a sink encoding
3. Reflect that contract in code comments in `include/vc4/Dialect/VC4/IR/VC4StructuredOps.td` or `lib/Dialect/VC4/IR/VC4Ops.cpp`.
4. Tighten verifiers only where needed to make the contract explicit and consistent.
   - Do not redesign the ops.
   - Do not move them into sink form.
5. Make sure `value-shape-roundtrip.mlir` and `value-shape-invalid.mlir` match the documented contract.
   - If any test still implies narrow typed lane semantics rather than 32-bit carrier semantics, fix it.
6. Add at least one explicit negative test that demonstrates a non-carrier typed or otherwise out-of-contract use is rejected.
7. Do **not** implement lowering/normalization in this prompt.

Implementation notes:
- This prompt is about stabilizing the dialect contract, not changing the architectural direction.
- Keep this prompt scoped to docs/comments/verifier consistency.

At the end:
- build and run tests
- fix failures before finishing
- summarize the final structured value-shape contract in 4–8 bullets
```

### Verification Prompt H5

```text
You are verifying the repository state after Hardening Prompt H5 of the VC4 dialect hardening plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- include/vc4/Dialect/VC4/IR/VC4StructuredOps.td
- lib/Dialect/VC4/IR/VC4Ops.cpp
- test/Dialect/VC4/value-shape-roundtrip.mlir
- test/Dialect/VC4/value-shape-invalid.mlir
- test/Dialect/VC4/alu-and-load-imm-roundtrip.mlir

Expected state after H5:
1. The implementation guide explicitly explains the structured SSA contract for `vc4.pack`, `vc4.unpack`, `vc4.rotate`, and the relevant `vc4.load_imm` structured behavior.
2. The docs explicitly state that these ops are normalization targets and not direct qasm-emission input.
3. Code comments in the IR definitions or verifiers align with the documentation.
4. The value-shape tests match the documented carrier-word semantics.
5. There is at least one explicit negative test for an out-of-contract value-shape use.

Your job:
- inspect the repository
- compare implementation against the exact expected state above
- fix mismatches directly
- run build/tests
- do not add lowering/codegen

Concrete checks:
- Inspect the implementation guide text itself, not just the code.
- Confirm the tests no longer imply narrow storage-type semantics for structured pack/unpack.
- Confirm the docs say these ops are not direct qasm-emission input.

Build/test instructions:
- Configure if needed, with `LLVM_EXTERNAL_LIT` fallback if needed.
- Build and run `vc4-opt` and `check-vc4`.
- Fix all failures before finishing.

At the end, report:
- whether H5 is satisfied exactly
- any fixes applied
- commands run
- whether `check-vc4` passed
```

---

## Suggested checkpoint plan

After each successful verification prompt:

```bash
git add -A
git commit -m "H1 verified: hardware enum audit"
# then after H2, H3, H4A, H4B, H4C, H5 similarly
```

Suggested tag after H5:

```bash
git tag vc4-dialect-hardened
```

At that point, the dialect should be in a much safer state for starting the next milestone: structured-to-scheduled lowering and/or qasm/launcher generation, while still leaving room for later refactoring if needed.
