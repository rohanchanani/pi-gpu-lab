# VC4 pre-codegen hardening sequence

This is a short, targeted hardening round to finish the dialect/emission contract before starting the qasm / `.c` / `.h` bundle pipeline.

The current tree is close, but not quite ready. The remaining issues are narrow and concrete:

- the current scheduled-hardware verifier still mis-models the “last three instructions must not access uniforms / VPM / VDR / VDW” rule
- the direct-emission contract still accepts scheduled QPU functions that do not have a real `thrend` epilogue
- sink-level spacing rules for `UNIFORMS_ADDRESS` and `TMU_NOSWAP` are not yet checked
- the current sink subset still lacks a verifier for “one closely-coupled peripheral access per instruction” in the representable compute subset

Run each implementation prompt from a fresh Codex session. After each verification prompt succeeds, commit.

---

## Prompt P1: shared QPU register-space helpers + fix the thread-end window bug

You are working in the `compiler/` repo root from a cold session.

Read these files first:
- `AGENTS.md` if present
- `include/vc4/Dialect/VC4/IR/VC4Enums.td`
- `lib/Dialect/VC4/IR/VC4Ops.cpp`
- `tools/vc4-opt/vc4-opt.cpp`
- `test/Dialect/VC4/qpu-scheduled-hardware-rules.mlir`
- `test/Dialect/VC4/qpu-scheduled-hardware-rules-invalid.mlir`

Goal:
Make the sink-level register-space modelling explicit and correct, and fix the current bug where the hardware-rules pass treats `raddr = 14` as if it were a uniform read.

Do the following:
1. Add a new shared header under `include/vc4/Dialect/VC4/IR/` named `VC4QPURegisterInfo.h`.
2. In that header, add named `constexpr` register addresses / ranges and helper predicates for the sink-level compute subset. At minimum include:
   - physical regfile read/write range `0..31`
   - `UNIFORM_READ = 32`
   - `VARYING_READ = 35`
   - `TMU_NOSWAP = 36`
   - `R5_WRITE = 37`
   - `UNIFORMS_ADDRESS = 40`
   - `VPM / VDR / VDW register-space range = 48..50`
   - `MUTEX = 51`
   - `SFU range = 52..55`
   - `TMU parameter write range = 56..63`
   - helper predicates for each of the above categories
3. Update `tools/vc4-opt/vc4-opt.cpp` to use the shared helpers instead of hard-coded magic numbers.
4. Correct the existing “thread-end hazard window” modelling so that it separately checks:
   - physical regfile address `14` hazard
   - uniform reads via `raddr = 32`
   - varying reads via `raddr = 35`
   - VPM / VDR / VDW register-space accesses via `48..50`
5. Fix all diagnostics so they describe the actual forbidden access category. Do not leave any message claiming that uniform reads use address `14`.
6. Add focused test coverage in `test/Dialect/VC4/`:
   - one negative case where a last-three-instructions slot reads a uniform (`raddr = 32`)
   - one negative case where a last-three-instructions slot reads a varying (`raddr = 35`)
   - one negative case where a last-three-instructions slot touches VPM / VDR / VDW (`48..50`)
   - one negative case where address `14` is tested independently
   - one positive case showing that mutex address `51` is *not* misclassified as VPM / VDR / VDW
7. Do not add fragment / TLB / scoreboard features.
8. Do not redesign the dialect. This is a correctness-and-shared-helpers patch only.

Build and test before finishing:
- `cmake -G Ninja -S . -B build-verify`
- if lit is not auto-discovered, retry with:
  - `cmake -G Ninja -S . -B build-verify -DLLVM_EXTERNAL_LIT="$(command -v llvm-lit || command -v lit)"`
- `ninja -C build-verify vc4-opt check-vc4`
- run the specific tests for the edited files with `llvm-lit` / `lit` if available

Deliverable:
- the code changes
- passing tests
- a short summary of exactly which bad register-space assumptions were fixed

---

## Verification prompt P1

You are verifying the repository state after Prompt P1 of the VC4 pre-codegen hardening plan.

Read:
- `AGENTS.md` if present
- `include/vc4/Dialect/VC4/IR/VC4QPURegisterInfo.h`
- `tools/vc4-opt/vc4-opt.cpp`
- `test/Dialect/VC4/qpu-scheduled-hardware-rules.mlir`
- `test/Dialect/VC4/qpu-scheduled-hardware-rules-invalid.mlir`

Expected state after Prompt P1:
1. There is a shared header with named QPU register-space helpers.
2. `vc4-opt.cpp` uses those helpers instead of repeating raw address constants for the covered categories.
3. The thread-end window logic no longer treats address `14` as a uniform read.
4. The verifier distinguishes these cases correctly:
   - regfile address `14`
   - uniform read `32`
   - varying read `35`
   - VPM / VDR / VDW register-space access `48..50`
5. There is a positive test proving mutex address `51` is not misclassified as VPM / VDR / VDW.
6. The tree is still compute-focused and has not added fragment / TLB / scoreboard work.

Your job:
- inspect the repository
- identify any mismatch versus the expected state
- fix mismatches directly
- configure/build/tests if possible
- when configuring, if lit is not auto-discovered, retry with
  `-DLLVM_EXTERNAL_LIT="$(command -v llvm-lit || command -v lit)"`
- run at least the targeted hardware-rules tests and then `check-vc4`
- do not delete consistent later work

---

## Prompt P2: make the direct qasm-input contract require a real thread-end epilogue

You are working in the `compiler/` repo root from a cold session.

Read these files first:
- `AGENTS.md` if present
- `tools/vc4-opt/vc4-opt.cpp`
- `test/Dialect/VC4/verify-emit-contract.mlir`
- `test/Dialect/VC4/verify-emit-contract-invalid.mlir`
- `test/Dialect/VC4/qpu-bundle-roundtrip.mlir`
- `test/Dialect/VC4/qpu-ldi-sema-roundtrip.mlir`
- `test/Dialect/VC4/qpu-branch-roundtrip.mlir`

Goal:
Strengthen the “directly emittable” contract so that a scheduled QPU function is only considered valid qasm input if it has an explicit hardware-valid program-end epilogue.

Do the following:
1. Reuse the existing flattened scheduled-instruction stream logic in `vc4-opt.cpp`.
2. Update `--vc4-verify-emit-contract` so that every non-external
   `domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>`
   function must satisfy all of the following:
   - the flattened instruction stream has at least 3 instruction slots
   - slot `N-3` is a `vc4.qpu.bundle` whose `sig` is `#vc4.qpu_signal<thrend>`
   - slots `N-2` and `N-1` exist and are scheduled non-branch ops (`vc4.qpu.bundle`, `vc4.qpu.ldi`, or `vc4.qpu.sema`)
   - no earlier slot in the same function carries `sig = #vc4.qpu_signal<thrend>`
3. Keep `thrend` represented via `vc4.qpu.bundle`; do not add a new sink op.
4. Emit a clear diagnostic saying the function is not directly emittable because qasm input requires an explicit `thrend` plus two delay-slot instructions.
5. Update the positive qasm-input test to include a real final three-slot epilogue.
6. Add negative tests for at least:
   - missing `thrend`
   - duplicate `thrend`
   - `thrend` not being in slot `N-3`
   - a `vc4.qpu.branch` incorrectly occupying one of the final two delay slots
7. Do not change the host structured emit contract in this prompt.
8. Do not invent new fragment / TLB concepts.

Build and test before finishing:
- `cmake -G Ninja -S . -B build-verify`
- if lit is not auto-discovered, retry with:
  - `cmake -G Ninja -S . -B build-verify -DLLVM_EXTERNAL_LIT="$(command -v llvm-lit || command -v lit)"`
- `ninja -C build-verify vc4-opt check-vc4`
- specifically run the `verify-emit-contract` tests

Deliverable:
- the code changes
- passing tests
- a short summary of the exact epilogue contract now enforced

---

## Verification prompt P2

You are verifying the repository state after Prompt P2 of the VC4 pre-codegen hardening plan.

Read:
- `AGENTS.md` if present
- `tools/vc4-opt/vc4-opt.cpp`
- `test/Dialect/VC4/verify-emit-contract.mlir`
- `test/Dialect/VC4/verify-emit-contract-invalid.mlir`

Expected state after Prompt P2:
1. The qasm-input side of `--vc4-verify-emit-contract` requires an explicit final `thrend` epilogue.
2. The required epilogue is exactly:
   - slot `N-3`: `vc4.qpu.bundle` with `sig = #vc4.qpu_signal<thrend>`
   - slots `N-2` and `N-1`: scheduled non-branch delay-slot ops
3. The pass rejects missing / misplaced / duplicated `thrend` cases.
4. The positive emit-contract test includes a valid `thrend` epilogue.
5. The host structured half of the emit contract has not been broadened or redesigned.

Your job:
- inspect the repository
- find mismatches and fix them directly
- configure/build/tests if possible
- when configuring, retry with explicit `LLVM_EXTERNAL_LIT` if needed
- run the targeted emit-contract tests and then `check-vc4`
- preserve consistent later work

---

## Prompt P3: add the remaining kernel-critical spacing rules (`UNIFORMS_ADDRESS` and `TMU_NOSWAP`)

You are working in the `compiler/` repo root from a cold session.

Read these files first:
- `AGENTS.md` if present
- `include/vc4/Dialect/VC4/IR/VC4QPURegisterInfo.h`
- `tools/vc4-opt/vc4-opt.cpp`
- `test/Dialect/VC4/qpu-scheduled-hardware-rules.mlir`
- `test/Dialect/VC4/qpu-scheduled-hardware-rules-invalid.mlir`
- `test/Dialect/VC4/qpu-scheduled-adjacent-hazards.mlir`
- `test/Dialect/VC4/qpu-scheduled-adjacent-hazards-invalid.mlir`

Goal:
Add the remaining sink-level spacing checks that are directly relevant to compute kernels and that the current tree does not yet check.

Do the following:
1. Add a new verifier pass in `vc4-opt.cpp` named
   `--vc4-verify-scheduled-io-spacing`
   unless you have a very strong reason to extend an existing pass instead.
2. This pass must operate only on non-external
   `domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>`
   functions, using the same flattened instruction-stream model already used by the other scheduled verifiers.
3. Enforce rule A:
   - if an instruction slot writes `UNIFORMS_ADDRESS` (`waddr = 40`), then the next **two** instruction slots must not perform a uniform read (`raddr = 32`)
4. Enforce rule B:
   - if an instruction slot writes `TMU_NOSWAP` (`waddr = 36`), then the first later TMU parameter write (`waddr in 56..63`) must be at least **three** instruction slots after the `TMU_NOSWAP` write
5. Use the shared helpers from `VC4QPURegisterInfo.h`; do not add more ad hoc magic numbers.
6. Add focused negative tests for both rules and at least one positive test for each.
7. Keep this prompt limited to spacing rules only. Do not fold in the “one peripheral access per instruction” rule here.
8. Do not add fragment / TLB / scoreboard features.

Build and test before finishing:
- `cmake -G Ninja -S . -B build-verify`
- if lit is not auto-discovered, retry with:
  - `cmake -G Ninja -S . -B build-verify -DLLVM_EXTERNAL_LIT="$(command -v llvm-lit || command -v lit)"`
- `ninja -C build-verify vc4-opt check-vc4`
- specifically run the new `scheduled-io-spacing` tests

Deliverable:
- the code changes
- passing tests
- a short summary of the exact spacing rules enforced

---

## Verification prompt P3

You are verifying the repository state after Prompt P3 of the VC4 pre-codegen hardening plan.

Read:
- `AGENTS.md` if present
- `include/vc4/Dialect/VC4/IR/VC4QPURegisterInfo.h`
- `tools/vc4-opt/vc4-opt.cpp`
- the new / edited `test/Dialect/VC4/*scheduled*` tests relevant to IO spacing

Expected state after Prompt P3:
1. There is a verifier pass for sink-level IO spacing.
2. The pass checks `UNIFORMS_ADDRESS` write -> next two slots must not do uniform read.
3. The pass checks `TMU_NOSWAP` write -> first TMU parameter write must be at least three slots later.
4. The implementation uses shared register helpers instead of new raw constants.
5. There are dedicated positive and negative tests for both rules.
6. No unrelated peripheral-access counting or fragment/TLB work was added in this prompt.

Your job:
- inspect the repository
- fix mismatches directly
- configure/build/tests if possible
- retry configure with explicit `LLVM_EXTERNAL_LIT` if needed
- run the targeted IO-spacing tests and then `check-vc4`
- preserve consistent later work

---

## Prompt P4: enforce the “one closely-coupled peripheral access per instruction” rule for the representable compute subset

You are working in the `compiler/` repo root from a cold session.

Read these files first:
- `AGENTS.md` if present
- `include/vc4/Dialect/VC4/IR/VC4QPURegisterInfo.h`
- `tools/vc4-opt/vc4-opt.cpp`
- `test/Dialect/VC4/qpu-bundle-invalid.mlir`
- `test/Dialect/VC4/qpu-ldi-sema-invalid.mlir`
- the new / edited scheduled verifier tests

Goal:
Add a conservative verifier for the hardware rule that one instruction must not perform more than one closely-coupled peripheral access, limited strictly to the currently representable compute subset.

Do the following:
1. Add a verifier pass in `vc4-opt.cpp` named
   `--vc4-verify-scheduled-peripheral-accesses`
   unless you have a compelling reason to extend an existing pass instead.
2. Limit the pass to the currently representable compute subset. Count these as closely-coupled peripheral accesses:
   - TMU read signal on `vc4.qpu.bundle` (`ldtmu0` / `ldtmu1`)
   - TMU parameter write (`waddr in 56..63`)
   - SFU write (`waddr in 52..55`)
   - mutex acquire read (`raddr = 51`) when encoded through a sink instruction that can represent it
   - semaphore access (`vc4.qpu.sema`)
   - VPM / VDR / VDW register-space access (`48..50`) if present in a scheduled sink instruction
3. Reject any single instruction slot that encodes more than one of those accesses.
4. Keep this verifier conservative and local. Do not invent TLB or fragment-shader behaviour.
5. Add focused tests for at least:
   - TMU read signal + SFU write in one `vc4.qpu.bundle`
   - TMU parameter write + mutex acquire read in one slot if representable
   - semaphore access combined with another access in the same slot if representable
   - at least one valid case that uses exactly one such access and passes
6. Reuse the shared register helpers.
7. Do not redesign the IR or add new sink ops.

Build and test before finishing:
- `cmake -G Ninja -S . -B build-verify`
- if lit is not auto-discovered, retry with:
  - `cmake -G Ninja -S . -B build-verify -DLLVM_EXTERNAL_LIT="$(command -v llvm-lit || command -v lit)"`
- `ninja -C build-verify vc4-opt check-vc4`
- specifically run the new peripheral-access tests

Deliverable:
- the code changes
- passing tests
- a short summary of exactly which sink-level access combinations are now rejected

---

## Verification prompt P4

You are verifying the repository state after Prompt P4 of the VC4 pre-codegen hardening plan.

Read:
- `AGENTS.md` if present
- `include/vc4/Dialect/VC4/IR/VC4QPURegisterInfo.h`
- `tools/vc4-opt/vc4-opt.cpp`
- the new / edited peripheral-access tests

Expected state after Prompt P4:
1. There is a sink-level verifier for the “one closely-coupled peripheral access per instruction” rule.
2. It is limited to the representable compute subset and does not invent fragment / TLB behaviour.
3. It uses shared register helpers.
4. There are focused positive and negative tests for the covered combinations.
5. The repo still builds cleanly and `check-vc4` passes.

Your job:
- inspect the repository
- fix mismatches directly
- configure/build/tests if possible
- retry configure with explicit `LLVM_EXTERNAL_LIT` if needed
- run the targeted peripheral-access tests and then `check-vc4`
- preserve consistent later work

---

## After P4 verifies

At that point, move to codegen.

Recommended initial codegen slice:
1. `vc4.func` (`domain=qpu`, `form=scheduled`) -> qasm printer
2. `vc4.func` (`domain=host`, `form=structured`) -> minimal C launcher generator
3. bundle manifest / `.h` declarations that tie the host launcher to the qasm kernel symbols
4. only then start lowering structured QPU ops into scheduled sink ops

