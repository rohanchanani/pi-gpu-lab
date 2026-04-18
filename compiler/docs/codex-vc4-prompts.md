# Codex Prompt Sequence for the first `vc4` dialect milestone

## How to use this file

1. Put `AGENTS.md`, `docs/vc4-dialect-spec.md`, and `docs/vc4-implementation-guide.md` into the repository first.
2. Run each implementation prompt in order from the repository root, including lettered prompts such as `3A`, `3B`, and so on.
3. After each implementation prompt completes, run the matching verification prompt in a **fresh** Codex session rooted at the same repository.
4. The verifier should inspect the files, run builds/tests when possible, and fix mismatches.
5. After a verification prompt succeeds, commit immediately. Prefer lightweight tags at major checkpoints, especially after Prompt 2.

**Verifier rule:** It is acceptable if the repository already contains consistent later-stage work. Do **not** delete consistent extra work merely because it is beyond the current prompt. Only fix mismatches against the required state for the prompt.

**Recovery rule:** If a later prompt causes any previously verified tests to fail, restore to the last verified commit/tag before retrying. Do not debug forward in a contaminated tree.

**Stability rule after Prompt 2:** Treat the Prompt 1/2 surface as stable. Do not rewrite the `vc4.module` / `vc4.func` / `vc4.return` / `vc4.builtin` parser-printer-verifier path unless a later prompt truly requires it.

**Test rule after Prompt 2:** Add new tests in new files for new prompt families. Keep Prompt 1/2 tests as regression guards.

**Assembly rule:** For new structured ops, generic assembly or conservative declarative assembly is acceptable initially. Do not introduce custom assembly unless the prompt explicitly requires it.

**Hardware-encoding rule:** When an enum models a real encoded QPU field, use the real hardware values from the datasheet, not dense placeholder numbering.

---

## Prompt 1 — bootstrap the standalone MLIR project

```text
You are working in the root of the `compiler` repository.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md

Task:
Bootstrap the repository as a standalone out-of-tree MLIR project for the first `vc4` dialect milestone.

Required outcomes:
1. Create a buildable CMake/Ninja project that uses installed LLVM/MLIR via `find_package`.
2. Create the top-level source layout needed for a dialect project:
   - include/
   - lib/
   - tools/vc4-opt/
   - test/
   - docs/ (preserve existing docs)
3. Add a minimal `vc4-opt` tool that links the project dialect library and can register the dialect.
4. Add lit/FileCheck test infrastructure and a `check-vc4` target.
5. Add the minimal shell of the `vc4` dialect library so the project can build even before most ops exist.
6. Do not implement lowering passes, codegen, or runtime integration.
7. Keep the repository narrow and clean: no fragment/TLB/control-list features.

Implementation guidance:
- Follow the standard standalone MLIR project pattern.
- Make lit discovery robust for standalone installs.
- Honor an explicitly passed `LLVM_EXTERNAL_LIT`.
- If it is not provided, detect either `llvm-lit` or `lit` in CMake with `find_program`.
- Do not hardcode a single machine-specific path into tracked source unless it is only a hint and not a requirement.
- Add only the smallest amount of code needed for the project to configure and build.
- It is fine if the initial dialect has little or no op surface yet, as long as the registration/build/test scaffolding exists.
- Prefer a stable directory layout that will support the remaining prompts.

At the end:
- update or create any minimal README/build notes only if needed
- run the build/tests if the environment allows
- fix build issues before finishing
```

### Verification prompt 1

```text
You are verifying the repository state after Prompt 1 of the VC4 dialect implementation plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- docs/codex-vc4-prompts.md

Expected state after Prompt 1:
1. The repo is a standalone out-of-tree MLIR project using CMake.
2. The root contains a valid `CMakeLists.txt`.
3. There is a dialect library skeleton for `vc4`.
4. There is a `tools/vc4-opt/` tool that registers or is prepared to register the dialect.
5. There is lit infrastructure under `test/`.
6. There is a `check-vc4` style test target or equivalent lit integration.
7. The repo still contains the docs and has not wandered into lowering/codegen/runtime work.
8. The codebase is still scoped to the dialect milestone only.

Your job:
- inspect the repository
- identify any mismatches versus the expected state above
- fix them directly
- when configuring the build, if lit is not auto-discovered, retry configure with
  `-DLLVM_EXTERNAL_LIT="$(command -v llvm-lit || command -v lit)"`
- if the repository still cannot support a working `check-vc4` configuration with either auto-discovery or that explicit override, treat it as a mismatch and fix it
- run configure/build/tests if possible
- do not delete consistent later-stage work if it already exists
- do not invent fragment/TLB/control-list features
```

---

## Prompt 2 — add dialect core, enums, types, and container ops

```text
You are continuing the first `vc4` dialect milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md

Task:
Implement the core dialect shell and the first real IR surface.

Required outcomes:
1. Add the main dialect TableGen and C++ registration structure.
2. Add the custom types from the spec:
   - !vc4.async.token
   - !vc4.tmu.desc
   - !vc4.vpm.desc
   - !vc4.dma.desc
3. Add the common enums needed by the initial op surface:
   - function form
   - threading mode
   - builtin kind
4. Implement the container/function ops:
   - vc4.module
   - vc4.func
   - vc4.return
   - vc4.builtin
5. Implement parser/printer support and verifiers for those types and ops.
6. Add tests:
   - parse/print tests for the custom types
   - parse/print tests for the container/function ops
   - verifier-negative tests for obvious illegal cases
7. Keep `vc4.func` attributes aligned with the spec:
   - kernel unit attr
   - threading enum
   - form enum

Do not implement arithmetic, TMU, VPM, DMA, or sink ops yet unless necessary boilerplate is shared.

At the end:
- build and run the relevant tests if possible
- fix failures before finishing
```

### Verification prompt 2

```text
You are verifying the repository state after Prompt 2 of the VC4 dialect implementation plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- docs/codex-vc4-prompts.md

Expected state after Prompt 2:
1. The core dialect shell exists and is registered.
2. The custom VC4 types exist:
   - !vc4.async.token
   - !vc4.tmu.desc
   - !vc4.vpm.desc
   - !vc4.dma.desc
3. The function/container enums exist for `form`, `threading`, and `builtin kind`.
4. The following ops exist and are registered:
   - vc4.module
   - vc4.func
   - vc4.return
   - vc4.builtin
5. These ops have parser/printer support and local verifiers.
6. There are tests covering:
   - type parse/print
   - op parse/print
   - verifier-negative cases
7. No arithmetic/TMU/VPM/DMA/sink op families are required yet, though consistent extra work is acceptable.

Your job:
- inspect the repository
- compare the implementation against the expected state
- fix any mismatches
- run build/tests if possible
- preserve any consistent later-stage work
- do not add new design beyond the spec
```

---

## Prompt 3A — add hardware-facing arithmetic/value-shape enums only

```text
You are continuing the first `vc4` dialect milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md

Task:
Add the enum and attr surface needed for the structured uniform/arithmetic/value-shape ops, without adding those ops yet.

Required outcomes:
1. Add enum/attr definitions for:
   - VC4 add opcodes
   - VC4 mul opcodes
   - VC4 cond codes
   - load-imm modes
   - regfile-A pack modes
   - regfile-A unpack modes
   - r4 unpack modes
   - MUL-pack modes
2. Where an enum models a real encoded QPU field, use the real hardware encoding values from the spec rather than dense placeholder numbering.
3. Register these attrs cleanly in the dialect.
4. Add minimal tests that prove the new attrs parse/print and build cleanly.

Guardrails:
- Do not add the new structured ops yet.
- Do not modify `vc4.module`, `vc4.func`, `vc4.return`, or `vc4.builtin` unless absolutely necessary for build integration.
- Do not rewrite the Prompt 2 parser/printer path.
- Add new tests in new files; do not repurpose Prompt 1/2 tests.

At the end:
- build and run the relevant tests if possible
- fix failures before finishing
```

### Verification prompt 3A

```text
You are verifying the repository state after Prompt 3A of the VC4 dialect implementation plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- docs/codex-vc4-prompts.md

Expected state after Prompt 3A:
1. All Prompt 2 functionality still exists and still passes.
2. The following enum/attr families now exist:
   - add opcodes
   - mul opcodes
   - cond codes
   - load-imm modes
   - regfile-A pack modes
   - regfile-A unpack modes
   - r4 unpack modes
   - MUL-pack modes
3. Encoded enums that correspond to hardware fields use the hardware encoding values, not dense placeholder numbering.
4. There are tests covering parse/print of the new attrs or enums.
5. No structured arithmetic/value-shape ops are required yet.

Your job:
- inspect the repository
- compare against the expected state
- fix any mismatches
- run the existing Prompt 1/2 tests and the new Prompt 3A tests if possible
- if Prompt 1/2 behavior regressed, treat that as a mismatch and restore stable behavior rather than debugging around it
```

---

## Prompt 3B — add uniforms and identity-like structured ops

```text
You are continuing the first `vc4` dialect milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md

Task:
Implement the lowest-risk structured uniform and pseudo/identity-style ops.

Required ops:
- vc4.uniform.read
- vc4.uniform.seek
- vc4.mov
- vc4.read

Required behavior:
1. Model these as structured SSA ops, not sink ops.
2. Use builtin scalar/vector types where appropriate.
3. Add local verifiers for obvious legality.
4. Keep `vc4.read` as an assembler-inspired pseudo op in structured form.
5. Keep `vc4.mov` distinct from `vc4.read` and distinct from `vc4.load_imm`, which is not part of this prompt.
6. Add round-trip parsing and negative verifier tests in new files.

Guardrails:
- Do not implement `vc4.alu.*`, `vc4.load_imm`, `vc4.pack`, `vc4.unpack`, or `vc4.rotate` yet.
- Do not rewrite the verified Prompt 2 `vc4.func` parser/printer or old tests unless absolutely necessary.
- Prefer declarative assembly or generic assembly. Do not add a custom parser/printer for these ops.

At the end:
- build and test if possible
- fix failures before finishing
```

### Verification prompt 3B

```text
You are verifying the repository state after Prompt 3B of the VC4 dialect implementation plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- docs/codex-vc4-prompts.md

Expected state after Prompt 3B:
1. All Prompt 2 and Prompt 3A functionality still exists and still passes.
2. The following ops now exist and are registered:
   - vc4.uniform.read
   - vc4.uniform.seek
   - vc4.mov
   - vc4.read
3. These ops have parser/printer support and local verifiers.
4. Tests cover round-trip parsing and negative verifier cases for this family.
5. `vc4.alu.*`, `vc4.load_imm`, `vc4.pack`, `vc4.unpack`, and `vc4.rotate` are not required yet.

Your job:
- inspect the repository
- compare against the expected state
- fix any mismatches
- run the relevant build/tests if possible
- if Prompt 1/2 tests or Prompt 3A tests regress, treat that as a mismatch and fix it
```

---

## Prompt 3C — add structured ALU ops and `vc4.load_imm`

```text
You are continuing the first `vc4` dialect milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md

Task:
Implement the structured ALU surface and `vc4.load_imm`.

Required ops:
- vc4.alu.add
- vc4.alu.mul
- vc4.load_imm

Required behavior:
1. Model the ops as structured SSA ops, not sink ops.
2. Add local verifiers for:
   - opcode/arity matching
   - integer/float type legality
   - scalar vs vector-shape compatibility
   - load-immediate payload/mode legality
3. Keep `vc4.load_imm` distinct from `vc4.mov`.
4. Use the enums from Prompt 3A rather than inventing new ad hoc attrs.
5. Add round-trip parsing and negative verifier tests in new files.

Guardrails:
- Do not implement `vc4.pack`, `vc4.unpack`, or `vc4.rotate` yet.
- Do not rewrite the verified Prompt 2 `vc4.func` parser/printer or old tests unless absolutely necessary.
- If declarative syntax for `vc4.load_imm` becomes unstable, generic assembly is acceptable for this prompt.
- Do not introduce scheduling, bundling, or sink ops.

At the end:
- build and test if possible
- fix failures before finishing
```

### Verification prompt 3C

```text
You are verifying the repository state after Prompt 3C of the VC4 dialect implementation plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- docs/codex-vc4-prompts.md

Expected state after Prompt 3C:
1. All earlier prompt functionality still exists and still passes.
2. The following ops now exist and are registered:
   - vc4.alu.add
   - vc4.alu.mul
   - vc4.load_imm
3. Local verifiers enforce obvious legality:
   - opcode/arity matching
   - type compatibility
   - load-immediate payload/mode legality
4. Tests cover round-trip parsing and negative verifier cases for this family.
5. `vc4.pack`, `vc4.unpack`, and `vc4.rotate` are not required yet.

Your job:
- inspect the repository
- compare against the expected state
- fix any mismatches
- run the relevant build/tests if possible
- if any earlier verified behavior regressed, treat that as a mismatch and restore it
```

---

## Prompt 3D — add structured `vc4.pack`, `vc4.unpack`, and `vc4.rotate`

```text
You are continuing the first `vc4` dialect milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md

Task:
Implement the remaining structured value-shape ops.

Required ops:
- vc4.pack
- vc4.unpack
- vc4.rotate

Required behavior:
1. Model the ops as structured SSA ops, not sink ops.
2. Add local verifiers for:
   - legal pack/unpack mode selection
   - shape compatibility
   - source/result type compatibility
   - rotate amount-operand vs immediate exclusivity
3. Use builtin scalar/vector types where appropriate.
4. Add round-trip parsing and negative verifier tests in new files.

Guardrails:
- Do not rewrite the verified Prompt 2 `vc4.func` parser/printer or old tests unless absolutely necessary.
- Conservative declarative assembly or generic assembly is acceptable if it keeps the repo stable.
- Do not add custom assembly printers/parsers for these ops unless they are truly required.
- Do not add sink ops in this prompt.

At the end:
- build and test if possible
- fix failures before finishing
```

### Verification prompt 3D

```text
You are verifying the repository state after Prompt 3D of the VC4 dialect implementation plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- docs/codex-vc4-prompts.md

Expected state after Prompt 3D:
1. All earlier prompt functionality still exists and still passes.
2. The following ops now exist and are registered:
   - vc4.pack
   - vc4.unpack
   - vc4.rotate
3. Local verifiers enforce obvious legality for mode selection, type compatibility, and rotate form legality.
4. Tests cover round-trip parsing and negative verifier cases for this family.
5. No sink ops are required yet.

Your job:
- inspect the repository
- compare against the expected state
- fix any mismatches
- run the relevant build/tests if possible
- if any earlier verified behavior regressed, treat that as a mismatch and restore it
```

---

## Prompt 4A — add TMU enums/attrs and `vc4.tmu.descriptor`

```text
You are continuing the first `vc4` dialect milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md

Task:
Implement the TMU descriptor surface first, before request/read ops.

Required outcomes:
1. Add TMU-related enums/attrs sufficient for the structured TMU surface:
   - TMU unit
   - TMU mode
   - texture type
   - filter enums
   - wrap mode
   - read-part selection if shared here
2. Implement `vc4.tmu.descriptor`.
3. Ensure `vc4.tmu.descriptor` produces `!vc4.tmu.desc`.
4. Add local verifiers for descriptor field combinations.
5. Add round-trip parsing and negative verifier tests in new files.

Guardrails:
- Do not implement `vc4.tmu.request`, `vc4.tmu.read`, or `vc4.tmu.noswap` yet unless tiny shared boilerplate is unavoidable.
- Keep the descriptor schema modest but sufficient to model the fields called out in the spec.
- Prefer attrs for constant descriptor fields rather than invasive custom syntax.

At the end:
- build and test if possible
- fix failures before finishing
```

### Verification prompt 4A

```text
You are verifying the repository state after Prompt 4A of the VC4 dialect implementation plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- docs/codex-vc4-prompts.md

Expected state after Prompt 4A:
1. All earlier prompt functionality still exists and still passes.
2. `vc4.tmu.descriptor` exists and is registered.
3. TMU-related enums/attrs sufficient for descriptor modeling now exist.
4. `vc4.tmu.descriptor` produces `!vc4.tmu.desc`.
5. Local verifiers reject illegal descriptor field combinations.
6. Tests cover round-trip parsing and negative cases for this surface.
7. `vc4.tmu.request`, `vc4.tmu.read`, and `vc4.tmu.noswap` are not required yet.

Your job:
- inspect the repository
- compare against the expected state
- fix mismatches
- run the relevant build/tests if possible
- preserve earlier verified behavior
```

---

## Prompt 4B — add `vc4.tmu.request`, `vc4.tmu.read`, and `vc4.tmu.noswap`

```text
You are continuing the first `vc4` dialect milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md

Task:
Implement the TMU request/read control surface.

Required ops:
- vc4.tmu.request
- vc4.tmu.read
- vc4.tmu.noswap

Required behavior:
1. `vc4.tmu.request` must support direct-address mode and texture-oriented modes from the spec.
2. `vc4.tmu.read` must support unit selection and result-part selection.
3. `vc4.tmu.request` should be compatible with `!vc4.async.token`.
4. `vc4.tmu.noswap` should exist as a distinct op.
5. Add parser/printer tests and verifier-negative tests in new files.

Guardrails:
- Build on the descriptor/enums from Prompt 4A rather than redesigning them.
- Do not implement SFU ops in this prompt.
- Prefer declarative or generic assembly first; do not add custom parser/printer unless clearly required.

At the end:
- build and test if possible
- fix failures before finishing
```

### Verification prompt 4B

```text
You are verifying the repository state after Prompt 4B of the VC4 dialect implementation plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- docs/codex-vc4-prompts.md

Expected state after Prompt 4B:
1. All earlier prompt functionality still exists and still passes.
2. The following TMU ops now exist and are registered:
   - vc4.tmu.request
   - vc4.tmu.read
   - vc4.tmu.noswap
3. Descriptor/result types line up with the spec.
4. Local verifiers reject illegal request/read combinations.
5. Tests cover round-trip parsing and negative cases for this family.
6. SFU ops are not required yet.

Your job:
- inspect the repository
- compare against the expected state
- fix mismatches
- run the relevant build/tests if possible
- preserve earlier verified behavior
```

---

## Prompt 4C — add `vc4.sfu.issue` and `vc4.sfu.read`

```text
You are continuing the first `vc4` dialect milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md

Task:
Implement the SFU structured surface.

Required ops:
- vc4.sfu.issue
- vc4.sfu.read

Required support:
- SFU kind enum

Required behavior:
1. `vc4.sfu.issue` and `vc4.sfu.read` should exist as distinct ops.
2. Add parser/printer tests and verifier-negative tests in new files.
3. Keep these as structured ops; do not model SFU latency or cross-instruction hazards yet.

Guardrails:
- Do not redesign TMU ops or earlier verified infrastructure.
- Prefer simple declarative or generic assembly.

At the end:
- build and test if possible
- fix failures before finishing
```

### Verification prompt 4C

```text
You are verifying the repository state after Prompt 4C of the VC4 dialect implementation plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- docs/codex-vc4-prompts.md

Expected state after Prompt 4C:
1. All earlier prompt functionality still exists and still passes.
2. `vc4.sfu.issue` and `vc4.sfu.read` now exist and are registered.
3. The SFU kind enum exists.
4. Tests cover round-trip parsing and negative cases for this family.

Your job:
- inspect the repository
- compare against the expected state
- fix mismatches
- run the relevant build/tests if possible
- preserve earlier verified behavior
```

---

## Prompt 5A — add `vc4.vpm.desc`, `vc4.vpm.read`, and `vc4.vpm.write`

```text
You are continuing the first `vc4` dialect milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md

Task:
Implement the VPM descriptor and read/write surface.

Required ops:
- vc4.vpm.desc
- vc4.vpm.read
- vc4.vpm.write

Required support:
- VPM descriptor enums:
  - kind
  - orientation
  - lane mode
  - element width

Required behavior:
1. `vc4.vpm.desc` must produce `!vc4.vpm.desc`.
2. `vc4.vpm.read/write` must verify descriptor kind matches op intent.
3. Add tests covering descriptor construction, read/write, and negative verifier cases in new files.

Guardrails:
- Do not implement DMA ops in this prompt.
- Prefer attrs for descriptor schema unless operands are clearly justified.
- Do not add sink ops in this prompt.

At the end:
- build and test if possible
- fix failures before finishing
```

### Verification prompt 5A

```text
You are verifying the repository state after Prompt 5A of the VC4 dialect implementation plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- docs/codex-vc4-prompts.md

Expected state after Prompt 5A:
1. All earlier prompt functionality still exists and still passes.
2. The following ops now exist and are registered:
   - vc4.vpm.desc
   - vc4.vpm.read
   - vc4.vpm.write
3. VPM descriptor enums exist and are wired into verifiers.
4. `vc4.vpm.desc` produces `!vc4.vpm.desc`.
5. Tests cover normal and illegal usage for this family.
6. DMA ops are not required yet.

Your job:
- inspect the repository
- compare against the expected state
- fix mismatches
- run the relevant build/tests if possible
- do not broaden the dialect beyond the documented scope
```

---

## Prompt 5B — add `vc4.dma.desc`, `vc4.dma.start`, `vc4.dma.status`, and `vc4.dma.wait`

```text
You are continuing the first `vc4` dialect milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md

Task:
Implement the DMA structured surface.

Required ops:
- vc4.dma.desc
- vc4.dma.start
- vc4.dma.status
- vc4.dma.wait

Required behavior:
1. `vc4.dma.desc` must produce `!vc4.dma.desc`.
2. DMA descriptor support must be sufficient to model the fields called out in the spec.
3. `vc4.dma.start` must be token-compatible for async use.
4. `vc4.dma.status` and `vc4.dma.wait` must exist as separate ops.
5. Add tests covering descriptor construction, start/status/wait, and negative verifier cases in new files.

Guardrails:
- Do not change VPM ops unless strictly necessary for shared descriptor/runtime typing.
- Do not implement sink ops in this prompt.
- Prefer conservative parser/printer choices over invasive custom syntax.

At the end:
- build and test if possible
- fix failures before finishing
```

### Verification prompt 5B

```text
You are verifying the repository state after Prompt 5B of the VC4 dialect implementation plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- docs/codex-vc4-prompts.md

Expected state after Prompt 5B:
1. All earlier prompt functionality still exists and still passes.
2. The following ops now exist and are registered:
   - vc4.dma.desc
   - vc4.dma.start
   - vc4.dma.status
   - vc4.dma.wait
3. DMA descriptor support exists and is sufficient to model the fields called out in the spec.
4. `vc4.dma.desc` produces `!vc4.dma.desc`.
5. Tests cover normal and illegal usage for this family.

Your job:
- inspect the repository
- compare against the expected state
- fix mismatches
- run the relevant build/tests if possible
- do not broaden the dialect beyond the documented scope
```

---

## Prompt 6A — add sync/thread ops and `vc4.async.wait`

```text
You are continuing the first `vc4` dialect milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md

Task:
Implement the synchronization and thread-control ops that do not require the host/system family.

Required ops:
- vc4.mutex
- vc4.semaphore
- vc4.host_interrupt
- vc4.thread_switch
- vc4.program_end
- vc4.async.wait

Required behavior:
1. `vc4.thread_switch` must verify against `vc4.func` threading mode.
2. `vc4.semaphore` must remain distinct from the future sink `vc4.qpu.sema`.
3. `vc4.async.wait` must accept the VC4 async token type used by TMU/DMA-style ops.
4. Add parse/print and negative tests for this family in new files.

Guardrails:
- Do not implement `vc4.cf.branch`, `vc4.enqueue_qpu`, `vc4.reserve_qpu`, `vc4.v3d.query`, or `vc4.v3d.configure` yet.
- Do not add side-effect resources in this prompt unless tiny shared boilerplate is truly necessary.

At the end:
- build and test if possible
- fix failures before finishing
```

### Verification prompt 6A

```text
You are verifying the repository state after Prompt 6A of the VC4 dialect implementation plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- docs/codex-vc4-prompts.md

Expected state after Prompt 6A:
1. All earlier prompt functionality still exists and still passes.
2. The following ops now exist and are registered:
   - vc4.mutex
   - vc4.semaphore
   - vc4.host_interrupt
   - vc4.thread_switch
   - vc4.program_end
   - vc4.async.wait
3. Thread-switch legality is checked against `vc4.func` threading mode.
4. Tests cover normal and illegal cases for this family.
5. `vc4.cf.branch` and host/system ops are not required yet.

Your job:
- inspect the repository
- compare against the expected state
- fix mismatches
- run the relevant build/tests if possible
- preserve earlier verified behavior
```

---

## Prompt 6B — add `vc4.cf.branch` and the host/system ops

```text
You are continuing the first `vc4` dialect milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md

Task:
Implement the remaining structured/system ops.

Required ops:
- vc4.cf.branch
- vc4.enqueue_qpu
- vc4.reserve_qpu
- vc4.v3d.query
- vc4.v3d.configure

Required support:
- structured branch condition enum
- system query/configure enums sufficient for the spec

Required behavior:
1. `vc4.cf.branch` should be a hardware-facing structured branch that branches on current flag state, not an SSA i1.
2. `vc4.enqueue_qpu` must reference a `vc4.func` symbol.
3. System query/configure ops should use enum-driven schemas and local verifiers.
4. Add parse/print and negative tests for this family in new files.

Guardrails:
- Do not add sink ops or side-effect resources in this prompt.
- Keep system op schemas local and verifier-driven rather than inventing pass infrastructure.

At the end:
- build and test if possible
- fix failures before finishing
```

### Verification prompt 6B

```text
You are verifying the repository state after Prompt 6B of the VC4 dialect implementation plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- docs/codex-vc4-prompts.md

Expected state after Prompt 6B:
1. All earlier prompt functionality still exists and still passes.
2. The following ops now exist and are registered:
   - vc4.cf.branch
   - vc4.enqueue_qpu
   - vc4.reserve_qpu
   - vc4.v3d.query
   - vc4.v3d.configure
3. `vc4.cf.branch` uses branch-condition attrs consistent with the spec.
4. Host/system ops have local schema validation.
5. Tests cover normal and illegal cases for this family.

Your job:
- inspect the repository
- compare against the expected state
- fix mismatches
- run the relevant build/tests if possible
- preserve earlier verified behavior
```

---

## Prompt 7 — add side-effect resources and function-form verification

```text
You are continuing the first `vc4` dialect milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md

Task:
Implement semantic infrastructure that the dialect now needs.

Required outcomes:
1. Implement `MemoryEffectOpInterface` for side-effecting VC4 ops.
2. Add the custom effect resources recommended by the spec:
   - UniformStream
   - MainMemory
   - TMUReq0 / TMUReq1
   - TMURcv0 / TMURcv1
   - SFU
   - VPMReadFIFO / VPMWriteFIFO
   - VDR / VDW
   - Mutex
   - Semaphore
   - HostIRQ
   - QPUScheduler
   - V3DSystem
3. Strengthen `vc4.func` verification so structured and scheduled forms cannot be mixed.
4. Add tests that specifically exercise side effects and function-form segregation.
5. Keep cross-instruction hazard analysis out of scope; do not add a scheduling pass here.

Guardrails:
- This prompt may strengthen `vc4.func` verification, but it should not rewrite the already verified Prompt 2 parser/printer path.
- Do not introduce lowering passes or scheduling passes.

At the end:
- build and test if possible
- fix failures before finishing
```

### Verification prompt 7

```text
You are verifying the repository state after Prompt 7 of the VC4 dialect implementation plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- docs/codex-vc4-prompts.md

Expected state after Prompt 7:
1. All earlier prompt functionality still exists and still passes.
2. `MemoryEffectOpInterface` is implemented for relevant ops.
3. The custom effect resources from the spec exist or are equivalently modeled.
4. `vc4.func` form verification prevents mixing structured and scheduled op families in one function.
5. Tests exist for:
   - effect/resource modeling
   - structured/scheduled segregation
   - thread-switch legality
6. No lowering passes or scheduling passes have been introduced.

Your job:
- inspect the repository
- compare against the expected state
- fix mismatches
- run build/tests if possible
- do not broaden the milestone into lowering/codegen
```

---

## Prompt 8A — add sink-level attrs plus `vc4.qpu.ldi` and `vc4.qpu.sema`

```text
You are continuing the first `vc4` dialect milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md

Task:
Implement the first two scheduled sink ops and the sink-level support they require.

Required ops:
- vc4.qpu.ldi
- vc4.qpu.sema

Required support:
- sink-level enums/attrs sufficient for these ops, including as needed:
  - signals
  - add/mul opcodes
  - conditions
  - pack/unpack
  - load-immediate mode
  - write swap / set flags
- any helper parser/printer code needed for readable sink syntax

Required behavior:
1. These ops must be explicit hardware-level ops with attrs, not SSA arithmetic wrappers.
2. `vc4.qpu.ldi` must model one real load-immediate instruction word.
3. `vc4.qpu.sema` must model one real semaphore instruction word.
4. Add local encoding verifiers for the rules that apply to these two sink ops.
5. Add parser/printer tests and negative verifier tests in new files.

Guardrails:
- Do not implement `vc4.qpu.bundle` or `vc4.qpu.branch` yet.
- Correct attrs and verifiers matter more than pretty custom syntax in this prompt.
- Reuse earlier hardware-facing enum encodings where appropriate instead of inventing duplicates.

At the end:
- build and test if possible
- fix failures before finishing
```

### Verification prompt 8A

```text
You are verifying the repository state after Prompt 8A of the VC4 dialect implementation plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- docs/codex-vc4-prompts.md

Expected state after Prompt 8A:
1. All earlier prompt functionality still exists and still passes.
2. The following sink ops now exist and are registered:
   - vc4.qpu.ldi
   - vc4.qpu.sema
3. Sink-level enums/attrs exist and are sufficient for these ops.
4. These ops use explicit low-level fields rather than structured SSA semantics.
5. Local encoding verifiers exist for the rules called out in the spec for these ops.
6. There are round-trip and negative tests for these sink ops.
7. `vc4.qpu.bundle` and `vc4.qpu.branch` are not required yet.

Your job:
- inspect the repository
- compare against the expected state
- fix mismatches
- run build/tests if possible
- preserve earlier verified behavior
```

---

## Prompt 8B — add `vc4.qpu.bundle`

```text
You are continuing the first `vc4` dialect milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md

Task:
Implement the main scheduled sink bundle op.

Required op:
- vc4.qpu.bundle

Required support:
- sink-level enums/attrs and helper logic sufficient for one ALU/small-immediate instruction word
- local encoding verifiers for:
  - mutually exclusive `raddr_b` vs small-immediate
  - legal pack/unpack combinations
  - local field legality

Required behavior:
1. `vc4.qpu.bundle` must model one ALU/small-immediate QPU instruction word.
2. It must be explicit hardware-level IR with attrs, not an SSA arithmetic wrapper.
3. Add parser/printer tests and negative verifier tests in new files.

Guardrails:
- Do not implement `vc4.qpu.branch` in this prompt.
- If readable custom syntax becomes invasive or risky, generic assembly is acceptable for this prompt.
- Correct field modeling and verifier behavior matter more than pretty syntax.

At the end:
- build and test if possible
- fix failures before finishing
```

### Verification prompt 8B

```text
You are verifying the repository state after Prompt 8B of the VC4 dialect implementation plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- docs/codex-vc4-prompts.md

Expected state after Prompt 8B:
1. All earlier prompt functionality still exists and still passes.
2. `vc4.qpu.bundle` now exists and is registered.
3. Sink-level enums/attrs exist and are sufficient for this op.
4. `vc4.qpu.bundle` uses explicit low-level fields rather than structured SSA semantics.
5. Local encoding verifiers exist for the rules called out in the spec.
6. There are round-trip and negative tests for this sink op.
7. `vc4.qpu.branch` is not required yet.

Your job:
- inspect the repository
- compare against the expected state
- fix mismatches
- run build/tests if possible
- preserve earlier verified behavior
```

---

## Prompt 9 — add `vc4.qpu.branch` and finalize scheduled-form legality

```text
You are continuing the first `vc4` dialect milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md

Task:
Implement the final sink op and complete scheduled-form legality for the milestone.

Required work:
1. Implement `vc4.qpu.branch`.
2. Model its delay slots as an explicit attached region.
3. Enforce the exact-three-delay-slot rule.
4. Ensure only scheduled-form functions may contain `vc4.qpu.*`.
5. Add tests for:
   - legal branch with exactly three delay-slot ops
   - illegal delay-slot counts
   - illegal structured/scheduled mixing involving branches
6. Keep cross-instruction hazard analysis out of scope.

Guardrails:
- If custom branch syntax is risky, a conservative region-based syntax is acceptable as long as the semantics and verifier are correct.
- Do not destabilize earlier structured/scheduled form verification while adding this final sink op.

At the end:
- build and test if possible
- fix failures before finishing
```

### Verification prompt 9

```text
You are verifying the repository state after Prompt 9 of the VC4 dialect implementation plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- docs/codex-vc4-prompts.md

Expected state after Prompt 9:
1. All earlier prompt functionality still exists and still passes.
2. `vc4.qpu.branch` now exists and is registered.
3. It has an explicit delay-slot region.
4. The verifier enforces exactly three delay-slot ops.
5. Scheduled-form legality is complete enough for the milestone:
   - `vc4.qpu.*` only in scheduled functions
   - structured ops rejected from scheduled functions
   - scheduled ops rejected from structured functions
6. There are tests covering both valid and invalid branch region structure.

Your job:
- inspect the repository
- compare against the expected state
- fix mismatches
- run build/tests if possible
- keep the implementation aligned to the spec
```

---

## Prompt 10 — final audit against the spec

```text
You are doing the final audit for the first `vc4` dialect milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- docs/codex-vc4-prompts.md

Task:
Audit the entire repository against the dialect spec and implementation guide.

Required outcomes:
1. Compare the implemented dialect surface against the full op list in the spec.
2. Compare the implemented types/attrs/enums against the spec.
3. Compare the tests against the required testing matrix in the spec.
4. Fill any missing gaps needed to claim the milestone is complete.
5. Synchronize docs only if the code now exposes something materially different but still correct for the milestone.
6. Do not add lowering/codegen/runtime work.
7. Do not reintroduce fragment/TLB/control-list/shader-state features.

Guardrails:
- Respect verified checkpoints. Do not destabilize earlier working parser/printer infrastructure merely to improve syntax aesthetics.

At the end:
- run the broadest reasonable build/test target available
- fix the remaining mismatches before finishing
```

### Verification prompt 10

```text
You are verifying the final repository state after Prompt 10 of the VC4 dialect implementation plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- docs/codex-vc4-prompts.md

Expected final state:
1. The repository is a standalone MLIR project.
2. The `vc4` dialect is registered and buildable.
3. `vc4-opt` exists and loads the dialect.
4. The custom types from the spec exist.
5. The full op surface from the compute/QPU-scoped spec exists:
   - vc4.module
   - vc4.func
   - vc4.return
   - vc4.builtin
   - vc4.uniform.read
   - vc4.uniform.seek
   - vc4.alu.add
   - vc4.alu.mul
   - vc4.mov
   - vc4.load_imm
   - vc4.pack
   - vc4.unpack
   - vc4.rotate
   - vc4.read
   - vc4.tmu.descriptor
   - vc4.tmu.request
   - vc4.tmu.read
   - vc4.tmu.noswap
   - vc4.sfu.issue
   - vc4.sfu.read
   - vc4.vpm.desc
   - vc4.vpm.read
   - vc4.vpm.write
   - vc4.dma.desc
   - vc4.dma.start
   - vc4.dma.status
   - vc4.dma.wait
   - vc4.mutex
   - vc4.semaphore
   - vc4.host_interrupt
   - vc4.thread_switch
   - vc4.program_end
   - vc4.async.wait
   - vc4.cf.branch
   - vc4.enqueue_qpu
   - vc4.reserve_qpu
   - vc4.v3d.query
   - vc4.v3d.configure
   - vc4.qpu.bundle
   - vc4.qpu.branch
   - vc4.qpu.ldi
   - vc4.qpu.sema
6. Parser/printer support exists for the dialect surface.
7. Local verifiers exist for important legality constraints.
8. Function-form segregation exists.
9. Side-effect resources are modeled.
10. Tests exist for round-trip parsing and negative verifier cases.
11. The repo has not drifted into lowering/codegen/runtime or graphics-fragment/TLB/control-list work.

Your job:
- inspect the repository
- compare against the expected final state
- fix mismatches
- run the broadest reasonable build/tests possible
- do not broaden the milestone beyond the documented dialect scope
```
