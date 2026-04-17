# Codex Prompt Sequence for the first `vc4` dialect milestone

## How to use this file

1. Put `AGENTS.md`, `docs/vc4-dialect-spec.md`, and `docs/vc4-implementation-guide.md` into the repository first.
2. Run each implementation prompt in order from the repository root.
3. After each implementation prompt completes, run the matching verification prompt in a **fresh** Codex session rooted at the same repository.
4. The verifier should inspect the files, run builds/tests when possible, and fix mismatches.

**Verifier rule:** It is acceptable if the repository already contains consistent later-stage work. Do **not** delete consistent extra work merely because it is beyond the current prompt. Only fix mismatches against the required state for the prompt.

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

## Prompt 3 — add uniforms and structured arithmetic/value-shape ops

```text
You are continuing the first `vc4` dialect milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md

Task:
Implement the structured uniform and arithmetic/value-shape surface.

Required ops:
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

Required enum/attr support:
- VC4 add opcodes
- VC4 mul opcodes
- VC4 cond codes
- load-imm modes
- pack/unpack mode enums needed by these ops

Required behavior:
1. Model the ops as structured SSA ops, not sink ops.
2. Add local verifiers for opcode/arity/type legality.
3. Keep `vc4.read` as an assembler-inspired pseudo op in structured form.
4. Keep `vc4.mov` distinct from `vc4.load_imm`.
5. Use builtin scalar/vector types where appropriate.
6. Add assembly tests and verifier-negative tests.

Do not implement bundling, scheduling, or qpu sink ops in this prompt.

At the end:
- build and test if possible
- fix failures before finishing
```

### Verification prompt 3

```text
You are verifying the repository state after Prompt 3 of the VC4 dialect implementation plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- docs/codex-vc4-prompts.md

Expected state after Prompt 3:
1. All Prompt 2 functionality still exists.
2. The following ops now exist and are registered:
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
3. The arithmetic-related enums and attrs exist:
   - add opcodes
   - mul opcodes
   - cond codes
   - load-imm modes
   - pack/unpack enums required by these ops
4. Local verifiers enforce obvious legality:
   - opcode/arity matching
   - type compatibility
   - rotate operand/immediate form legality
   - load-immediate payload/mode legality
5. Tests cover round-trip parsing and negative verifier cases.
6. Sink ops are not required yet.

Your job:
- inspect the repository
- compare against the expected state
- fix any mismatches
- run build/tests if possible
- keep the implementation aligned to the spec
```

---

## Prompt 4 — add TMU and SFU ops

```text
You are continuing the first `vc4` dialect milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md

Task:
Implement the TMU and SFU structured ops.

Required ops:
- vc4.tmu.descriptor
- vc4.tmu.request
- vc4.tmu.read
- vc4.tmu.noswap
- vc4.sfu.issue
- vc4.sfu.read

Required support:
- TMU enums and attrs from the spec
- SFU kind enum
- any descriptor attrs needed for constant TMU configuration

Required behavior:
1. `vc4.tmu.descriptor` must produce `!vc4.tmu.desc`.
2. `vc4.tmu.request` must support direct-address mode and texture-oriented modes from the spec.
3. `vc4.tmu.read` must support unit selection and result-part selection.
4. `vc4.tmu.request` and `vc4.dma.start` (when implemented later) should be compatible with `!vc4.async.token`.
5. `vc4.sfu.issue` and `vc4.sfu.read` should exist as distinct ops.
6. Add parser/printer tests and verifier-negative tests.

Do not implement VPM/DMA or sink ops in this prompt unless required shared boilerplate already exists.

At the end:
- build and test if possible
- fix failures before finishing
```

### Verification prompt 4

```text
You are verifying the repository state after Prompt 4 of the VC4 dialect implementation plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- docs/codex-vc4-prompts.md

Expected state after Prompt 4:
1. All earlier prompt functionality still exists.
2. The following TMU/SFU ops now exist and are registered:
   - vc4.tmu.descriptor
   - vc4.tmu.request
   - vc4.tmu.read
   - vc4.tmu.noswap
   - vc4.sfu.issue
   - vc4.sfu.read
3. TMU-related enums/attrs exist:
   - TMU unit
   - TMU mode
   - texture type
   - filter enums
   - wrap mode
   - read part selection
4. SFU kind enum exists.
5. Descriptor/result types line up with the spec.
6. Local verifiers reject illegal descriptor/operand combinations.
7. There are round-trip and negative tests for this family.

Your job:
- inspect the repository
- compare against the expected state
- fix mismatches
- run build/tests if possible
- preserve any consistent later-stage work
```

---

## Prompt 5 — add VPM and DMA ops

```text
You are continuing the first `vc4` dialect milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md

Task:
Implement the VPM and DMA structured ops.

Required ops:
- vc4.vpm.desc
- vc4.vpm.read
- vc4.vpm.write
- vc4.dma.desc
- vc4.dma.start
- vc4.dma.status
- vc4.dma.wait

Required support:
- VPM descriptor enums:
  - kind
  - orientation
  - lane mode
  - element width
- DMA descriptor enums and fields sufficient for the spec

Required behavior:
1. `vc4.vpm.desc` must produce `!vc4.vpm.desc`.
2. `vc4.dma.desc` must produce `!vc4.dma.desc`.
3. `vc4.vpm.read/write` must verify descriptor kind matches op intent.
4. `vc4.dma.start` must be token-compatible for async use.
5. `vc4.dma.status` and `vc4.dma.wait` must exist as separate ops.
6. Add tests covering descriptor construction, read/write/start/status/wait, and negative verifier cases.

Do not implement sink ops in this prompt.

At the end:
- build and test if possible
- fix failures before finishing
```

### Verification prompt 5

```text
You are verifying the repository state after Prompt 5 of the VC4 dialect implementation plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- docs/codex-vc4-prompts.md

Expected state after Prompt 5:
1. All earlier prompt functionality still exists.
2. The following ops now exist and are registered:
   - vc4.vpm.desc
   - vc4.vpm.read
   - vc4.vpm.write
   - vc4.dma.desc
   - vc4.dma.start
   - vc4.dma.status
   - vc4.dma.wait
3. VPM descriptor enums exist and are wired into verifiers.
4. DMA descriptor support exists and is sufficient to model the fields called out in the spec.
5. Descriptor result types and use sites line up with the spec.
6. Tests cover normal and illegal usage.

Your job:
- inspect the repository
- compare against the expected state
- fix mismatches
- run build/tests if possible
- do not broaden the dialect beyond the documented scope
```

---

## Prompt 6 — add sync, thread-control, structured branch, and host/system ops

```text
You are continuing the first `vc4` dialect milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md

Task:
Implement the remaining structured/system ops.

Required ops:
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

Required support:
- mutex/semaphore/thread-switch enums
- structured branch condition enum
- system query/configure enums sufficient for the spec

Required behavior:
1. `vc4.thread_switch` must verify against `vc4.func` threading mode.
2. `vc4.cf.branch` should be a hardware-facing structured branch that branches on current flag state, not an SSA i1.
3. `vc4.enqueue_qpu` must reference a `vc4.func` symbol.
4. System query/configure ops should use enum-driven schemas and local verifiers.
5. Add parse/print and negative tests for this family.

Do not implement sink ops or side-effect resources in this prompt unless small shared boilerplate is needed.

At the end:
- build and test if possible
- fix failures before finishing
```

### Verification prompt 6

```text
You are verifying the repository state after Prompt 6 of the VC4 dialect implementation plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- docs/codex-vc4-prompts.md

Expected state after Prompt 6:
1. All earlier prompt functionality still exists.
2. The following ops now exist and are registered:
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
3. Thread-switch legality is checked against `vc4.func` threading mode.
4. `vc4.cf.branch` uses branch-condition attrs consistent with the spec.
5. Host/system ops have local schema validation.
6. Tests cover normal and illegal cases.

Your job:
- inspect the repository
- compare against the expected state
- fix mismatches
- run build/tests if possible
- preserve any consistent later-stage work
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
1. All earlier prompt functionality still exists.
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

## Prompt 8 — add sink ops `vc4.qpu.bundle`, `vc4.qpu.ldi`, and `vc4.qpu.sema`

```text
You are continuing the first `vc4` dialect milestone.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md

Task:
Implement the first three scheduled sink ops.

Required ops:
- vc4.qpu.bundle
- vc4.qpu.ldi
- vc4.qpu.sema

Required support:
- sink-level enums/attrs for:
  - signals
  - add/mul opcodes
  - conditions
  - pack/unpack
  - load-immediate mode
  - write swap / set flags
  - mux selections
- any helper parser/printer code needed for readable sink syntax

Required behavior:
1. These ops must be explicit hardware-level ops with attrs, not SSA arithmetic wrappers.
2. `vc4.qpu.bundle` must model one ALU/small-immediate instruction word.
3. `vc4.qpu.ldi` must model one real load-immediate instruction word.
4. `vc4.qpu.sema` must model one real semaphore instruction word.
5. Add local encoding verifiers:
   - mutually exclusive `raddr_b` vs small-immediate
   - legal pack/unpack mode combinations
   - local field legality
6. Add parser/printer tests and negative verifier tests.

Do not implement `vc4.qpu.branch` in this prompt.

At the end:
- build and test if possible
- fix failures before finishing
```

### Verification prompt 8

```text
You are verifying the repository state after Prompt 8 of the VC4 dialect implementation plan.

Read:
- AGENTS.md
- docs/vc4-dialect-spec.md
- docs/vc4-implementation-guide.md
- docs/codex-vc4-prompts.md

Expected state after Prompt 8:
1. All earlier prompt functionality still exists.
2. The following sink ops now exist and are registered:
   - vc4.qpu.bundle
   - vc4.qpu.ldi
   - vc4.qpu.sema
3. Sink-level enums/attrs exist and are sufficient for these ops.
4. These ops use explicit low-level fields rather than structured SSA semantics.
5. Local encoding verifiers exist for the rules called out in the spec.
6. There are round-trip and negative tests for these sink ops.
7. `vc4.qpu.branch` is not required yet.

Your job:
- inspect the repository
- compare against the expected state
- fix mismatches
- run build/tests if possible
- preserve any consistent later-stage work
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
1. All earlier prompt functionality still exists.
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
- compare it to the expected final state above
- fix any remaining mismatches
- run build/tests if possible
- leave the repo in a state that can honestly claim completion of the first `vc4` dialect milestone
```