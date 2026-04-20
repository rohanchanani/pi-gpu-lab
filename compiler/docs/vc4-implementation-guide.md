# VC4 Dialect Implementation Guide

## 1. Goal of this guide

This guide is the practical companion to `docs/vc4-dialect-spec.md`.

The spec is the contract.  
This guide is the recommended way to map that contract onto code in a standalone MLIR project.

## 2. Recommended repository layout

```text
compiler/
  AGENTS.md
  CMakeLists.txt
  cmake/
  docs/
    vc4-dialect-spec.md
    vc4-implementation-guide.md
    codex-vc4-prompts.md
  include/
    vc4/
      Dialect/
        VC4/
          IR/
            VC4Dialect.h
            VC4Dialect.td
            VC4Enums.td
            VC4Types.h
            VC4Types.td
            VC4Ops.h
            VC4StructuredOps.td
            VC4MemoryOps.td
            VC4SystemOps.td
            VC4QPUOps.td
  lib/
    Dialect/
      VC4/
        IR/
          VC4Dialect.cpp
          VC4Types.cpp
          VC4Ops.cpp
          VC4StructuredOps.cpp
          VC4MemoryOps.cpp
          VC4SystemOps.cpp
          VC4QPUOps.cpp
          VC4SideEffects.cpp
  tools/
    vc4-opt/
      vc4-opt.cpp
  test/
    lit.cfg.py
    lit.site.cfg.py.in
    Dialect/
      VC4/
        ...
```

This exact tree is not mandatory, but the separation between public headers, dialect IR sources, tool registration,
and lit tests should remain clear.

## 3. Build system recommendation

Use the standard standalone MLIR project pattern:

- `find_package(LLVM REQUIRED CONFIG)`
- `find_package(MLIR REQUIRED CONFIG)`
- include LLVM and MLIR CMake helper modules
- generate TableGen headers/sources
- build one dialect library
- build `vc4-opt`
- make lit discovery robust for standalone installs:
  - honor an explicitly passed `LLVM_EXTERNAL_LIT`
  - otherwise detect either `llvm-lit` or `lit`
  - keep manual configure with `-DLLVM_EXTERNAL_LIT=/path/to/lit` supported
- add a `check-vc4` lit target

Do not vendor LLVM/MLIR into this repo for the first milestone.

## 4. File split recommendation

### 4.1 TableGen files

Use separate `.td` files by concern:

- `VC4Dialect.td`: dialect definition and common bases
- `VC4Enums.td`: all enum definitions
- `VC4Types.td`: `!vc4.async.token`, `!vc4.tmu.desc`, `!vc4.vpm.desc`, `!vc4.dma.desc`
- `VC4StructuredOps.td`: container, builtin, uniform, arithmetic, control-flow ops
- `VC4MemoryOps.td`: TMU, SFU, VPM, DMA ops
- `VC4SystemOps.td`: mutex, semaphore, host interrupt, thread ops, enqueue/reserve/query/configure
- `VC4QPUOps.td`: `vc4.qpu.*` sink ops

### 4.2 C++ files

Keep custom logic localized:

- `VC4Dialect.cpp`: dialect registration, type parsing hooks, op registration
- `VC4Types.cpp`: custom types
- `VC4Ops.cpp`: shared helpers
- `VC4StructuredOps.cpp`: verifiers/printers not expressible in ODS
- `VC4MemoryOps.cpp`: TMU/SFU/VPM/DMA custom logic
- `VC4SystemOps.cpp`: system op custom logic
- `VC4QPUOps.cpp`: sink op parser/printer + encoding verifiers
- `VC4SideEffects.cpp`: custom resource declarations and `MemoryEffectOpInterface` helpers

## 5. Implementation order

Recommended order:

1. project scaffold and `vc4-opt`
2. dialect shell, enums, types
3. container/function ops
4. arithmetic/value-shape encoded enums only
5. uniform + move/pseudo-read ops
6. ALU ops + `vc4.load_imm`
7. `vc4.pack` / `vc4.unpack` / `vc4.rotate`
8. TMU descriptor surface
9. TMU request/read/noswap
10. SFU ops
11. VPM descriptor/read/write
12. DMA descriptor/start/status/wait
13. sync/thread ops
14. structured branch + host/system ops
15. side-effect resources and form verifiers
16. sink `vc4.qpu.ldi` + `vc4.qpu.sema`
17. sink `vc4.qpu.bundle`
18. sink `vc4.qpu.branch`
19. final audit

That order is intentionally finer-grained than a purely feature-family order. The goal is to keep the tool parseable,
keep each prompt small enough to verify in isolation, and avoid contaminating already-verified infrastructure.

## 6. Checkpoint and recovery discipline

After every successful verification prompt:

- commit immediately
- prefer lightweight tags for important boundaries, especially the first verified state after Prompt 2
- treat the tagged state as the recovery point for later prompts

If a later prompt causes a previously verified prompt to regress:

- restore to the last verified commit/tag
- do not debug forward in a contaminated tree
- reattempt with a smaller prompt or narrower change set

This is especially important once `vc4.func`, `vc4.module`, and their tests have been verified.

## 7. Verified-surface stability rules

Once Prompt 2 verifies successfully, treat the following as stable infrastructure:

- `vc4.module`
- `vc4.func`
- `vc4.return`
- `vc4.builtin`
- the Prompt 1/2 parser/printer path
- the Prompt 1/2 tests

Later prompts should add new ops, enums, attrs, verifiers, and tests without rewriting the already verified
container/function surface.

Do not change the `vc4.func` parser/printer or repurpose old Prompt 1/2 tests unless a later prompt truly
requires it. If such a change is unavoidable, all earlier tests must still pass unchanged.

Function-level hardening should prefer adding enum attrs such as execution-domain
distinctions and verifier checks over rewriting the core `vc4.func` parser/printer.

## 8. Parser/printer strategy

Use declarative assembly format for most structured ops, but do not overvalue pretty syntax early.

Good candidates for declarative formats:

- `vc4.builtin`
- `vc4.uniform.read`
- `vc4.uniform.seek`
- `vc4.alu.add`
- `vc4.alu.mul`
- `vc4.mov`
- `vc4.read`
- `vc4.mutex`
- `vc4.semaphore`
- `vc4.thread_switch`
- `vc4.program_end`
- `vc4.async.wait`
- `vc4.enqueue_qpu`
- `vc4.reserve_qpu`
- `vc4.v3d.query`
- `vc4.v3d.configure`

Use custom assembly only where it is clearly justified:

- `vc4.qpu.bundle`
- `vc4.qpu.branch`
- `vc4.qpu.ldi`
- `vc4.qpu.sema`

For higher-risk structured ops, generic assembly or conservative declarative assembly is acceptable in the first
implementation if it keeps the repo stable:

- `vc4.load_imm`
- `vc4.pack`
- `vc4.unpack`
- `vc4.rotate`
- TMU descriptor/request/read ops
- VPM and DMA descriptor ops

Correct registration, verifier behavior, and testability matter more than aesthetic syntax in milestone one.

## 9. Enum and encoding guidance

When an enum or attr models a real QPU-encoded field, use the hardware encoding values from the spec rather
than dense placeholder numbering.

This applies in particular to:

- ADD opcodes
- MUL opcodes
- condition codes
- load-immediate modes when directly encoding hardware instruction forms
- pack/unpack selections
- signal fields
- other sink-level fields that correspond directly to instruction bits

If an enum is a structured-dialect convenience rather than a direct encoding, document that clearly in code.

## 10. Verifier boundaries

Keep verifier responsibilities narrow.

### 10.1 Local verifiers should enforce

- enum validity
- attribute combination legality
- operand/result type compatibility
- descriptor mode/schema correctness
- structured vs scheduled op segregation
- exact 3-delay-slot rule for `vc4.qpu.branch`

### 10.2 Do not over-implement in milestone one

Do not try to solve all global scheduling hazards yet.

Examples to defer:

- regfile write/read next-instruction hazard
- SFU `r4` latency window
- TMU_NOSWAP placement distance
- end-of-program trailing hazards
- vector-rotate dependency hazards

The spec requires the dialect to acknowledge those hazards, not to fully solve them now.

## 11. Side-effect modeling recommendation

Implement `MemoryEffectOpInterface` using custom resources.

Suggested mapping:

- `vc4.uniform.read` / `vc4.uniform.seek` -> `UniformStream`
- `vc4.tmu.request` -> `TMUReq0` / `TMUReq1` and `MainMemory`
- `vc4.tmu.read` -> `TMURcv0` / `TMURcv1`
- `vc4.sfu.issue` / `vc4.sfu.read` -> `SFU`
- `vc4.vpm.read` -> `VPMReadFIFO`
- `vc4.vpm.write` -> `VPMWriteFIFO`
- `vc4.dma.start` / `status` / `wait` -> `VDR` / `VDW` and `MainMemory`
- `vc4.mutex` -> `Mutex`
- `vc4.semaphore` -> `Semaphore`
- `vc4.host_interrupt` -> `HostIRQ`
- `vc4.enqueue_qpu` / `vc4.reserve_qpu` -> `QPUScheduler`
- `vc4.v3d.query` / `vc4.v3d.configure` -> `V3DSystem`

## 12. Testing matrix

Every op family should have:

- one parse/print round-trip test
- at least one negative verifier test
- one example using the syntax actually implemented in the repo

Add dedicated tests for:

- `vc4.func` `form = structured` rejecting `vc4.qpu.*`
- `vc4.func` `form = scheduled` rejecting structured ops
- `vc4.thread_switch` rejected in `threading = single`
- `vc4.qpu.branch` delay-slot count rules
- `vc4.qpu.bundle` local encoding legality

Testing discipline for later prompts:

- keep Prompt 1/2 tests as regression guards
- add new tests in new files for new prompt families
- do not repurpose earlier tests just because a later prompt needs new coverage

## 13. Definition of done

The milestone is done when:

- `vc4-opt` builds
- the dialect registers correctly
- all ops in the spec exist
- custom types exist
- parsers/printers work
- verifiers work
- tests are present and passing
- docs still describe the implemented surface accurately

## 14. What to avoid

Avoid these common mistakes:

- reintroducing graphics-only fragment/TLB/control-list IR
- letting structured and scheduled forms coexist in one function
- rewriting already verified parser/printer infrastructure during later prompts without necessity
- using raw integer attrs where a clear enum attr is available
- using dense placeholder enum values for hardware-encoded fields
- depending on downstream codegen decisions in the dialect layer
- inventing lowering passes as part of the dialect milestone
