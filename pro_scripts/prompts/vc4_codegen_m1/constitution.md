# VC4 Codegen Milestone 1 Constitution

This document is the stable project contract for GPT Pro and Codex during VC4 codegen Milestone 1.

## Locked Milestone 1 goal

Given a `vc4.module` with exactly one kernel `vc4.func` where:

- `domain = #vc4.execution_domain<qpu>`
- `form = #vc4.function_form<scheduled>`
- the function is marked `kernel`
- the function has a valid `"vc4.launch_abi"` attribute
- the body contains only final scheduled QPU sink operations:
  - `vc4.qpu.bundle`
  - `vc4.qpu.ldi`
  - `vc4.qpu.sema`
  - `vc4.qpu.branch`

emit a candidate bundle containing:

- `kernel.qasm`
- `kernel_launch.c`
- `kernel_launch.h`
- `manifest.json`

The candidate bundle must build and run on the Raspberry Pi VC4 hardware for `minimal_thrend`-like and simple memory-output-like tests.

## Required architecture

Use a standalone artifact-emission tool backed by reusable compiler libraries:

```text
compiler/include/vc4/Target/VC4/**
compiler/lib/Target/VC4/**
compiler/tools/vc4-codegen/**
```

The intended user-facing command is:

```bash
vc4-codegen input.mlir --emit-bundle candidate/
```

The tool may internally run or require the existing VC4 verifier pipeline. The final emission boundary is a filesystem artifact boundary, not an ordinary IR-to-IR MLIR pass.

## Non-goals for Milestone 1

Do not implement any of the following unless a later slice explicitly asks for it:

- lowering from `gpu` dialect to `vc4`
- lowering from structured VC4 ops to scheduled `vc4.qpu.*` ops
- register allocation
- instruction scheduling
- hazard repair
- optimization
- multi-kernel module images
- broad runtime redesign
- exact textual matching of reference qasm as the pass criterion

## Reference and candidate rules

Reference bundles are hardware ground truth and are immutable during codegen implementation.

Do not mutate:

```text
compiler/test/CodeGen/VC4/Hardware/Run/*/reference/**
compiler/test/CodeGen/VC4/Hardware/Run/*/expected.json
compiler/test/CodeGen/VC4/catalog.json
```

unless the slice explicitly says otherwise. Milestone 1 slices do not say otherwise.

Candidate bundles are generated from `input.mlir`. Candidate success is checked against the same semantic oracle as the reference side, normally via `expected.json` and `VC4_TEST_RESULT`, not by exact qasm text comparison.

## Public launcher API rule

The generated public launcher API must be semantic.

It may expose user-level buffer/scalar arguments from `vc4.launch_abi`, but it must not expose raw uniform arrays or scheduler-internal details as public user arguments.

In particular, do not expose these as public user parameters unless a slice explicitly overrides this rule:

- `qpu_id`
- `num_qpus`
- raw uniform word arrays
- raw uniform pointer arrays

Uniform suffix builtins such as `qpu_id` and `num_qpus` are generated and packed internally by the launcher/runtime code.

## GPT Pro responsibility

GPT Pro owns substantive compiler design and semantic implementation:

- Target/VC4 codegen architecture
- qasm emitter behavior
- launch ABI model and launcher generation
- semantic/hardware failure diagnosis
- coherent patches for a single declared slice

GPT Pro must produce patches only for the current slice and must obey the allowed/forbidden path policy.

## Codex responsibility

Codex is only a constrained mechanical patcher. It may fix compile errors, include issues, CMake wiring, or narrow syntax problems after deterministic classification by the autorun script.

Codex must not diagnose or patch qasm semantics, launch ABI semantics, verifier semantics, runtime scheduling semantics, or hardware failures.

## Automation authority

The deterministic Python autorun script owns:

- slice order
- dependency checking
- context generation
- GPT prompt rendering
- patch validation
- allowed/forbidden path enforcement
- gate execution
- failure classification
- Codex dispatch
- state recording
- auto-commit after passing slices

Neither GPT Pro nor Codex may decide to broaden the milestone, skip gates, touch forbidden paths, or continue after a failed prerequisite.
