# VC4 Feature-Driven Vertical Workflow

**Path:** `compiler/docs/vc4_feature_driven_vertical_workflow.md`  
**Status:** Standing methodology for future value/TTIR feature phases  
**Audience:** Codex/AI agents and prompt-package authors working on `rohanchanani/pi-gpu-lab`  
**Applies to:** all new user-visible value-layer and TTIR/Triton feature work after the value/TTIR bridge exists

---

## 1. Purpose

This document specifies the feature-driven vertical workflow for the VC4 compiler project.

The goal is to prevent future phases from drifting into one of two bad patterns:

```text
Bad pattern A:
  implement a large value abstraction far ahead of real Triton evidence

Bad pattern B:
  accept whatever an exploratory Triton probe happened to emit, then let
  unrelated body features hijack the phase
```

The correct pattern is:

```text
choose the feature
  -> design controlled fixtures for that feature
  -> inspect real emitted TTIR
  -> implement exactly the value support needed
  -> prove value support on hardware
  -> implement TTIR lowering for the controlled TTIR forms
  -> prove TTIR support on hardware
  -> lock mixed acceptance and move on
```

This workflow keeps the compiler vertical, hardware-proven, and relevant to real Triton without allowing exploratory probes or fixture names to become the implementation spec.

---

## 2. Foundational invariants

All phases following this workflow must preserve the project stack:

```text
Triton source
  -> real emitted TTIR / tt dialect
  -> standard value layer
       func + tiny vc4value + vector + memref + arith + math + scf/cf
  -> vc4kernel
  -> ssavc4
  -> scheduled vc4
  -> artifacts/runtime
  -> real VC4 hardware
```

Hard invariants:

- No direct TTIR/Triton -> `vc4kernel` as the accepted frontend path.
- No direct value/TTIR -> `ssavc4` or scheduled `vc4`.
- No producer dialect ops inside verified `vc4kernel`.
- `vc4value` remains tiny launch/policy plumbing.
- Real hardware is required for executable semantic claims.
- Fixture names, paths, public names, and status strings must never decide compiler semantics.
- Prompt packages must stop with `FAILURE` or `BLOCKED` when the correct long-term path is unclear or outside scope.

---

## 3. Terminology

### 3.1 Feature

A **feature** is a coherent source-visible or value-visible capability that can be scoped, tested, and hardware-proven.

Examples:

```text
scalar/coherent control flow
mask classification and memory legality
strided transfer reads
gather loads
reductions
math/SFU policy
subword/f16 storage
shuffle/transpose/rotate
vector.contract / dot
block pointers
tensor descriptors
```

A feature phase may include a tightly coupled feature band, but only if the package explicitly names the whole band and proves the whole band.

### 3.2 Exploratory probe

An **exploratory probe** is a real Triton source program used to learn what Triton emits.

Exploratory probes answer:

```text
What TTIR ops/regions/attrs/types does real Triton produce?
What value-layer forms will eventually be needed?
What adjacent features show up unexpectedly?
What should the taxonomy classify as supported/staged/rejected?
```

Exploratory probes may:

- live under `.vc4_auto`;
- use real Triton source;
- emit real TTIR;
- include unrelated body features;
- be used to guide design.

Exploratory probes must not automatically become acceptance fixtures.

### 3.3 Controlled fixture

A **controlled fixture** is a hand-designed source-controlled test fixture for a specific phase feature.

Controlled fixtures:

- target exactly the current feature or feature band;
- avoid unrelated staged body features unless explicitly in scope;
- are used for static tests, hardware isolation, and mixed acceptance;
- must be broad enough to prevent overfitting;
- must be accompanied by audits that reject fixture-name special casing.

For TTIR phases, controlled fixtures include:

```text
source-controlled Triton .py
source-controlled real emitted .ttir.mlir snapshot
manifest with generator command/version/provenance
CPU oracle / sentinels / expected-result contract for hardware fixtures
```

### 3.4 Isolation fixture

An **isolation fixture** targets one new feature shape or edge case.

Isolation fixtures are used to:

- prove the new feature in focused forms;
- break the implementation with edge cases;
- triage later regressions.

They are not the routine cumulative final acceptance suite.

### 3.5 Mixed fixture

A **mixed fixture** combines the new feature with multiple already-supported feature families.

Mixed fixtures are the cumulative acceptance suite. They should look increasingly like real kernels and should create feature interactions likely to expose bugs.

---

## 4. Core workflow

For every new user-visible feature, use this sequence unless the user explicitly approves a narrower repair or docs-only phase.

```text
Phase X0  Rebaseline and feature selection
Phase Xa  Exploratory probe and controlled fixture design
Phase Xb  Value-surface delta contract
Phase Xc  Value -> vc4kernel implementation and static lock
Phase Xd  Value hardware isolation
Phase Xe  Value mixed acceptance
Phase Xf  TTIR -> value implementation and static lock
Phase Xg  TTIR hardware isolation
Phase Xh  TTIR mixed acceptance and layered regression
Phase Xi  Final lock and readiness for the next feature
Phase Xx  Failure triage / repair prompts
```

Prompt packages are dynamic. If an early prompt changes the facts, later prompts may require addenda, prepends, or replacement. Do not run later prompts mechanically when prerequisites no longer match.

---

## 5. Phase X0 — rebaseline and feature selection

### Purpose

Confirm the repo is clean, prior phases are locked, and the next feature is explicitly named.

### Inputs

- Prior phase final lock doc.
- Current support matrices.
- Current mixed acceptance manifests.
- Current AGENTS.md and this document.
- Any user-provided feature goal.

### Required tasks

1. Capture git state.
2. Verify prior phase lock lines.
3. Verify `READY_FOR_TRITON=NO` remains unless a global readiness phase changed it.
4. Name the feature or feature band.
5. Define initial accepted/staged/rejected boundaries.
6. Identify likely layers touched.
7. Run static baseline checks.

### Required outputs

```text
PHASEX0_RESULT=LOCKED or FAILURE
FEATURE_NAME=<name>
FEATURE_SCOPE_LOCKED=YES or NO
PRIOR_PHASE_LOCK_VERIFIED=YES or NO
STATIC_BASELINE=PASS or FAIL
READY_FOR_PHASEXA_FIXTURE_DESIGN=YES or NO
READY_FOR_TRITON=NO
```

### Failure policy

Stop if the prior phase is not locked, the repo is dirty, the feature is ambiguous, or baseline checks fail.

---

## 6. Phase Xa — exploratory probe and controlled fixture design

This phase may be split into two prompts when the feature is complex:

```text
Xa1 exploratory TTIR inventory
Xa2 controlled fixture design
```

### Purpose

Use real Triton evidence to understand emitted TTIR, then design controlled fixtures that test the feature without importing unrelated features.

### Exploratory probe tasks

1. Write or locate small exploratory Triton probes.
2. Generate real TTIR under `.vc4_auto`.
3. Parse/roundtrip generated TTIR with the C++ TTIR lane.
4. Inventory:
   - `tt.*`
   - `scf.*`
   - `cf.*`
   - `arith.*`
   - `math.*`
   - tensors/vectors/types
   - attrs
   - backend dialects
   - adjacent unsupported body features.
5. Classify each emitted form.

Exploratory probes do not need hardware proof and should not become acceptance fixtures unless deliberately promoted.

### Controlled fixture design tasks

Design source-controlled fixtures for the phase:

```text
value isolation fixtures
value mixed fixture updates
Triton/TTIR isolation fixtures
Triton/TTIR mixed fixture updates
```

For each controlled fixture, specify:

- feature shape tested;
- allowed body features;
- forbidden body features;
- expected emitted TTIR forms;
- expected value IR forms;
- edge cases;
- hardware oracle design;
- sentinel strategy;
- required `saw_*` claims;
- active QPU expectations.

### Required outputs

```text
PHASEXA_RESULT=LOCKED or FAILURE
EXPLORATORY_PROBES_CLASSIFIED=YES or NOT_NEEDED
CONTROLLED_FIXTURES_DESIGNED=YES or NO
UNRELATED_FEATURES_EXCLUDED_FROM_ACCEPTED_FIXTURES=YES or NO
READY_FOR_PHASEXB_VALUE_SURFACE_DELTA=YES or NO
READY_FOR_TRITON=NO
```

### Failure policy

If real Triton emits an unrelated staged body feature in a probe, do not broaden the phase. Either design a controlled fixture avoiding that feature or stop and ask whether the user wants to expand scope.

---

## 7. Phase Xb — value-surface delta contract

### Purpose

Lock the additional value-layer surface needed to express the controlled fixture TTIR forms.

### Tasks

1. Update value surface docs and matrices.
2. Classify new value forms as:
   - accepted/lowerable now;
   - surface-admissible but staged;
   - deterministic reject;
   - permanent reject with proof;
   - internal-only.
3. Update the value verifier if needed.
4. Add surface-valid and surface-invalid tests.
5. Add diagnostics for staged/rejected forms.
6. Explicitly list body features excluded from this phase.

### Required outputs

```text
PHASEXB_RESULT=LOCKED or FAILURE
VALUE_SURFACE_DELTA_LOCKED=YES or NO
LOWERABLE_VALUE_FORMS=<list>
STAGED_VALUE_FORMS=<list>
VALUE_VERIFIER_TESTS=PASS or FAIL
READY_FOR_PHASEXC_VALUE_LOWERING=YES or NO
READY_FOR_TRITON=NO
```

### Failure policy

Do not claim value surface support for forms that the value-to-`vc4kernel` path cannot yet lower unless they are explicitly staged and diagnosed.

---

## 8. Phase Xc — value -> vc4kernel implementation and static lock

### Purpose

Implement the actual value-to-`vc4kernel` lowering for the new value forms.

### Tasks

1. Implement conversion using standard MLIR APIs.
2. Add static conversion tests:
   - valid lowerable forms;
   - invalid/staged forms;
   - boundary scans.
3. Ensure output `vc4kernel` contains no producer dialect ops.
4. If the lowering requires `vc4kernel`, `VC4KernelToSSAVC4`, `SSAVC4ToVC4`, artifact, or runtime changes, classify that explicitly.
5. Stop if a lower layer exposes a real blocker outside the prompt scope.

### Required outputs

```text
PHASEXC_RESULT=LOCKED or FAILURE
VALUE_TO_VC4KERNEL_LOWERING=PASS or FAIL
STATIC_TO_SCHEDULED_VC4_PIPELINE=PASS or FAIL
LOWEST_LAYER_TOUCHED=<layer>
READY_FOR_PHASEXD_VALUE_HARDWARE_ISOLATION=YES or NO
READY_FOR_TRITON=NO
```

### Failure policy

Do not workaround lower-layer bugs by reshaping value IR. If the real fix is lower-half, report the exact lower-layer blocker or open a lower-half repair prompt.

---

## 9. Phase Xd — value hardware isolation

### Purpose

Prove the new value feature on real hardware with focused fixtures.

### Fixture rules

Add at least three isolation fixtures unless the prompt explicitly justifies another count.

Each fixture must:

- lower value IR through `vc4kernel -> ssavc4 -> scheduled vc4 -> hardware`;
- use strict CPU oracle;
- use sentinels;
- sweep edge cases;
- use active QPU coverage appropriate to the feature, normally 12 active QPUs for full fixtures;
- preserve intermediate IR and artifacts;
- produce auditable `VC4_TEST_RESULT` lines.

### Required outputs

```text
PHASEXD_RESULT=LOCKED or FAILURE
VALUE_ISOLATION_HARDWARE=PASS or FAIL
VALUE_ISOLATION_FIXTURE_COUNT=<n>
ACTIVE_QPUS=<coverage summary>
READY_FOR_PHASEXE_VALUE_MIXED_ACCEPTANCE=YES or NO
READY_FOR_TRITON=NO
```

### Failure policy

Do not weaken oracle, sentinels, timeout, active QPU count, case count, or expected results to pass. Debug by inspecting all layers.

---

## 10. Phase Xe — value mixed acceptance

### Purpose

Update cumulative value-layer mixed acceptance to include the new value feature interacting with existing supported features.

### Tasks

1. Add one or more mixed fixtures, or strengthen existing mixed fixtures.
2. Update manifest and claims.
3. Run value mixed hardware acceptance.
4. Run layered mixed regressions if lower layers were touched.
5. Audit claims.

### Required outputs

```text
PHASEXE_RESULT=LOCKED or FAILURE
VALUE_MIXED_ACCEPTANCE=PASS or FAIL
VALUE_MIXED_CLAIM_AUDIT=PASS or FAIL
LAYERED_REGRESSION_POLICY_APPLIED=YES or NO
READY_FOR_PHASEXF_TTIR_IMPORTER=YES or NO
READY_FOR_TRITON=NO
```

### Failure policy

Mixed fixture failures are high-value bug reports. Run relevant isolation fixtures, reduce the mixed fixture, and inspect each layer. Do not drop mixed interactions to get acceptance.

---

## 11. Phase Xf — TTIR -> value implementation and static lock

### Purpose

Implement importer support for the controlled TTIR forms now that the required value path is hardware-proven.

### Inputs

- Source-controlled controlled Triton sources.
- Source-controlled emitted TTIR snapshots.
- Manifest/provenance.
- Value hardware proof from Xd/Xe.

### Tasks

1. Implement TTIR importer support using parsed MLIR operations, attrs, types, regions, blocks, and SSA use-defs.
2. Emit only standard value-layer IR.
3. Preserve layer boundaries.
4. Add static importer tests for controlled snapshots.
5. Add staged/reject tests for unsupported adjacent forms.
6. Run full static pipeline:
   `TTIR -> value -> value verifier -> vc4kernel -> ssavc4 -> vc4`.

### Forbidden

- Python semantic lowering.
- Hand-written fake TTIR as accepted proof.
- TTIR text parsing or regex semantics.
- Fixture-name semantics.
- Direct TTIR -> `vc4kernel`.
- Regenerating TTIR unless this prompt is explicitly a generation prompt.

### Required outputs

```text
PHASEXF_RESULT=LOCKED or FAILURE
TTIR_TO_VALUE_LOWERING=PASS or FAIL
TTIR_STATIC_PIPELINE=PASS or FAIL
FRONTEND_ROBUSTNESS_AUDIT=PASS or FAIL
READY_FOR_PHASEXG_TTIR_HARDWARE_ISOLATION=YES or NO
READY_FOR_TRITON=NO
```

### Failure policy

If importer output hits a value/lower-layer gap that should have been covered by the value phase, stop and report the gap. Do not hide missing value support in the importer.

---

## 12. Phase Xg — TTIR hardware isolation

### Purpose

Prove the controlled Triton isolation fixtures on hardware through the full stack.

### Required path

```text
source-controlled Triton .py
  -> source-controlled real .ttir.mlir snapshot
  -> C++ TTIR importer
  -> standard value IR
  -> vc4kernel
  -> ssavc4
  -> scheduled vc4
  -> artifacts/runtime
  -> real hardware
```

### Tasks

1. Add or verify hardware runner support for source-controlled TTIR snapshots.
2. Add TTIR isolation hardware fixtures.
3. Run each fixture on real hardware.
4. Preserve all intermediates.
5. Add claim coverage.

### Required outputs

```text
PHASEXG_RESULT=LOCKED or FAILURE
TTIR_ISOLATION_HARDWARE=PASS or FAIL
TTIR_ISOLATION_FIXTURE_COUNT=<n>
TTIR_ISOLATION_CLAIM_AUDIT=PASS or FAIL
READY_FOR_PHASEXH_TTIR_MIXED_ACCEPTANCE=YES or NO
READY_FOR_TRITON=NO
```

### Failure policy

Do not regenerate TTIR in this phase. Do not fall back to hand-written value IR. Do not weaken hardware proof.

---

## 13. Phase Xh — TTIR mixed acceptance and layered regression

### Purpose

Update cumulative TTIR mixed acceptance to include the new TTIR feature interacting with all existing supported TTIR features.

### Tasks

1. Add or update TTIR mixed fixtures.
2. Update manifests and claims.
3. Run TTIR mixed hardware acceptance.
4. Apply layered mixed regression:
   - identify the lowest layer touched during the phase;
   - run mixed regressions for that layer and every layer above it;
   - run mixed suites only, not every isolation fixture.
5. Run claim audits and source-boundary audits.

### Required outputs

```text
PHASEXH_RESULT=LOCKED or FAILURE
TTIR_MIXED_ACCEPTANCE=PASS or FAIL
TTIR_MIXED_CLAIM_AUDIT=PASS or FAIL
LAYERED_MIXED_REGRESSION=PASS or FAIL
READY_FOR_PHASEXI_FINAL_LOCK=YES or NO
READY_FOR_TRITON=NO
```

### Failure policy

If a mixed fixture resurfaces a bug in an earlier feature, treat it as a real regression. Do not remove the interaction from the mixed fixture unless the feature claim was wrong and is being explicitly staged/retracted.

---

## 14. Phase Xi — final lock and next-phase readiness

### Purpose

Finalize the feature phase with exact accepted/staged/rejected forms, hardware evidence, audits, caveats, and next-phase readiness.

### Tasks

1. Verify all subphase reports.
2. Run final static checks.
3. Run or reuse final hardware according to a documented rule.
4. Run final audits:
   - surface matrix;
   - snapshot provenance;
   - source-boundary;
   - anti-special-casing;
   - claim audits;
   - no-toolchain-drift.
5. Write final lock doc.
6. Write next-phase readiness decision.

### Required outputs

```text
PHASEX_RESULT=LOCKED or BLOCKED or FAILURE
FEATURE_NAME=<name>
CONTROLLED_FIXTURES=YES or NO
VALUE_ISOLATION_HARDWARE=PASS or FAIL
VALUE_MIXED_ACCEPTANCE=PASS or FAIL
TTIR_ISOLATION_HARDWARE=PASS or FAIL
TTIR_MIXED_ACCEPTANCE=PASS or FAIL
LAYERED_MIXED_REGRESSION=PASS or FAIL
FRONTEND_ROBUSTNESS_AUDIT=PASS or FAIL
NO_SPECIAL_CASING_AUDIT=PASS or FAIL
READY_FOR_NEXT_FEATURE=YES or NO_BLOCKED_BY_<reason>
READY_FOR_TRITON=NO
```

### Failure policy

Do not final-lock with hidden caveats that make the next phase unsafe. If a foundational issue remains, report `BLOCKED` or `FAILURE`.

---

## 15. Phase Xx — failure triage and repair prompts

Failure prompts must be narrow and hypothesis-driven.

### Required triage steps

1. Classify the failing layer:
   - controlled source/generator;
   - TTIR parse;
   - importer;
   - value verifier;
   - value-to-`vc4kernel`;
   - `vc4kernel` verifier;
   - `VC4KernelToSSAVC4`;
   - `SSAVC4ToVC4`;
   - artifact/runtime;
   - hardware/oracle.
2. Preserve all artifacts.
3. Reduce only for diagnosis; do not weaken acceptance.
4. Fix the lowest real root cause.
5. Rerun the original unreduced fixture before committing.
6. Run layered mixed regression if a lower layer was touched.

### Required outputs

```text
PHASEXX_RESULT=RESOLVED or BLOCKED
FAILURE_CLASS=<class>
ROOT_CAUSE=<summary>
FIX=<summary>
ORIGINAL_FIXTURE_RERUN=PASS or FAIL
READY_TO_RESUME_PHASE=YES or NO
READY_FOR_TRITON=NO
```

---

## 16. Controlled fixture design requirements

### 16.1 Isolation fixture variety

Isolation fixtures must be aggressive enough to prevent special-casing.

For each new feature, cover:

- minimal case;
- empty/zero case if meaningful;
- one-element/one-iteration case if meaningful;
- many-element/many-iteration case;
- tail/mask edge cases;
- both true/false branch paths where applicable;
- positive/negative/sign edge cases for integer features;
- finite/NaN/Inf policy cases for floating features when in scope;
- spill pressure when realistic;
- 12-QPU full active coverage for hardware acceptance unless explicitly justified.

### 16.2 Mixed fixture variety

Mixed fixtures must combine the new feature with existing feature families. Over time, every supported feature should appear in multiple mixed fixtures and in different contexts.

A mixed fixture should not be a concatenation of isolation snippets. It should look like a real kernel where feature interactions are natural.

### 16.3 Fixture claim discipline

Every `saw_*`, `no_*`, or equivalent claim must be supported by:

```text
CHECKED_OUTPUT
CHECKED_AUDIT
PHASE_GUARD
```

Claims may not be based on:

- fixture names;
- path names;
- comments;
- status strings;
- dead IR;
- decorative operations;
- unverified assumptions.

---

## 17. Anti-special-casing audit

Every feature phase must include a source audit for special casing and brittle semantics.

### Search targets

At minimum, search relevant compiler source for:

```text
std::regex
regex
stringify(Type)
stringify(Attribute)
op->print
type.print
attr.print
contains(
find(
starts_with
ends_with
Operation::remove
fixture names
test names
candidate names
generated path fragments
public argument names used semantically
```

### Classification for hits

Each hit must be classified:

```text
DIAGNOSTIC_ONLY
EXACT_OP_OR_ATTR_NAME
NON_SEMANTIC_METADATA
QUARANTINED_TYPE_ADAPTER
SEMANTIC_STRING_MATCHING_BLOCKER
OWNERSHIP_BLOCKER
FIXTURE_SPECIAL_CASE_BLOCKER
```

### Allowed string use

Allowed:

- exact op-name constants;
- exact dialect namespace checks;
- exact attr-name lookup followed by typed interpretation;
- diagnostics;
- non-semantic provenance metadata.

### Forbidden string use

Forbidden:

- printed TTIR parsing;
- permissive substring semantic matching;
- fixture-name dispatch;
- public argument names deciding memory/tail/type semantics;
- source paths deciding behavior;
- status strings deciding support.

---

## 18. Toolchain and TTIR snapshot policy

Exploratory snapshot generation and controlled snapshot generation are allowed only in phases explicitly scoped for generation.

Implementation, hardware, mixed, and final-lock phases must consume source-controlled snapshots.

Allowed in generation phases:

- use existing pinned generator environment;
- create a small pinned generator environment only if explicitly allowed;
- generate real TTIR from real Triton source;
- record generator command, Triton version, and hash/provenance.

Forbidden outside generation phases:

- create venv;
- pip install Triton;
- clone Triton;
- build Triton;
- build LLVM;
- regenerate TTIR;
- depend on Python semantic lowering.

If regeneration is unavailable but source-controlled snapshots and provenance exist, implementation and hardware phases should proceed using those snapshots.

---

## 19. Scope-control examples

### Example A: control flow phase sees `arith.sitofp`

```text
Feature:
  scalar/coherent control flow

Probe emits:
  scf.for plus arith.sitofp in body

Correct:
  classify probe as SCF_CF plus STAGED_BY_BODY_FEATURE=arith.sitofp;
  write a controlled fixture with the same control-flow shape and supported body ops;
  leave arith.sitofp for a numeric-cast phase.

Incorrect:
  implement sitofp inside the control-flow phase without value isolation,
  TTIR isolation, hardware proof, mixed acceptance, docs, and audits.
```

### Example B: mask classifier phase sees gather

```text
Feature:
  mask legality

Probe emits:
  vector/TTIR gather load

Correct:
  classify gather as staged memory expansion;
  write controlled mask fixtures using contiguous loads/stores;
  schedule gather for a later memory phase.

Incorrect:
  add partial gather lowering merely because one mask probe used it.
```

### Example C: reduction phase sees math.exp

```text
Feature:
  reductions

Probe emits:
  reduction plus exp

Correct:
  classify exp as staged math/SFU policy;
  control the fixture body to avoid exp;
  schedule exp for the math/SFU phase.

Incorrect:
  lower exp approximately without explicit math policy hardware proof.
```

---

## 20. Dynamic package revision protocol

Prompt packages are not immutable once facts arrive.

After early prompts, the user may send back Codex results. The package author should decide whether to:

```text
continue with later prompts unmodified
prepend a small guard
replace a later prompt
insert a repair prompt
regenerate the rest of the package
stop and discuss design ambiguity
```

### Later prompts may proceed unmodified only if:

- all prerequisite lines match exactly;
- controlled fixture set did not change;
- feature scope did not change;
- touched-layer classification did not change;
- no new blocker was found;
- no toolchain/generation dependency leaked into implementation/hardware phases.

### Addenda/prepends are appropriate when:

- a baseline repair must be verified;
- a prompt needs an extra guard against toolchain drift;
- a docs line changed but semantics did not;
- a known caveat must be treated as a hard gate.

### Replacement prompts are required when:

- a fixture emits an unrelated body feature;
- the feature contract changes;
- a lower-layer blocker appears;
- the implementation target changes;
- the scope must split into two phases.

---

## 21. Exit criteria for a feature phase

A feature phase is complete only when:

```text
- controlled fixtures are source-controlled;
- real TTIR snapshots exist for TTIR fixtures;
- value surface delta is documented and tested;
- value -> vc4kernel lowering is implemented and statically tested;
- value isolation hardware passes;
- value mixed acceptance passes;
- TTIR -> value lowering is implemented and statically tested;
- TTIR isolation hardware passes;
- TTIR mixed acceptance passes;
- layered mixed regression passes according to touched layers;
- anti-special-casing audits pass;
- claim audits pass;
- final lock doc lists accepted, staged, and rejected forms;
- next phase readiness is explicit;
- READY_FOR_TRITON remains NO unless explicitly changed by a global phase.
```

If any of these are not true, the phase is not locked.

---

## 22. Relationship to `READY_FOR_TRITON`

Feature phases may prove real Triton/TTIR kernels on hardware without setting
the global Triton-ready flag to YES.

Use:

```text
READY_FOR_NEXT_FEATURE=YES
READY_FOR_PHASE<N+1>=YES
READY_FOR_TRITON=NO
```

until an explicit global Triton readiness/closure phase changes the global gate.

---

## 23. Summary

The project’s upper-stack workflow is:

```text
exploratory probes inform design
controlled fixtures define the phase contract
value support is implemented and hardware-proven
TTIR support is implemented and hardware-proven
mixed fixtures enforce cumulative behavior
audits prevent special-casing
final locks are honest
```

This is the default methodology for future feature phases.
