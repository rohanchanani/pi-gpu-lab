# Copyable Codex Prompts for VC4Tile CuTe-Idiomatic Predication Hardening

These prompts implement the remaining action items after Action Item 1, the semantics document:

```text
compiler/docs/codegen/vc4tile-cute-predication-semantics.md
```

Each prompt assumes that document has been placed in the repo at that path. Every prompt tells Codex to read it first and stay faithful to it.

General policy for all prompts:

```text
Do not implement Triton/IREE/JAX/PyTorch/StableHLO producer lowering.
Do not implement executable sub-32 precision.
Do not create a direct VC4Tile-to-VC4 path.
Do not bypass VC4Tile surface -> canonicalization/copy planning -> VC4Tile core -> SSAVC4 -> scheduled VC4.
Do not special-case fixture names, expected JSON, harness names, public names, or runtime logs.
Do not fake VC4_TEST_RESULT.
Do not alter CPU/reference oracles to match broken output.
Do not add verifier-only comments/literals to satisfy scans.
Hardware is the gold standard. Whenever a patch changes executable lowering behavior, planner decisions that select executable lowering paths, VC4 hardware mapping, runtime-visible scheduling/resource behavior, or hardware fixture expectations, run relevant targeted hardware fixtures in that same step. The fixture set should be focused on the active change; do not run the full matrix unless the prompt/user asks for it. If hardware is skipped, explicitly state why the patch is strictly non-executable metadata, diagnostics, documentation, or verifier-only work.
If a semantic decision is needed that is not already locked by the semantics doc, stop and print: VC4_PREDICATION_NEEDS_SEMANTIC_DECISION
```

---

## Prompt 2 — Predicate Model

```text
You are Codex working on VC4Tile CuTe-idiomatic predication hardening.

Action item: 2 — Predicate Model.

Before editing anything, read this normative document completely:

  compiler/docs/codegen/vc4tile-cute-predication-semantics.md

Your job is to introduce the first-class internal predicate-fragment model described by that document. This stage is about representation, not broad lowering. Do not add ad hoc matcher logic. Do not change hardware semantics. Do not change expected JSON. Do not implement Triton/IREE/JAX/PyTorch producer lowering. Do not implement executable sub-32 precision.

Goal:

Create a reusable VC4Tile predicate model that represents semantic predicates over logical coordinates and distinguishes dense fragmentable predicates from explicit sparse fallback. The model must support at least:

  - full predicate
  - empty predicate
  - logical rank and shape
  - 1D contiguous tail interval
  - 2D rectangular bounds
  - row_fragment(row, start_col, width)
  - row_set_fragment(row_start, row_count, start_col, width)
  - scalar_fragment(row, col)
  - sparse_fragment_set only as explicit fallback
  - inactive destination policy: zero_fill or preserve where needed by copy planning

The primary model must not treat vector<16xi1> as the semantic predicate. A vector mask is only a lowered carrier for a fragment. Preserve the semantics from the doc: predicates are logical-coordinate predicates.

Where to look first:

  - compiler/docs/codegen/vc4tile-cute-predication-semantics.md
  - compiler/include/vc4/Dialect/VC4Tile/IR/VC4TileOps.td
  - compiler/include/vc4/Dialect/VC4Tile/IR/VC4TileAttrs.td
  - compiler/lib/Dialect/VC4Tile/IR/VC4TileOps.cpp
  - compiler/lib/Conversion/VC4TileToSSAVC4/VC4TileToSSAVC4.cpp
  - existing mask/predicate operations: mask_all, tail_mask, tile_bounds_mask, mask_and/mask_or/mask_not if present
  - existing copy planner helpers that currently pattern-match masks

Implementation requirements:

  1. Add the model in the narrowest appropriate place. Prefer internal C++ helper structs/classes in VC4TileToSSAVC4.cpp or a small local helper file if the repo already has a pattern for that. Do not introduce a new public dialect type unless it is clearly required by existing style.
  2. The model must encode semantic class and shape/rank. It must be impossible for a caller to confuse a raw vector lane mask with a semantic coordinate predicate.
  3. Add explicit enum/string/debug names for dense vs sparse fallback classification.
  4. Add helper constructors for full, empty, row fragment, row-set fragment, tail, rectangular bounds, and sparse fallback.
  5. Add lightweight unit/lit coverage that proves the model is reachable through existing surface predicate syntax, but do not attempt to lower every memory path yet.
  6. Preserve all existing passing behavior.

Tests to add or update:

  - A dialect or conversion lit test showing full/empty/tail/rectangular predicate forms roundtrip or are recognized by the normalizer stub.
  - Invalid diagnostic tests for rank/shape mismatch if the existing surface has enough information to verify it now.
  - Hardware is not required only if this remains strictly representation/diagnostic-only and does not change executable lowering, planner-to-lowering routing, hardware mapping, or fixture expectations. If any executable behavior changes, run relevant targeted hardware fixtures.

Verification to run:

  - ninja -C compiler/build vc4-opt
  - ninja -C compiler/build check-vc4
  - targeted VC4Tile dialect/conversion lit tests touched by this patch
  - git diff --check

Expected output:

  - Summarize changed files.
  - State exactly where the predicate model lives.
  - State which semantic classes are represented.
  - State that no producer lowering, no direct VC4Tile-to-VC4 path, and no executable sub-32 precision were added.
  - State which tests passed.

If you find that representing the model requires a semantic choice not already made by compiler/docs/codegen/vc4tile-cute-predication-semantics.md, stop without editing further and print exactly:

  VC4_PREDICATION_NEEDS_SEMANTIC_DECISION
```

---

## Prompt 3 — Predicate Algebra

```text
You are Codex working on VC4Tile CuTe-idiomatic predication hardening.

Action item: 3 — Predicate Algebra.

Before editing anything, read:

  compiler/docs/codegen/vc4tile-cute-predication-semantics.md

Also inspect the Predicate Model implementation from the previous action item. Do not proceed if the first-class predicate model is missing; print VC4_PREDICATION_NEEDS_SEMANTIC_DECISION instead.

Goal:

Implement semantic predicate algebra normalization over logical-coordinate predicates. This must normalize supported boolean combinations into the predicate model, not into ad hoc vector<16xi1> lane masks.

Required supported algebra:

  - true/full
  - false/empty
  - P && true  -> P
  - P || false -> P
  - P && false -> empty
  - P || true  -> full
  - !!P        -> P
  - P && P     -> P
  - P || P     -> P
  - commuted forms for supported binary ops
  - nested associative forms like (A && B) && C and A && (B && C)
  - bounds AND tail
  - tail AND bounds
  - bounds OR tail when normalizable to finite dense fragments
  - tail AND NOT(bounds) when normalizable to finite dense fragments or explicit sparse fallback
  - rect AND bounds
  - rect AND tail

Sparse fallback policy:

  - If an expression cannot normalize to dense fragments but can be represented as explicit sparse_fallback, classify it as sparse_fallback.
  - If sparse fallback is not allowed by the caller/consumer, produce a deterministic diagnostic.
  - Do not silently lower unrecognized algebra to per-lane stores.

Where to look:

  - compiler/docs/codegen/vc4tile-cute-predication-semantics.md
  - predicate model implementation from Action Item 2
  - existing mask ops and verifier code in VC4TileOps.cpp
  - existing copy planner helpers in VC4TileToSSAVC4.cpp
  - existing conversion tests for mask_all, tail_mask, tile_bounds_mask, mask_and/mask_or/mask_not, or equivalent ops

Implementation requirements:

  1. Add a normalizer function/API that takes predicate-producing operations or already-modeled predicates and returns the canonical predicate model.
  2. Normalize based on semantics, not only exact operation spelling. At minimum, support commuted and nested forms for the known ops.
  3. Keep the result explicitly classified: dense, empty, full, row fragments, row-set fragments, scalar fragments, or sparse_fallback.
  4. Emit deterministic diagnostics for unsupported algebra. The diagnostic must describe the semantic reason, e.g. cannot normalize into dense fragments and sparse fallback disabled.
  5. Keep this stage focused on predicate algebra. Do not broadly rewrite all memory lowering yet.

Tests to add or update:

  - Lit tests for each simplification rule.
  - Lit tests for commuted bounds/tail forms.
  - Lit tests for nested AND/OR forms.
  - Invalid diagnostic tests for unsupported algebra with sparse fallback disabled.
  - Tests must prove semantic predicate algebra disappears before VC4Tile core/SSAVC4 if this stage connects to planning; otherwise prove the normalizer emits the expected planned representation or deterministic diagnostic.

Verification to run:

  - ninja -C compiler/build vc4-opt
  - ninja -C compiler/build check-vc4
  - targeted lit tests for predication algebra
  - git diff --check

Expected output:

  - Summarize changed files.
  - List supported algebra forms.
  - List unsupported forms and their diagnostics.
  - State that vector<16xi1> is not the primary semantic predicate model.
  - State that no producer lowering, no direct VC4Tile-to-VC4, and no executable sub-32 paths were added.
  - State tests run and results.

If this requires semantics not locked by the document, stop and print:

  VC4_PREDICATION_NEEDS_SEMANTIC_DECISION
```

---

## Prompt 4 — Fragment Planner

```text
You are Codex working on VC4Tile CuTe-idiomatic predication hardening.

Action item: 4 — Fragment Planner.

Before editing anything, read:

  compiler/docs/codegen/vc4tile-cute-predication-semantics.md

Also inspect the Predicate Model and Predicate Algebra implementations from prior action items. Do not proceed if either is missing.

Goal:

Create the canonical predicate-to-transfer-fragments planner API. This planner must be the single reusable path that memory and compute consumers will eventually use. It should replace ad hoc per-memory-path row/tail matching over time.

Required planner behavior:

  - Input: normalized semantic predicate + consumer view/layout/path information + inactive policy.
  - Output: typed fragment plan.
  - Plan classes must include:
      full_block
      empty
      row_fragment
      row_set_fragment
      scalar_fragment
      sparse_fallback
      unsupported_with_reason
  - The plan must preserve predicate provenance for diagnostics.
  - The plan must identify whether the fragment is dense or sparse fallback.
  - The plan must identify load/store inactive semantics: zero_fill or preserve.
  - The plan must not choose a VC4 hardware mechanism that requires unsupported dynamic 2D masks or dynamic VPM alignment.

Where to look:

  - compiler/lib/Conversion/VC4TileToSSAVC4/VC4TileToSSAVC4.cpp
  - existing copy planner helpers
  - existing tile_load/tile_store/copy_tile lowering
  - existing VDR/VDW/VPM helper code
  - existing tests named around plan-copy, tail, affine, shared, transpose, or predicated copy

Implementation requirements:

  1. Add a reusable planner function/API. Prefer local C++ helpers in the conversion/copy-planner implementation unless existing repo architecture suggests a helper library.
  2. The planner must accept a consumer/path descriptor instead of inspecting only op names.
  3. The planner must classify paths clearly: dense fast path, row-tail path, guarded row skip, sparse fallback, unsupported.
  4. Existing ad hoc helper paths may remain temporarily, but new code must call the planner where possible.
  5. Add clear TODOs only for future migration points; do not leave behavior ambiguous.
  6. Do not change compute semantics or hardware oracles in this stage unless required by planner integration.

Tests:

  - Conversion lit tests proving dense full-tile predicates plan to full_block/full path.
  - Conversion lit tests proving row/tail predicates plan to row_fragment/row_set fragments.
  - Conversion lit tests proving unsupported sparse masks produce explicit sparse_fallback or diagnostic, not silent lane-store fallback.
  - Existing relevant hardware tests must run and pass if planner integration changes executable lowering behavior or routes an existing predicate to a different executable path.

Verification:

  - ninja -C compiler/build vc4-opt
  - ninja -C compiler/build check-vc4
  - targeted plan-copy / predication conversion lit tests
  - targeted hardware fixtures if any executable lowering behavior or planner-to-lowering routing changed
  - git diff --check

Expected output:

  - Summarize changed files.
  - Identify the planner API and where it lives.
  - List fragment plan classes implemented.
  - State which existing ad hoc paths still remain and which now use the planner.
  - State tests run and results.

If you cannot decide how a fragment should map without a new semantic decision, stop and print:

  VC4_PREDICATION_NEEDS_SEMANTIC_DECISION
```

---

## Prompt 5 — Verifier Work

```text
You are Codex working on VC4Tile CuTe-idiomatic predication hardening.

Action item: 5 — Verifier Work.

Read first:

  compiler/docs/codegen/vc4tile-cute-predication-semantics.md

Goal:

Add verifier enforcement for the predicate semantics now represented by the predicate model/algebra/planner. This stage must reject ambiguous or unsupported predicate uses before lowering and must verify that no semantic predicate survives past the intended boundary.

Required verifier rules before planning:

  - predicate rank matches consumer rank
  - predicate tile shape matches consumer tile shape
  - predicate/view/layout association is explicit
  - dynamic bounds are scalar i32 where required
  - inactive destination policy is explicit or defaultable by semantic role
  - sparse fallback eligibility is explicit
  - executable sub-32 predicated paths reject deterministically

Required verifier rules after planning / before SSAVC4 conversion:

  - no semantic predicate algebra remains past planning
  - no unplanned surface predicate/copy ops remain before SSAVC4 conversion
  - all fragments are hardware-legal for their memory path/layout
  - dynamic active-lane use is one-row only unless separately proven
  - dynamic VPM alignment is not required or has been statically split
  - VPM rows/bytes and barrier resources are in bounds

Where to look:

  - compiler/lib/Dialect/VC4Tile/IR/VC4TileOps.cpp
  - compiler/lib/Conversion/VC4TileToSSAVC4/VC4TileToSSAVC4.cpp
  - --verify-vc4tile-core implementation
  - --canonicalize-vc4tile-surface / --plan-vc4tile-copies / --convert-vc4tile-to-ssavc4 pass boundaries
  - existing invalid tests under compiler/test/Dialect/VC4Tile and compiler/test/Conversion/VC4TileToSSAVC4

Implementation requirements:

  1. Add deterministic verifier checks exactly at the semantic boundary where they belong.
  2. Do not make the core verifier reject legal core fragments just because they originated from predicates.
  3. Do make SSAVC4 conversion reject unplanned semantic predicate/surface operations with a clear ordering diagnostic.
  4. Diagnostics must explain semantic reasons, not vague unsupported messages.
  5. Do not weaken existing core verifier checks.

Tests:

  - Invalid tests for rank mismatch.
  - Invalid tests for shape mismatch.
  - Invalid tests for missing inactive policy where required.
  - Invalid tests for sparse fallback disabled.
  - Invalid tests for unplanned surface predicates reaching SSAVC4 conversion.
  - Invalid tests for dynamic multi-row VDW active lanes if such a path can be expressed.

Verification:

  - ninja -C compiler/build vc4-opt
  - ninja -C compiler/build check-vc4
  - targeted invalid diagnostics lit subset
  - git diff --check

Expected output:

  - Summarize changed files.
  - List verifier rules added.
  - List invalid tests added.
  - State no producer lowering/direct VC4Tile-to-VC4/sub-32 executable paths were added.
  - State tests run and results.

If a verifier rule needs a semantic choice not already specified, stop and print:

  VC4_PREDICATION_NEEDS_SEMANTIC_DECISION
```

---

## Prompt 6 — Diagnostics

```text
You are Codex working on VC4Tile CuTe-idiomatic predication hardening.

Action item: 6 — Diagnostics.

Read first:

  compiler/docs/codegen/vc4tile-cute-predication-semantics.md

Goal:

Improve diagnostics for predicate normalization/planning/verification so unsupported patterns fail with semantic reasons. This is not a broad feature implementation stage. Do not change hardware semantics or oracles.

Required diagnostic classes:

  - unsupported predicate algebra
  - unsupported layout + predicate combination
  - non-fragmentable sparse mask with sparse fallback disabled
  - dynamic VPM alignment required but unsupported
  - unsupported 2D dynamic active-lane predicate
  - store predicate requires inactive destination preservation but path would overwrite inactive elements
  - load predicate requires inactive zero-fill but path cannot provide it
  - sub-32 predicated executable path rejected by M5 policy

Diagnostic style:

  - Say what semantic requirement failed.
  - Mention whether dense fragment planning failed or sparse fallback was disabled.
  - Mention the consumer/path if known: tile_load, tile_store, copy_tile, shared/register/global, tile_contract, etc.
  - Do not use vague messages like “unsupported currently” when a precise reason is available.

Where to look:

  - Predicate normalizer/planner diagnostics
  - VC4Tile op verifiers
  - --verify-vc4tile-core diagnostics
  - --plan-vc4tile-copies diagnostics
  - --convert-vc4tile-to-ssavc4 ordering diagnostics
  - invalid tests added in prior stages

Implementation requirements:

  1. Centralize diagnostic text where reasonable so equivalent failures produce consistent wording.
  2. Update invalid tests to check meaningful substrings.
  3. Do not add comments or source literals solely to satisfy verifier scans.
  4. Do not change pass ordering or semantics.

Tests:

  - For each diagnostic class above, add or update one invalid lit test if expressible.
  - Tests should check semantic substrings such as “sparse fallback disabled”, “inactive destination preservation”, “zero-fill”, “dynamic VPM alignment”, “32-bit”, etc.

Verification:

  - ninja -C compiler/build vc4-opt
  - ninja -C compiler/build check-vc4
  - targeted invalid diagnostics lit tests
  - git diff --check

Expected output:

  - Summarize changed files.
  - List diagnostic classes covered.
  - List tests added/updated.
  - State tests run and results.

If you discover a missing semantic decision, stop and print:

  VC4_PREDICATION_NEEDS_SEMANTIC_DECISION
```

---

## Prompt 7 — VC4 Hardware Mapping

```text
You are Codex working on VC4Tile CuTe-idiomatic predication hardening.

Action item: 7 — VC4 Hardware Mapping.

Read first:

  compiler/docs/codegen/vc4tile-cute-predication-semantics.md
  compiler/docs/codegen/vc4-as-cuda-mapping-guide.md
  VideoCore IV Architecture Reference Guide material already present in repo/context if available

Goal:

Map normalized predicate fragments to legal VC4 mechanisms, constrained by the semantics document. This stage should implement or tighten the planner-to-core lowering for dense full/row/tail fragments and explicit fallback classification. It must not fake unsupported dynamic 2D masking.

Required mapping hierarchy:

  - full static rectangular fragment -> block VDR/VDW/VPM mechanisms where legal
  - static row fragment -> row VPM/VDW/VDR or normal vector path
  - dynamic row tail -> one-row dynamic active-lane style lowering where already proven
  - row skip -> branch/guarded control flow
  - sparse predicate -> explicit sparse fallback only if allowed
  - dynamic arbitrary 2D predicate -> reject unless normalized into supported fragments

VC4 constraints to respect:

  - 16 lanes per QPU warp
  - user-visible VPM is 64 rows x 16 32-bit words
  - VDR/VCD/VDW/VPM modes are structured, not arbitrary dynamic mask tensors
  - dynamic active lanes are one-row/tail only unless separately proven
  - dynamic VPM alignment must not be faked

Where to look:

  - compiler/lib/Conversion/VC4TileToSSAVC4/VC4TileToSSAVC4.cpp
  - SSAVC4 VDR/VCD/VDW support in SSAVC4ToVC4
  - existing VPM/VDW/VDR tests and hardware fixtures
  - copy planner tests from M5
  - shared transpose and VDR load fixtures

Implementation requirements:

  1. Wire the fragment planner's dense fragment classes into actual existing VC4Tile core/SSAVC4-compatible lowering paths.
  2. Preserve inactive load zero-fill and inactive store preservation semantics.
  3. Do not add arbitrary 2D mask lowering.
  4. If a dense fragment cannot legally map to hardware, diagnose rather than approximate.
  5. If sparse fallback is implemented, label it explicitly in the plan/lowering/test output where possible.

Tests:

  - Lit tests proving full static rectangle uses dense/block path where expected.
  - Lit tests proving row-tail uses row/dynamic-active-lane path.
  - Lit tests proving arbitrary 2D dynamic masks reject.
  - Hardware tests are required whenever this changes executable behavior or routes a predicate fragment to a different executable path. Every store fixture needs sentinel preservation and every load fixture needs zero-fill checks.

Verification:

  - ninja -C compiler/build vc4-opt
  - ninja -C compiler/build vc4-codegen
  - ninja -C compiler/build check-vc4
  - targeted hardware fixture(s) whenever executable lowering or planner-to-hardware routing changed
  - git diff --check

Expected output:

  - Summarize changed files.
  - List fragment classes mapped to hardware.
  - List unsupported classes and diagnostics.
  - State hardware fixtures run, or explicitly state why the patch was strictly non-executable.
  - State no unsupported dynamic 2D mask lowering was added.

If hardware behavior is uncertain and requires characterization, stop and print:

  VC4_PREDICATION_NEEDS_HARDWARE_CHARACTERIZATION
```

---

## Prompt 8 — Memory Path Coverage

```text
You are Codex working on VC4Tile CuTe-idiomatic predication hardening.

Action item: 8 — Memory Path Coverage.

Read first:

  compiler/docs/codegen/vc4tile-cute-predication-semantics.md

Goal:

Apply the shared predicate fragment planner to all relevant VC4Tile memory paths. The key objective is that every memory consumer uses the same semantic predicate model and inactive-lane policies, rather than duplicating ad hoc matching.

Paths to cover:

  - global -> register
  - register -> global
  - global -> shared VPM
  - shared VPM -> global
  - register -> shared VPM
  - shared VPM -> register
  - shared -> shared views/copies only if they are already part of the VC4Tile surface

Required inactive semantics:

  - loads into register: inactive = zero-fill, no inactive read
  - stores to global: inactive destination preserved, no inactive write
  - global -> shared for compute staging: inactive shared destination zero-fill by default
  - register -> shared: inactive shared destination preserve by default unless zero_fill explicitly requested
  - shared -> global: inactive global destination preserved
  - shared -> register: inactive register result zero-filled

Where to look:

  - tile_load/tile_store/copy_tile lowering
  - shared_tile_alloc/shared_load/shared_store paths
  - VDR/VCD global-to-VPM load support
  - VDW VPM-to-global store support
  - existing hardware fixtures: tile_load_store, register_shared_roundtrip, shared_register_roundtrip, shared_tile_roundtrip, global_to_shared_to_global, shared_transpose

Implementation requirements:

  1. Route each memory path through the common planner, even if some paths only support a subset and reject the rest.
  2. Preserve correct zero-fill/preserve semantics for inactive elements.
  3. Reject unsupported sparse/layout combinations with diagnostics from Action Item 6.
  4. Do not degrade full dense cases into sparse fallback.
  5. Keep M5 32-bit-only executable policy.

Tests:

  - Lit tests for each memory path showing planner output/core lowering.
  - Invalid tests for unsupported predicate/layout combinations per path.
  - Hardware tests for each executable path touched:
      global/register masks
      register/global masks
      global/shared masks
      shared/global masks
      register/shared masks
      shared/register masks
  - Every predicated store fixture must check sentinels.
  - Every predicated load fixture must check zero-fill.

Verification:

  - ninja -C compiler/build vc4-opt
  - ninja -C compiler/build vc4-codegen
  - ninja -C compiler/build check-vc4
  - targeted hardware fixtures for all changed executable memory paths
  - git diff --check

Expected output:

  - Summarize changed files.
  - List memory paths now using the common planner.
  - List memory paths still unsupported and their deterministic diagnostics.
  - List hardware fixtures run and results.

If a path requires unproven VC4 behavior, stop and print:

  VC4_PREDICATION_NEEDS_HARDWARE_CHARACTERIZATION
```

---

## Prompt 9 — Layout Coverage

```text
You are Codex working on VC4Tile CuTe-idiomatic predication hardening.

Action item: 9 — Layout Coverage.

Read first:

  compiler/docs/codegen/vc4tile-cute-predication-semantics.md

Goal:

Extend predicate planning across supported layouts while preserving the locked rule that predicate truth is over logical coordinates and layout affects only address mapping and fragment splitting.

Layouts to consider in order:

  - row_major 1D / 1x16
  - row_major 2D / 4x4
  - pitched row-major
  - affine 2D views already supported by M5
  - transposed views already supported by M5
  - column-major only where VC4 VPM/VDW/VDR modes actually support it

Required rule:

  P(row, col) is interpreted in logical coordinates of the view being consumed. A transposed or affine view must explicitly rebase or reinterpret coordinates through the view mapping. Do not silently swap row/col because it makes a test pass.

Where to look:

  - VC4Tile layout attrs/types
  - tile_view / transpose_view implementation
  - copy planner layout code
  - plan-copy-affine-stride, plan-transpose-view, shared transpose tests
  - hardware fixtures for 2D row-major, affine stride, shared transpose, and any new predicated transposed tests

Implementation requirements:

  1. Add layout-aware fragment splitting to the planner for layouts that are already supported.
  2. For unsupported layout + predicate combinations, emit deterministic diagnostics.
  3. Do not fake dynamic VPM alignment.
  4. For transposed/affine views, ensure predicate congruence is explicit and logical-coordinate preserving.
  5. Do not add column-major executable behavior unless hardware mapping is clearly supported and tested.

Tests:

  - Lit tests for row-major and pitched row-major predicate fragment planning.
  - Lit tests for affine 2D predicate planning or deterministic rejection.
  - Lit tests for transposed view predicate planning or deterministic rejection.
  - Hardware tests are required wherever executable mapping is implemented or changed; include sentinel/zero-fill checks.

Verification:

  - ninja -C compiler/build vc4-opt
  - ninja -C compiler/build vc4-codegen
  - ninja -C compiler/build check-vc4
  - targeted hardware fixtures for any new or changed executable layout path
  - git diff --check

Expected output:

  - Summarize changed files.
  - List layouts supported with predicates.
  - List layouts rejected and diagnostic reasons.
  - State whether column-major remains unsupported or was hardware-tested.
  - State tests run and results.

If layout mapping requires an unproven hardware behavior, stop and print:

  VC4_PREDICATION_NEEDS_HARDWARE_CHARACTERIZATION
```

---

## Prompt 10 — Compute Consumer Coverage

```text
You are Codex working on VC4Tile CuTe-idiomatic predication hardening.

Action item: 10 — Compute Consumer Coverage.

Read first:

  compiler/docs/codegen/vc4tile-cute-predication-semantics.md
  compiler/docs/codegen/vc4tile-m5-compute-primitives-design.md

Goal:

Apply normalized predicate semantics to compute consumers: tile_select, tile_reduce, tile_dot, tile_contract, and tile_matmul. The important rule is that compute predicates are semantic coordinate predicates, not accidental lane masks.

Required semantics:

  - tile_select: result(coord) = P(coord) ? true_value(coord) : false_value(coord)
  - sum/add reductions: inactive coords contribute zero
  - max/min reductions: inactive coords use identity or are skipped; all-empty behavior must be defined or rejected
  - tile_dot/tile_contract/tile_matmul K-tail inputs: inactive K coords contribute zero
  - tile_contract/tile_matmul M/N-tail outputs: inactive output coords preserve destination on store
  - no stale accumulator values may leak into inactive output stores
  - matrix predicates must be per-output-coordinate predicates; do not use dot/broadcast shortcuts for matrix tails

Where to look:

  - tile_elementwise lowering
  - tile_reduce lowering
  - tile_dot/tile_contract/tile_matmul lowering
  - M5-11 repaired matmul/contract implementation
  - existing hardware fixtures for elementwise, reductions, dot, matmul, contract, 8x8x8 stress if present

Implementation requirements:

  1. Make compute consumers call the predicate normalizer/planner where applicable.
  2. Implement only the compute predicate classes whose semantics are locked and whose hardware/core lowering is supported.
  3. Reject unsupported sparse compute predicates unless an explicit fallback is designed and tested.
  4. Preserve 32-bit-only executable policy.
  5. Do not regress the repaired M5-11 true per-output matmul/contract semantics.

Tests:

  - Lit tests for tile_select with normalized predicates.
  - Lit tests for tile_reduce with tails and empty/all cases as defined.
  - Lit tests for tile_contract/matmul K-tail zero-fill and M/N output preservation.
  - Hardware tests for at least:
      predicated tile_select tail
      predicated tile_reduce tail
      matmul/contract K-tail zero-fill
      matmul/contract M/N-tail output sentinel preservation

Verification:

  - ninja -C compiler/build vc4-opt
  - ninja -C compiler/build vc4-codegen
  - ninja -C compiler/build check-vc4
  - targeted hardware fixtures for compute consumers touched
  - M5-11 dot/matmul/contract hardware fixtures if those paths were touched
  - git diff --check

Expected output:

  - Summarize changed files.
  - List compute consumers covered.
  - List unsupported compute predicate forms and diagnostics.
  - State M5-11 matmul/contract repair remains intact.
  - List hardware fixtures run and results.

If a compute identity or all-empty behavior is not already defined, stop and print:

  VC4_PREDICATION_NEEDS_SEMANTIC_DECISION
```

---

## Prompt 11 — Tests

```text
You are Codex working on VC4Tile CuTe-idiomatic predication hardening.

Action item: 11 — Tests.

Read first:

  compiler/docs/codegen/vc4tile-cute-predication-semantics.md

Goal:

Build the full test matrix for CuTe-idiomatic predication. This stage should not invent new semantics. It should verify the implementation created by prior action items and expose any shortcuts or gaps.

Required test categories:

  - Lit tests for each normalized predicate form.
  - Lit tests proving semantic predicates disappear before SSAVC4 conversion.
  - Lit tests proving full tiles stay on dense/full-block fast paths.
  - Lit tests proving tail tiles split into row fragments.
  - Lit tests proving sparse fallback is explicit when implemented.
  - Invalid tests for unsupported algebra/layout combinations.
  - Hardware tests for global/register masks.
  - Hardware tests for register/global masks.
  - Hardware tests for global/shared masks.
  - Hardware tests for shared/global masks.
  - Hardware tests for register/shared masks.
  - Hardware tests for shared/register masks.
  - Hardware tests for mixed row/column/K tails where supported.
  - Hardware tests for pitched row stores where supported.
  - Hardware tests for transposed/affine layouts where supported.
  - Sentinel checks on every predicated store fixture.
  - Zero-fill checks on every predicated load fixture.

Test integrity requirements:

  - Harnesses must compare actual device output to CPU oracle.
  - Harnesses must not compute the output on host and then claim device success.
  - Do not print fixed VC4_TEST_RESULT status=PASS.
  - Do not weaken expected.json.
  - Do not remove difficult cases to make tests pass.
  - Hardware fixtures must use the real candidate generation/run path.

Where to look:

  - compiler/test/CodeGen/VC4Tile/Hardware/Run/*
  - compiler/test/CodeGen/VC4Tile/Support/run_vc4tile_candidate_codegen_test.sh
  - existing expected.json style
  - existing predication and copy planner tests
  - M5 final acceptance fixture matrices

Implementation requirements:

  1. Add tests incrementally but comprehensively.
  2. For every executable semantic class, include hardware proof unless the path is documented as non-executable/metadata-only.
  3. Use varied sizes, including larger CuTe-like tiled cases, not just minimal 1x16 and 4x4 tests.
  4. Include edge cases: zero active, one active, exact full, non-divisible tail, row tail, column tail, K tail, and sentinel preservation.
  5. Add tests to the appropriate lit/milestone matrices if the repo uses declarative verifier matrices for this work.

Verification:

  - ninja -C compiler/build vc4-opt
  - ninja -C compiler/build vc4-codegen
  - ninja -C compiler/build check-vc4
  - run every new hardware fixture on real hardware
  - git diff --check

Expected output:

  - Summarize tests added by category.
  - List every hardware fixture added and exactly what it proves.
  - State sentinel/zero-fill coverage.
  - State commands run and results.

If an expected test cannot be written because the feature is not implemented, do not fake it. Add a deterministic invalid/unsupported test if appropriate and report the gap.
```

---

## Prompt 12 — Cleanup / Architecture

```text
You are Codex working on VC4Tile CuTe-idiomatic predication hardening.

Action item: 12 — Cleanup / Architecture.

Read first:

  compiler/docs/codegen/vc4tile-cute-predication-semantics.md

Goal:

Remove or quarantine old ad hoc predicate handling paths after the shared model/algebra/planner has been installed and covered by tests. This is cleanup, not a semantic expansion stage.

Required cleanup:

  - Remove duplicated row-tail matching where the common planner now handles it.
  - Rename fallback paths to make sparse fallback explicit.
  - Consolidate duplicated fragment lowering helpers.
  - Ensure diagnostics come from common normalizer/planner/verifier helpers.
  - Ensure comments/documentation reference the normative semantics doc.
  - Update M5 design docs if they still describe ad hoc predicate behavior.
  - Ensure no semantic predicate survives into SSAVC4 conversion.

Where to look:

  - compiler/lib/Conversion/VC4TileToSSAVC4/VC4TileToSSAVC4.cpp
  - compiler/lib/Dialect/VC4Tile/IR/VC4TileOps.cpp
  - compiler/docs/codegen/vc4tile-m5-full-design.md
  - compiler/docs/codegen/vc4tile-m5-copy-planner-design.md
  - all tests added in prior predication stages

Implementation requirements:

  1. Preserve behavior and tests.
  2. Do not delete fallback functionality unless it is replaced by the common planner.
  3. Do not hide unsupported behavior behind vague TODOs.
  4. Do not make broad unrelated refactors.
  5. Keep the pipeline path intact.

Tests/verification:

  - ninja -C compiler/build vc4-opt
  - ninja -C compiler/build vc4-codegen
  - ninja -C compiler/build check-vc4
  - run targeted predication lit tests
  - run targeted predication hardware fixtures if executable lowering code moved or if behavior/routing of an executable path changed
  - git diff --check

Expected output:

  - Summarize old paths removed or quarantined.
  - List any ad hoc paths intentionally left and why.
  - State tests run and results.
  - State no behavior/oracle/expected JSON was weakened.

If cleanup would require changing semantics, stop and print:

  VC4_PREDICATION_NEEDS_SEMANTIC_DECISION
```

---

## Prompt 13 — Hardware Characterization

```text
You are Codex working on VC4Tile CuTe-idiomatic predication hardening.

Action item: 13 — Hardware Characterization.

Read first:

  compiler/docs/codegen/vc4tile-cute-predication-semantics.md
  compiler/docs/codegen/vc4-as-cuda-mapping-guide.md
  any available VideoCore IV architecture guide notes in the repo/context

Goal:

Design and run focused hardware characterization probes for predicate behaviors that are not yet proven. This stage must not assume hardware behavior. It must either prove behavior on real hardware or leave the corresponding compiler path unsupported with deterministic diagnostics.

Characterization areas:

  - vertical/column-major predicated stores with dynamic widths
  - transposed shared VPM views under tail masks
  - VDR predicated row loads with pitched/global strides
  - mixed shared/global copies under repeated launches
  - resource pressure when many guarded fragments are emitted
  - branch-heavy sparse fallback safe limits
  - any dynamic VPM alignment behavior needed by layout coverage
  - any multi-row dynamic active-lane behavior someone wants to rely on

Rules:

  - Do not implement a compiler fast path before the hardware behavior is proven.
  - Probes must run on real VC4 hardware through the existing runtime path.
  - Probes must have CPU oracles and sentinel/zero-fill checks where relevant.
  - If a probe shows behavior is unsafe/uncertain, update diagnostics/tests to reject the corresponding pattern.
  - Do not create fixed PASS scripts.
  - Do not modify expected JSON to hide failures.

Where to look:

  - existing hardware characterization fixtures
  - vpm_slice_visibility / vpm_setup_clobber style fixtures if present
  - VC4Tile hardware candidate runner
  - SSAVC4/VC4 VPM/VDR/VDW tests
  - compiler/docs/codegen/vc4-as-cuda-mapping-guide.md

Implementation requirements:

  1. Add small, focused characterization fixtures, not broad production kernels.
  2. Each fixture should prove exactly one hardware behavior.
  3. Each fixture should emit enough logs to diagnose failure without ambiguity.
  4. Keep characterization separate from semantic lowering tests when possible.
  5. Update docs/diagnostics only based on proven results.

Verification:

  - ninja -C compiler/build vc4-codegen
  - ninja -C compiler/build check-vc4
  - run every new hardware characterization fixture on real hardware, with a power-cycle before hardware test batches if that is the current lab requirement
  - git diff --check

Expected output:

  - List every characterization fixture added.
  - For each fixture, state the exact hardware behavior tested.
  - For each fixture, report PASS/FAIL and what that means for compiler legality.
  - List any compiler patterns that should remain unsupported based on results.

If the test design would require changing compiler semantics rather than only probing hardware, stop and print:

  VC4_PREDICATION_NEEDS_SEMANTIC_DECISION
```
